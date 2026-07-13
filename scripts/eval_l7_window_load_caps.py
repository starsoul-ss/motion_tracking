#!/usr/bin/env python3
"""Measure the maximum downward two-hand load for every motion window."""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils.l7_load_eval import (
    make_eval_env,
    merge_reset,
    reset_envs,
    set_load,
    window_bounds,
)


@dataclass
class State:
    active: bool = False
    motion_id: int = -1
    window_idx: int = 0
    load_kg: float = 0.0
    attempts: int = 0
    failed_loads: list[float] = field(default_factory=list)
    failure_phases: list[str] = field(default_factory=list)
    failed_actual_loads: list[float] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    root_error_max: float = 0.0
    keypoint_error_max: float = 0.0
    joint_error_max: float = 0.0
    tracking_bad_steps: int = 0
    target_hold_steps: int = 0
    reached_target_frame: int = -1
    confirming: bool = False
    confirmation_successes: int = 0
    confirmation_failures: int = 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--ramp-sec", type=float, default=1.0)
    parser.add_argument("--max-load-kg", type=float, default=30.0)
    parser.add_argument("--load-step-kg", type=float, default=5.0)
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--task-shard-count", type=int, default=1)
    parser.add_argument("--motion-ids-file")
    parser.add_argument("--root-error-threshold-m", type=float, default=0.6)
    parser.add_argument("--keypoint-error-threshold-m", type=float, default=0.3)
    parser.add_argument("--joint-error-threshold-rad", type=float, default=0.5)
    parser.add_argument("--tracking-error-grace-steps", type=int, default=5)
    parser.add_argument("--load-tolerance-kg", type=float, default=0.05)
    parser.add_argument("--confirm-boundary-successes", action="store_true")
    parser.add_argument("--confirm-error-fraction", type=float, default=0.8)
    args = parser.parse_args()
    if args.max_load_kg < 0.0 or args.load_step_kg <= 0.0:
        parser.error("--max-load-kg must be nonnegative and --load-step-kg must be positive")
    if args.tracking_error_grace_steps <= 0 or args.load_tolerance_kg < 0.0 or not 0.0 < args.confirm_error_fraction <= 1.0:
        parser.error("--tracking-error-grace-steps must be positive and --load-tolerance-kg nonnegative")
    return args


def write_outputs(prefix: Path, rows: list[dict]):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "motion_id",
        "dataset_name",
        "teacher_mem_path",
        "student_mem_path",
        "local_motion_id",
        "source_path",
        "window_idx",
        "expected_window_count",
        "window_start_frame",
        "window_end_frame",
        "max_success_load_kg",
        "attempts",
        "failed_loads_kg",
        "failure_phases",
        "failed_actual_loads_kg",
        "failure_reasons",
        "status",
        "fail_frame",
        "termination_reasons",
        "actual_load_kg",
        "root_error_max_m",
        "keypoint_error_max_m",
        "joint_error_max_rad",
        "target_hold_steps",
        "required_hold_steps",
        "reached_target_frame",
        "confirmed_success",
        "confirmation_successes",
        "confirmation_failures",
    ]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    args = parse_args()
    env, agent, base_env, scheduler = make_eval_env(
        args.cfg_path, args.checkpoint_path, args.num_envs, args.max_load_kg
    )
    command = base_env.command_manager
    lengths = command.dataset.global_lengths.detach().cpu().tolist()
    source_paths = command.dataset.motion_source_paths
    motion_labels = command.dataset.motion_labels
    dataset_ids = command.dataset.motion_to_dataset_id.detach().cpu().tolist()
    dataset_counts = command.dataset.teacher_dataset.dataset_counts.detach().cpu().tolist()
    dataset_offsets = [0]
    for count in dataset_counts[:-1]:
        dataset_offsets.append(dataset_offsets[-1] + int(count))
    window_steps = max(1, round(args.window_sec / base_env.step_dt))
    ramp_steps = max(1, round(args.ramp_sec / base_env.step_dt))
    windows = [window_bounds(length, window_steps) for length in lengths]
    selected_motion_ids = None
    if args.motion_ids_file:
        selected_motion_ids = json.loads(Path(args.motion_ids_file).read_text())
        if isinstance(selected_motion_ids, dict):
            selected_motion_ids = selected_motion_ids["global_motion_ids"]
        selected_motion_ids = set(map(int, selected_motion_ids))
    motion_ids = [
        motion_id
        for motion_id in range(len(lengths))
        if motion_id % args.task_shard_count == args.task_shard_index
        and (selected_motion_ids is None or motion_id in selected_motion_ids)
    ]
    motion_ids.sort(key=lambda motion_id: len(windows[motion_id]), reverse=True)

    states = [State() for _ in range(base_env.num_envs)]
    rows = []
    cursor = 0

    def reset_attempt_metrics(state: State):
        state.root_error_max = 0.0
        state.keypoint_error_max = 0.0
        state.joint_error_max = 0.0
        state.tracking_bad_steps = 0
        state.target_hold_steps = 0
        state.reached_target_frame = -1

    def reset_window_state(state: State):
        state.load_kg = args.max_load_kg
        state.attempts = 1
        state.failed_loads.clear()
        state.failure_phases.clear()
        state.failed_actual_loads.clear()
        state.failure_reasons.clear()
        state.confirming = False
        state.confirmation_successes = 0
        state.confirmation_failures = 0
        reset_attempt_metrics(state)

    def attempt_start(state: State):
        start, _ = windows[state.motion_id][state.window_idx]
        return max(0, start - ramp_steps)

    def assign_motion(env_id: int):
        nonlocal cursor
        if cursor >= len(motion_ids):
            states[env_id] = State()
            return False
        motion_id = motion_ids[cursor]
        cursor += 1
        states[env_id] = State(active=True, motion_id=motion_id, load_kg=args.max_load_kg, attempts=1)
        scheduler.assign(env_id, motion_id, attempt_start(states[env_id]))
        return True

    initial_ids = [env_id for env_id in range(base_env.num_envs) if assign_motion(env_id)]
    set_load(base_env, initial_ids, [0.0] * len(initial_ids), [args.max_load_kg] * len(initial_ids), args.ramp_sec)
    td = env.reset()

    policy = agent.get_rollout_policy("eval")
    policy.eval()
    env.eval()
    total_windows = sum(len(windows[motion_id]) for motion_id in motion_ids)
    progress = tqdm(total=total_windows, desc="window-load-cap")

    def required_hold_steps(state: State):
        start, end = windows[state.motion_id][state.window_idx]
        first_hold_frame = max(
            start,
            attempt_start(state) + (ramp_steps if state.load_kg > 0.0 else 0),
        )
        first_counted_frame = max(first_hold_frame, attempt_start(state) + 1)
        return end - first_counted_frame

    def record(state: State, status: str, load_kg: float, fail_frame="", reasons="", actual_load=""):
        start, end = windows[state.motion_id][state.window_idx]
        label = motion_labels[state.motion_id]
        dataset_id = int(dataset_ids[state.motion_id])
        group = command.dataset.groups[dataset_id]
        origin = int(label.get("window_start_frame", label.get("segment_start", 0)))
        rows.append(
            {
                "motion_id": state.motion_id,
                "dataset_name": group["name"],
                "teacher_mem_path": group["teacher_mem_path"],
                "student_mem_path": group["student_mem_path"] or "",
                "local_motion_id": state.motion_id - dataset_offsets[dataset_id],
                "source_path": source_paths[state.motion_id],
                "window_idx": state.window_idx,
                "expected_window_count": len(windows[state.motion_id]),
                "window_start_frame": start + origin,
                "window_end_frame": end + origin,
                "max_success_load_kg": load_kg,
                "attempts": state.attempts,
                "failed_loads_kg": ";".join(f"{value:g}" for value in state.failed_loads),
                "failure_phases": ";".join(state.failure_phases),
                "failed_actual_loads_kg": ";".join(f"{value:g}" for value in state.failed_actual_loads),
                "failure_reasons": ";".join(state.failure_reasons),
                "status": status,
                "fail_frame": fail_frame,
                "termination_reasons": reasons,
                "actual_load_kg": actual_load,
                "root_error_max_m": state.root_error_max,
                "keypoint_error_max_m": state.keypoint_error_max,
                "joint_error_max_rad": state.joint_error_max,
                "target_hold_steps": state.target_hold_steps,
                "required_hold_steps": required_hold_steps(state),
                "reached_target_frame": state.reached_target_frame,
                "confirmed_success": bool(status == "success" and state.confirming),
                "confirmation_successes": state.confirmation_successes,
                "confirmation_failures": state.confirmation_failures,
            }
        )
        progress.update(1)

    with set_exploration_type(ExplorationType.MODE):
        while any(state.active for state in states):
            td = policy(td)
            out = env.step(td)
            terminated = out.get(("next", "terminated")).reshape(-1).detach().cpu().tolist()
            truncated = out.get(("next", "truncated")).reshape(-1).detach().cpu().tolist()
            times = command.t.detach().cpu().tolist()
            root_errors = (command.asset.data.root_link_pos_w - command.reward_root_pos_w).norm(dim=-1).detach().cpu().tolist()
            keypoint_errors = (command._motion.body_pos_b[:, 0, command.keypoint_idx_motion] - command._current_keypoint_pos_b).norm(dim=-1).mean(dim=-1).detach().cpu().tolist()
            joint_errors = (command._motion.joint_pos[:, 0, command.joint_idx_motion] - command.asset.data.joint_pos[:, command.joint_idx_asset]).abs().mean(dim=-1).detach().cpu().tolist()
            load_manager = base_env.randomizations["window_cap_hand_load"]
            actual_loads = load_manager.current_total_load_kg.detach().cpu().tolist()
            term_stats = out.get(("next", "stats", "termination"))
            reset_ids = []
            reset_current = []
            reset_target = []

            for env_id, state in enumerate(states):
                if not state.active:
                    continue
                start, end = windows[state.motion_id][state.window_idx]
                last_frame = end - 1
                time_now = int(times[env_id])
                actual_load = float(actual_loads[env_id])
                at_target = abs(actual_load - state.load_kg) <= args.load_tolerance_kg
                if at_target and time_now >= start:
                    if state.target_hold_steps == 0:
                        state.reached_target_frame = time_now
                    state.target_hold_steps += 1
                else:
                    state.target_hold_steps = 0
                    state.reached_target_frame = -1
                state.root_error_max = max(state.root_error_max, root_errors[env_id])
                state.keypoint_error_max = max(state.keypoint_error_max, keypoint_errors[env_id])
                state.joint_error_max = max(state.joint_error_max, joint_errors[env_id])
                tracking_bad = (
                    root_errors[env_id] > args.root_error_threshold_m
                    or keypoint_errors[env_id] > args.keypoint_error_threshold_m
                    or joint_errors[env_id] > args.joint_error_threshold_rad
                )
                state.tracking_bad_steps = state.tracking_bad_steps + 1 if tracking_bad else 0
                tracking_failed = state.tracking_bad_steps >= args.tracking_error_grace_steps
                active_reasons = [
                    str(key) for key, value in term_stats.items() if bool(value[env_id].item())
                ]
                early_done = (bool(terminated[env_id]) or bool(truncated[env_id])) and time_now < last_frame
                reached_end = time_now >= last_frame
                expected_end_timeout = (
                    bool(truncated[env_id])
                    and reached_end
                    and "motion_timeout" in active_reasons
                    and set(active_reasons) <= {"motion_timeout"}
                )
                unexpected_truncation = bool(truncated[env_id]) and not expected_end_timeout
                hold_required = required_hold_steps(state)
                enough_hold = hold_required > 0 and state.target_hold_steps >= hold_required
                success = (
                    reached_end
                    and not bool(terminated[env_id])
                    and not unexpected_truncation
                    and not tracking_failed
                    and enough_hold
                )

                if reached_end and hold_required <= 0:
                    record(
                        state,
                        "insufficient_window",
                        0.0,
                        time_now,
                        "window_shorter_than_load_ramp",
                        actual_load,
                    )
                    state.window_idx += 1
                    if state.window_idx >= len(windows[state.motion_id]):
                        if assign_motion(env_id):
                            reset_ids.append(env_id)
                            reset_current.append(0.0)
                            reset_target.append(states[env_id].load_kg)
                        continue
                    reset_window_state(state)
                    scheduler.assign(env_id, state.motion_id, attempt_start(state))
                    reset_ids.append(env_id)
                    reset_current.append(0.0)
                    reset_target.append(state.load_kg)
                    continue

                if success:
                    near_error_limit = (
                        state.root_error_max >= args.root_error_threshold_m * args.confirm_error_fraction
                        or state.keypoint_error_max >= args.keypoint_error_threshold_m * args.confirm_error_fraction
                        or state.joint_error_max >= args.joint_error_threshold_rad * args.confirm_error_fraction
                    )
                    needs_confirmation = state.load_kg <= 0.0 or state.load_kg < args.max_load_kg or near_error_limit
                    if state.confirming:
                        state.confirmation_successes += 1
                        if (
                            state.load_kg <= 0.0
                            and state.confirmation_successes < 2
                            and state.confirmation_failures < 2
                        ):
                            state.attempts += 1
                            reset_attempt_metrics(state)
                            scheduler.assign(env_id, state.motion_id, attempt_start(state))
                            reset_ids.append(env_id)
                            reset_current.append(0.0)
                            reset_target.append(state.load_kg)
                            continue
                    elif args.confirm_boundary_successes and needs_confirmation:
                        state.confirming = True
                        state.confirmation_successes = 1
                        state.confirmation_failures = 0
                        state.attempts += 1
                        reset_attempt_metrics(state)
                        scheduler.assign(env_id, state.motion_id, attempt_start(state))
                        reset_ids.append(env_id)
                        reset_current.append(0.0)
                        reset_target.append(state.load_kg)
                        continue
                    record(state, "success", state.load_kg, actual_load=actual_load)
                    state.window_idx += 1
                    if state.window_idx >= len(windows[state.motion_id]):
                        if assign_motion(env_id):
                            reset_ids.append(env_id)
                            reset_current.append(0.0)
                            reset_target.append(states[env_id].load_kg)
                        continue
                    reset_window_state(state)
                    scheduler.assign(env_id, state.motion_id, attempt_start(state))
                    reset_ids.append(env_id)
                    reset_current.append(0.0)
                    reset_target.append(state.load_kg)
                    continue

                failed = early_done or tracking_failed or reached_end
                if failed:
                    reasons = ";".join(active_reasons)
                    if tracking_failed:
                        reasons = ";".join(filter(None, (reasons, "tracking_error")))
                    if reached_end and not enough_hold:
                        reasons = ";".join(filter(None, (reasons, "target_load_not_held")))
                    phase = "ramp" if not at_target else ("window" if time_now >= start else "prewindow_hold")
                    state.failed_loads.append(state.load_kg)
                    state.failure_phases.append(phase)
                    state.failed_actual_loads.append(actual_load)
                    state.failure_reasons.append(reasons or "unknown_failure")
                    if state.load_kg <= 0.0 and args.confirm_boundary_successes:
                        if not state.confirming:
                            state.confirming = True
                            state.confirmation_successes = 0
                            state.confirmation_failures = 0
                        state.confirmation_failures += 1
                        if state.confirmation_successes < 2 and state.confirmation_failures < 2:
                            state.attempts += 1
                            reset_attempt_metrics(state)
                            scheduler.assign(env_id, state.motion_id, attempt_start(state))
                            reset_ids.append(env_id)
                            reset_current.append(0.0)
                            reset_target.append(state.load_kg)
                            continue
                    elif state.confirming:
                        state.confirmation_failures += 1
                        if state.confirmation_successes == 1 and state.confirmation_failures == 1:
                            state.attempts += 1
                            reset_attempt_metrics(state)
                            scheduler.assign(env_id, state.motion_id, attempt_start(state))
                            reset_ids.append(env_id)
                            reset_current.append(0.0)
                            reset_target.append(state.load_kg)
                            continue
                    if state.load_kg <= 0.0:
                        if unexpected_truncation:
                            status = "truncated_at_zero"
                        elif state.confirmation_successes > 0:
                            status = "unstable_at_zero"
                        else:
                            status = "failed_at_zero" if phase == "window" else "warmup_failure_at_zero"
                        record(state, status, 0.0, time_now, reasons, actual_load)
                        state.window_idx += 1
                        if state.window_idx >= len(windows[state.motion_id]):
                            if not assign_motion(env_id):
                                continue
                            state = states[env_id]
                        else:
                            reset_window_state(state)
                    else:
                        state.load_kg = max(state.load_kg - args.load_step_kg, 0.0)
                        state.attempts += 1
                        state.confirming = False
                        state.confirmation_successes = 0
                        state.confirmation_failures = 0
                        reset_attempt_metrics(state)
                    scheduler.assign(env_id, state.motion_id, attempt_start(state))
                    reset_ids.append(env_id)
                    reset_current.append(0.0)
                    reset_target.append(state.load_kg)
                    continue

            next_td = step_mdp(out)
            if reset_ids:
                set_load(base_env, reset_ids, reset_current, reset_target, args.ramp_sec)
                reset_td = reset_envs(env, base_env, reset_ids)
                td = merge_reset(next_td, reset_td, reset_ids)
            else:
                td = next_td

    progress.close()
    write_outputs(Path(args.output_prefix).expanduser(), rows)
    base_env.close()


if __name__ == "__main__":
    main()
