#!/usr/bin/env python3
"""Evaluate a checkpoint on a fixed random sample of windows or motions."""

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils.l7_load_eval import make_eval_env, merge_reset, reset_envs, set_load, window_bounds


@dataclass(frozen=True)
class Task:
    motion_id: int
    window_idx: int
    start: int
    end: int
    load_kg: float


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--load-kgs", type=float, nargs="+", required=True, help="Total two-hand load in kg.")
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--ramp-sec", type=float, default=1.0)
    parser.add_argument("--sample-motions", action="store_true", help="Evaluate complete motions instead of windows.")
    parser.add_argument("--source-path-contains", default="", help="Only sample motions whose source path contains this text.")
    parser.add_argument("--constant-from-start", action="store_true", help="Apply the target load from the first frame.")
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--task-shard-count", type=int, default=1)
    args = parser.parse_args()
    if any(load_kg < 0.0 for load_kg in args.load_kgs):
        parser.error("--load-kgs values must be nonnegative")
    return args


def write_outputs(prefix: Path, rows: list[dict]):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "motion_id",
        "source_path",
        "window_idx",
        "window_start_frame",
        "window_end_frame",
        "total_load_kg",
        "success",
        "fail_frame",
        "termination_reasons",
        "steps",
        "actual_load_mean_kg",
        "root_error_mean_m",
        "root_error_max_m",
        "keypoint_error_mean_m",
        "keypoint_error_max_m",
        "joint_error_mean_rad",
        "joint_error_max_rad",
    ]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for load_kg in sorted({row["total_load_kg"] for row in rows}):
        group = [row for row in rows if row["total_load_kg"] == load_kg]
        summary_rows.append(
            {
                "total_load_kg": load_kg,
                "per_hand_load_kg": load_kg / 2.0,
                "evaluations": len(group),
                "successes": sum(row["success"] for row in group),
                "success_rate": sum(row["success"] for row in group) / len(group),
                "actual_load_mean_kg": float(np.mean([row["actual_load_mean_kg"] for row in group])),
                "root_error_mean_m": float(np.mean([row["root_error_mean_m"] for row in group])),
                "root_error_max_m": max(row["root_error_max_m"] for row in group),
                "keypoint_error_mean_m": float(np.mean([row["keypoint_error_mean_m"] for row in group])),
                "keypoint_error_max_m": max(row["keypoint_error_max_m"] for row in group),
                "joint_error_mean_rad": float(np.mean([row["joint_error_mean_rad"] for row in group])),
                "joint_error_max_rad": max(row["joint_error_max_rad"] for row in group),
            }
        )
    summary_fields = list(summary_rows[0]) if summary_rows else []
    with prefix.with_name(prefix.name + "_summary").with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)


@torch.no_grad()
def main():
    args = parse_args()
    env, agent, base_env, scheduler = make_eval_env(
        args.cfg_path, args.checkpoint_path, args.num_envs, max(args.load_kgs)
    )
    command = base_env.command_manager
    source_paths = command.dataset.motion_source_paths
    motion_labels = command.dataset.motion_labels
    lengths = command.dataset.global_lengths.detach().cpu().tolist()
    window_steps = max(1, round(args.window_sec / base_env.step_dt))
    ramp_steps = max(1, round(args.ramp_sec / base_env.step_dt))

    eligible_motion_ids = [
        motion_id
        for motion_id, (source_path, length) in enumerate(zip(source_paths, lengths))
        if length > 1 and args.source_path_contains in source_path
    ]
    if not eligible_motion_ids:
        raise ValueError(f"No motions matched source filter {args.source_path_contains!r}")
    rng = np.random.default_rng(args.seed)
    if args.sample_motions:
        if args.sample_count > len(eligible_motion_ids):
            raise ValueError(f"Requested {args.sample_count} motions, only {len(eligible_motion_ids)} matched")
        selected = rng.choice(eligible_motion_ids, size=args.sample_count, replace=False)
        sampled = [(motion_id, 0, 0, lengths[motion_id]) for motion_id in selected]
    else:
        windows = []
        for motion_id in eligible_motion_ids:
            for window_idx, (start, end) in enumerate(window_bounds(lengths[motion_id], window_steps)):
                windows.append((motion_id, window_idx, start, end))
        selected = rng.choice(len(windows), size=min(args.sample_count, len(windows)), replace=False)
        sampled = [windows[idx] for idx in selected]
    tasks = [Task(*window, load_kg) for window in sampled for load_kg in args.load_kgs]
    tasks = [task for idx, task in enumerate(tasks) if idx % args.task_shard_count == args.task_shard_index]

    active = [False] * base_env.num_envs
    current_tasks = [None] * base_env.num_envs
    rows = []
    cursor = 0
    metric_names = ("root", "keypoint", "joint", "actual_load")
    metric_sums = {name: [0.0] * base_env.num_envs for name in metric_names}
    metric_maxes = {name: [0.0] * base_env.num_envs for name in metric_names}
    metric_steps = [0] * base_env.num_envs

    def reset_metrics(env_id: int):
        metric_steps[env_id] = 0
        for name in metric_names:
            metric_sums[name][env_id] = 0.0
            metric_maxes[name][env_id] = 0.0

    def assign(env_id: int):
        nonlocal cursor
        if cursor >= len(tasks):
            active[env_id] = False
            return False
        task = tasks[cursor]
        cursor += 1
        active[env_id] = True
        current_tasks[env_id] = task
        reset_metrics(env_id)
        scheduler.assign(env_id, task.motion_id, max(0, task.start - ramp_steps))
        return True

    initial_ids = [env_id for env_id in range(base_env.num_envs) if assign(env_id)]
    set_load(
        base_env,
        initial_ids,
        [current_tasks[env_id].load_kg if args.constant_from_start else 0.0 for env_id in initial_ids],
        [current_tasks[env_id].load_kg for env_id in initial_ids],
        args.ramp_sec,
    )
    td = env.reset()

    policy = agent.get_rollout_policy("eval")
    policy.eval()
    env.eval()
    progress = tqdm(total=len(tasks), desc="load-sweep")

    with set_exploration_type(ExplorationType.MODE):
        while any(active):
            td = policy(td)
            out = env.step(td)
            terminated = out.get(("next", "terminated")).reshape(-1).detach().cpu().tolist()
            truncated = out.get(("next", "truncated")).reshape(-1).detach().cpu().tolist()
            times = command.t.detach().cpu().tolist()
            root_errors = (command.asset.data.root_link_pos_w - command.reward_root_pos_w).norm(dim=-1).detach().cpu().tolist()
            keypoint_errors = (command._motion.body_pos_b[:, 0, command.keypoint_idx_motion] - command._current_keypoint_pos_b).norm(dim=-1).mean(dim=-1).detach().cpu().tolist()
            joint_errors = (command._motion.joint_pos[:, 0, command.joint_idx_motion] - command.asset.data.joint_pos[:, command.joint_idx_asset]).abs().mean(dim=-1).detach().cpu().tolist()
            actual_loads = base_env.randomizations["window_cap_hand_load"].current_total_load_kg.detach().cpu().tolist()
            term_stats = out.get(("next", "stats", "termination"))
            reset_ids = []

            for env_id, is_active in enumerate(active):
                if not is_active:
                    continue
                task = current_tasks[env_id]
                values = {
                    "root": root_errors[env_id],
                    "keypoint": keypoint_errors[env_id],
                    "joint": joint_errors[env_id],
                    "actual_load": actual_loads[env_id],
                }
                metric_steps[env_id] += 1
                for name, value in values.items():
                    metric_sums[name][env_id] += value
                    metric_maxes[name][env_id] = max(metric_maxes[name][env_id], value)
                active_reasons = [str(key) for key, value in term_stats.items() if bool(value[env_id].item())]
                reached_end = times[env_id] >= task.end - 1
                expected_timeout = (
                    bool(truncated[env_id])
                    and reached_end
                    and "motion_timeout" in active_reasons
                    and set(active_reasons) <= {"motion_timeout"}
                )
                unexpected_truncation = bool(truncated[env_id]) and not expected_timeout
                success = reached_end and not bool(terminated[env_id]) and not unexpected_truncation
                if not (success or bool(terminated[env_id]) or unexpected_truncation):
                    continue
                label = motion_labels[task.motion_id]
                origin = int(label.get("window_start_frame", label.get("segment_start", 0)))
                steps = metric_steps[env_id]
                rows.append(
                    {
                        "motion_id": task.motion_id,
                        "source_path": source_paths[task.motion_id],
                        "window_idx": task.window_idx,
                        "window_start_frame": task.start + origin,
                        "window_end_frame": task.end + origin,
                        "total_load_kg": task.load_kg,
                        "success": int(success),
                        "fail_frame": "" if success else times[env_id],
                        "termination_reasons": ";".join(active_reasons),
                        "steps": steps,
                        "actual_load_mean_kg": metric_sums["actual_load"][env_id] / steps,
                        "root_error_mean_m": metric_sums["root"][env_id] / steps,
                        "root_error_max_m": metric_maxes["root"][env_id],
                        "keypoint_error_mean_m": metric_sums["keypoint"][env_id] / steps,
                        "keypoint_error_max_m": metric_maxes["keypoint"][env_id],
                        "joint_error_mean_rad": metric_sums["joint"][env_id] / steps,
                        "joint_error_max_rad": metric_maxes["joint"][env_id],
                    }
                )
                progress.update(1)
                if assign(env_id):
                    reset_ids.append(env_id)

            next_td = step_mdp(out)
            if reset_ids:
                set_load(
                    base_env,
                    reset_ids,
                    [current_tasks[env_id].load_kg if args.constant_from_start else 0.0 for env_id in reset_ids],
                    [current_tasks[env_id].load_kg for env_id in reset_ids],
                    args.ramp_sec,
                )
                reset_td = reset_envs(env, base_env, reset_ids)
                td = merge_reset(next_td, reset_td, reset_ids)
            else:
                td = next_td

    progress.close()
    write_outputs(Path(args.output_prefix).expanduser(), rows)
    base_env.close()


if __name__ == "__main__":
    main()
