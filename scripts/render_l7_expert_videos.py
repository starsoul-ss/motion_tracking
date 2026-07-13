#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torchrl.envs import ExplorationType, set_exploration_type

from active_adaptation.envs.base import _ViserDebugDraw
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig
from scripts.utils.helpers import make_env_policy
from scripts.utils.l7_load_eval import (
    MotionScheduler,
    configure_controlled_load,
    neutralize_eval_randomization,
    set_load,
    unwrap_base_env,
)


GROUP_QUOTAS = (2, 3, 3, 2)


def select_clips(dataset, clip_steps: int, seed: int, group_quotas=GROUP_QUOTAS):
    rng = np.random.default_rng(seed)
    dataset_ids = dataset.motion_to_dataset_id.cpu().numpy()
    lengths = dataset.global_lengths.cpu().numpy()
    clips = []
    for group_id, count in enumerate(group_quotas):
        candidates = np.flatnonzero((dataset_ids == group_id) & (lengths >= clip_steps))
        if len(candidates) < count:
            raise RuntimeError(f"dataset group {group_id} has only {len(candidates)} clips long enough")
        for motion_id in rng.choice(candidates, size=count, replace=False):
            length = int(lengths[motion_id])
            start = int(rng.integers(0, length - clip_steps + 1))
            clips.append((int(motion_id), group_id, start, length))
    return clips


def parse_clip_spec(spec: str, default_current_load, default_target_load):
    fields = spec.split(":")
    if len(fields) not in (2, 3, 4):
        raise ValueError(f"invalid clip {spec!r}; expected motion:start[:target_load] or motion:start:current_load:target_load")
    motion_id, start = map(int, fields[:2])
    current_load = default_current_load
    target_load = default_target_load
    if len(fields) == 3:
        target_load = float(fields[2])
    elif len(fields) == 4:
        current_load, target_load = map(float, fields[2:])
    if any(value is not None and (not np.isfinite(value) or value < 0.0) for value in (current_load, target_load)):
        raise ValueError(f"clip load values must be nonnegative: {spec!r}")
    return motion_id, start, current_load, target_load


def font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def annotate(frame, title: str, detail: str):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 62), fill=(0, 0, 0))
    draw.text((12, 7), title, font=font(21), fill=(255, 255, 255))
    draw.text((12, 34), detail, font=font(17), fill=(160, 255, 160))
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--continue-after-done", action="store_true")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--clips",
        help="Comma-separated motion:start[:target_load] or motion:start:current_load:target_load specs",
    )
    parser.add_argument("--load-kg", type=float)
    parser.add_argument("--current-load-kg", type=float)
    parser.add_argument("--ramp-seconds", type=float, default=1.0)
    parser.add_argument("--dataset", help="Use one teacher memmap dataset instead of cfg groups")
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--filename-prefix", default="")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        raise ValueError("--limit must be in [1, 100]")
    if any(
        value is not None and (not np.isfinite(value) or value < 0.0)
        for value in (args.load_kg, args.current_load_kg)
    ):
        raise ValueError("load values must be nonnegative")
    default_target_load = 0.0 if args.load_kg is None else args.load_kg
    clip_specs = None
    if args.clips:
        clip_specs = [
            parse_clip_spec(spec, args.current_load_kg, default_target_load)
            for spec in args.clips.split(",")
        ]

    cfg = OmegaConf.load(Path(args.cfg).expanduser())
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(Path(args.checkpoint).expanduser().resolve())
    cfg.vecnorm = "eval"
    cfg.seed = args.seed
    cfg.eval_render = False
    cfg.wandb.mode = "disabled"
    cfg.task.num_envs = 1
    cfg.task.viewer.headless = True
    cfg.app.headless = True
    cfg.app.enable_cameras = False
    cfg.task.command.reinit_prob = 0.0
    if args.dataset:
        cfg.task.profile.command.dataset.groups = [{"name": "selected", "teacher_mem_path": args.dataset, "weight": 1.0}]
    for key in cfg.task.command.init_noise:
        cfg.task.command.init_noise[key] = 0.0
    if args.deterministic_eval:
        neutralize_eval_randomization(cfg)
    requested_loads = [value for value in (args.current_load_kg, default_target_load) if value is not None]
    if clip_specs:
        requested_loads.extend(value for spec in clip_specs for value in spec[2:] if value is not None)
    max_eval_load = max(requested_loads)
    configure_controlled_load(cfg, max_eval_load)

    fps = int(round(1.0 / float(cfg.task.sim.step_dt)))
    clip_steps = 20000 if args.seconds <= 0 else int(round(args.seconds * fps))
    cfg.task.max_episode_length = clip_steps + 10
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env, agent, _, _ = make_env_policy(cfg)
    base_env = unwrap_base_env(env)
    scheduler = MotionScheduler(base_env)
    scheduler.install()
    if hasattr(agent, "step_schedule"):
        agent.step_schedule(1.0, 0)
    if hasattr(env, "step_schedule"):
        env.step_schedule(1.0, 0)
    policy = agent.get_rollout_policy("eval")

    dataset = base_env.command_manager.dataset
    if clip_specs:
        clips = []
        for motion_id, start, current_load, target_load in clip_specs:
            if not 0 <= motion_id < len(dataset.global_lengths):
                raise ValueError(f"motion_id out of range: {motion_id}")
            length = int(dataset.global_lengths[motion_id])
            if not 0 <= start < length:
                raise ValueError(f"start_frame out of range for motion {motion_id}: {start}")
            clips.append((motion_id, int(dataset.motion_to_dataset_id[motion_id]), start, length, current_load, target_load))
    else:
        remaining = args.limit
        quotas = (remaining,) if args.dataset else tuple(
            min(quota, max(remaining - sum(GROUP_QUOTAS[:group_id]), 0))
            for group_id, quota in enumerate(GROUP_QUOTAS)
        )
        clips = [
            (*clip, args.current_load_kg, default_target_load)
            for clip in select_clips(dataset, clip_steps, args.seed, quotas)
        ]
    clips = clips[: args.limit]
    viewer_cfg = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name="robot",
        env_idx=0,
        max_extra_envs=0,
        width=args.width,
        height=args.height,
        distance=4.0,
        elevation=-15.0,
        azimuth=135.0,
        enable_reflections=False,
    )
    renderer = OffscreenRenderer(base_env.sim.mj_model, viewer_cfg, base_env.scene)
    renderer.initialize()
    debug_draw = None
    manifest = []
    load = base_env.randomizations["window_cap_hand_load"]
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.filename_prefix)

    def add_reference_ghost(visualizer):
        nonlocal debug_draw
        if debug_draw is None:
            debug_draw = _ViserDebugDraw(visualizer)
        else:
            debug_draw._scene = visualizer
        env_ids = torch.zeros(1, dtype=torch.long, device=base_env.device)
        qpos = base_env.command_manager._motion_qpos_for_debug(
            base_env.command_manager._motion,
            env_ids,
            root_offset_w=(0.35, 0.0, 0.0),
        )[0]
        debug_draw.ghost(qpos.detach().cpu().numpy(), base_env.sim.mj_model, color=(0.0, 1.0, 0.0), alpha=0.45)

    try:
        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            for index, (motion_id, group_id, start, length, clip_current_load, clip_target_load) in enumerate(clips, 1):
                label = dict(dataset.motion_labels[motion_id])
                source = str(label.get("source_path", f"motion_{motion_id}"))
                source_name = Path(source).stem
                slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_name)[:60]
                video_path = output_dir / f"{prefix}{index:02d}_motion_{motion_id}_{slug}.mp4"
                scheduler.assign(0, motion_id, start)
                current_load = clip_target_load if clip_current_load is None else clip_current_load
                set_load(base_env, [0], [current_load], [clip_target_load], args.ramp_seconds)
                td = env.reset()
                rendered_steps = 0
                terminated = False
                first_done_step = None
                termination_reasons = []
                termination_body_z = []
                root_errors = []
                keypoint_errors = []
                joint_errors = []
                steps = length - start if args.seconds <= 0 else min(clip_steps, length - start)
                with imageio.get_writer(
                    video_path,
                    fps=fps,
                    codec="libx264",
                    quality=8,
                    macro_block_size=None,
                ) as writer:
                    for step in range(steps):
                        torch.compiler.cudagraph_mark_step_begin()
                        td = policy(td)
                        transition = env.step(td)
                        td = transition["next"]
                        command = base_env.command_manager
                        root_error = float((command.asset.data.root_link_pos_w - command.reward_root_pos_w).norm(dim=-1)[0])
                        keypoint_error = float((command._motion.body_pos_b[:, 0, command.keypoint_idx_motion] - command._current_keypoint_pos_b).norm(dim=-1).mean(dim=-1)[0])
                        joint_error = float((command._motion.joint_pos[:, 0, command.joint_idx_motion] - command.asset.data.joint_pos[:, command.joint_idx_asset]).abs().mean(dim=-1)[0])
                        root_errors.append(root_error)
                        keypoint_errors.append(keypoint_error)
                        joint_errors.append(joint_error)
                        renderer.update(base_env.sim.data, debug_vis_callback=add_reference_ghost)
                        frame = renderer.render()
                        current = float(load.current_total_load_kg[0])
                        target = float(load.target_total_load_kg[0])
                        title = f"{index:02d}/{len(clips):02d}  {dataset.groups[group_id]['name']}  motion={motion_id}  {source_name}"
                        detail = (
                            f"robot=policy, green=reference (+0.35m) | "
                            f"load_total={current:.1f}/{target:.1f}kg, per_hand={current / 2.0:.1f}kg | "
                            f"err root={root_error:.2f}m kp={keypoint_error:.2f}m joint={joint_error:.2f}rad"
                        )
                        if terminated:
                            detail += f" | TERMINATED@{first_done_step}"
                        writer.append_data(annotate(frame, title, detail))
                        rendered_steps += 1
                        if bool(td["done"].item()) and not terminated:
                            terminated = True
                            first_done_step = step + 1
                            term_stats = td.get(("stats", "termination"), None)
                            if term_stats is not None:
                                termination_reasons = [str(key) for key, value in term_stats.items() if bool(value.item())]
                            if "body_z_termination" in termination_reasons:
                                target_z, current_z = command._body_z_values(list(command._body_z_termination_patterns))
                                termination_body_z = [
                                    {"body": name, "target_z": float(target), "current_z": float(current), "delta_z": float(current - target)}
                                    for name, target, current in zip(command._body_z_names_asset, target_z[0], current_z[0])
                                ]
                            if not args.continue_after_done:
                                break
                record = {
                    "video": video_path.name,
                    "motion_id": motion_id,
                    "group": dataset.groups[group_id]["name"],
                    "source_path": source,
                    "total_load_kg": clip_target_load,
                    "per_hand_load_kg": clip_target_load / 2.0,
                    "start_frame": start,
                    "motion_length": length,
                    "rendered_steps": rendered_steps,
                    "terminated_early": first_done_step is not None and "motion_timeout" not in termination_reasons,
                    "first_done_step": first_done_step,
                    "termination_reasons": termination_reasons,
                    "termination_body_z": termination_body_z,
                    "tracking_error": {
                        "root_mean_m": float(np.mean(root_errors)), "root_max_m": float(np.max(root_errors)),
                        "keypoint_mean_m": float(np.mean(keypoint_errors)), "keypoint_max_m": float(np.max(keypoint_errors)),
                        "joint_mean_rad": float(np.mean(joint_errors)), "joint_max_rad": float(np.max(joint_errors)),
                    },
                    "fps": fps,
                }
                manifest.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        renderer.close()

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
