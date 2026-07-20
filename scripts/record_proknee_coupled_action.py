#!/usr/bin/env python
"""Record closed-loop coupled prosthesis action checkpoints."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from musclemimic.proknee import MuscleProKneeCoupledActionEnv
from musclemimic.proknee.action_models import CoupledActionPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--motion-path", default="KIT/custom/official_retarget_walk6m_longseq_v1_poses")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--record-path", default="/home/user/Workspace/musclemimic/videos/coupled_action_deploy.mp4")
    parser.add_argument("--deploy-log", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--prosthesis-muscle-scale", type=float, default=None)
    parser.add_argument("--mask-preset", default=None)
    parser.add_argument("--obs-mode", choices=["easy", "pose_only"], default=None)
    parser.add_argument("--no-web-encode", action="store_true")
    return parser.parse_args()


def load_policy(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def write_web_compatible_mp4(src_path: str) -> str:
    base, ext = os.path.splitext(src_path)
    dst = f"{base}_web{ext or '.mp4'}"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            src_path,
            "-movflags",
            "+faststart",
            "-c:v",
            "libx264",
            "-profile:v",
            "baseline",
            "-pix_fmt",
            "yuv420p",
            "-an",
            dst,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return dst


def highlight_left_leg(model, audit) -> int:
    body_ids = {int(model.jnt_bodyid[j.joint_id]) for j in audit.joints}
    changed = 0
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) in body_ids:
            alpha = float(model.geom_rgba[gid, 3])
            model.geom_rgba[gid] = np.asarray([0.95, 0.08, 0.08, alpha], dtype=model.geom_rgba.dtype)
            changed += 1
    return changed


def build_mean_action(model, *, body_residual: bool):
    @jax.jit
    def mean_action(params, obs, priv):
        out = model.apply(params, obs, priv, body_residual=body_residual)
        if body_residual and out["body_mean"].shape[-1] > 0:
            return jnp.concatenate([out["prosthesis_mean"], out["body_mean"]], axis=-1)
        return out["prosthesis_mean"]

    return mean_action


def split_actions(action: np.ndarray, prosthesis_dim: int, body_dim: int):
    prosthesis = action[:prosthesis_dim]
    body = None if body_dim == 0 else action[prosthesis_dim : prosthesis_dim + body_dim]
    return prosthesis, body


def main() -> int:
    args = parse_args()
    ckpt = load_policy(args.policy)
    os.makedirs(os.path.dirname(os.path.abspath(args.record_path)), exist_ok=True)
    motion_path = args.motion_path.removesuffix(".npz")
    p_scale = float(args.prosthesis_muscle_scale) if args.prosthesis_muscle_scale is not None else float(
        ckpt.get("prosthesis_muscle_scale", 0.0)
    )
    mask_preset = args.mask_preset or str(ckpt.get("mask_preset", "strict19"))
    obs_mode = args.obs_mode or str(ckpt.get("obs_mode", "easy"))
    env = MuscleProKneeCoupledActionEnv(
        args.checkpoint,
        dataset_group=None,
        rel_dataset_path=[motion_path],
        deterministic_oracle=True,
        action_mode=str(ckpt.get("action_mode", "torque")),
        action_torque_limit=tuple(ckpt.get("action_torque_limit", (110.0, 65.0, 45.0, 22.0))),
        action_torque_slew_limit=ckpt.get("action_torque_slew_limit", (10.0, 5.0, 3.0, 1.5)),
        prosthesis_muscle_scale=p_scale,
        body_residual_scale=float(ckpt.get("body_residual_scale", 0.0)),
        mask_preset=mask_preset,
        obs_mode=obs_mode,
    )
    body_residual = bool(ckpt.get("body_residual_adapter", False))
    model = CoupledActionPolicy(
        prosthesis_action_dim=int(ckpt.get("prosthesis_action_dim", env.action_dim)),
        body_residual_dim=int(ckpt.get("body_residual_dim", 0)),
        action_scale=tuple(ckpt.get("action_torque_limit", (110.0, 65.0, 45.0, 22.0))),
    )
    mean_fn = build_mean_action(model, body_residual=body_residual)
    data = env.reset()
    changed = highlight_left_leg(env.env.model, env.audit)
    print(f"Recording coupled action policy: {args.policy}")
    print(f"Motion: {motion_path}")
    print(f"Output: {args.record_path}")
    print(f"highlighted={changed}, body_residual={body_residual}, prosthesis_muscle_scale={p_scale}")
    print(f"mask_preset={mask_preset} disabled_muscles={len(env.disabled_muscle_names)} obs_mode={obs_mode}")

    log_f = open(args.deploy_log, "w", newline="") if args.deploy_log else None
    log_w = csv.writer(log_f) if log_f else None
    if log_w:
        log_w.writerow(
            [
                "step",
                "done",
                "reward",
                "root_height",
                "root_up_z",
                "ncon",
                "toe_z",
                "tracking_mae",
                "torque_oracle_mae",
                "action_knee",
                "action_ankle",
                "action_subtalar",
                "action_mtp",
            ]
        )

    renderer = mujoco.Renderer(env.env.model, width=args.width, height=args.height)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 6.0
    cam.elevation = -20.0
    cam.azimuth = 90.0
    try:
        with imageio.get_writer(args.record_path, fps=args.fps, quality=8) as writer:
            for step in range(1, args.steps + 1):
                obs = jnp.asarray(data.obs[None, :], dtype=jnp.float32)
                priv = jnp.asarray(data.priv_info[None, :], dtype=jnp.float32)
                action = np.asarray(mean_fn(ckpt["params"], obs, priv))[0]
                p_action, b_action = split_actions(action, env.action_dim, int(ckpt.get("body_residual_dim", 0)))
                data = env.step_action(p_action, b_action)
                applied = np.asarray(data.info.get("prosthesis_applied_torque_mean", np.zeros(4)), dtype=np.float64)
                oracle = np.asarray(data.info.get("prosthesis_oracle_torque_target", np.zeros(4)), dtype=np.float64)
                torque_mae = float(np.mean(np.abs(applied - oracle)))
                if log_w:
                    log_w.writerow(
                        [
                            step,
                            int(data.done),
                            f"{data.reward:.6f}",
                            f"{float(data.info.get('prosthesis_root_height', 0.0)):.6f}",
                            f"{float(data.info.get('prosthesis_root_up_z', 1.0)):.6f}",
                            int(data.info.get("prosthesis_ncon", 0)),
                            f"{float(data.info.get('prosthesis_toe_z', 0.0)):.6f}",
                            f"{float(data.info.get('prosthesis_tracking_mae', 0.0)):.6f}",
                            f"{torque_mae:.6f}",
                            *[f"{float(x):.6f}" for x in p_action],
                        ]
                    )
                cam.lookat[:] = np.asarray(env.env.data.qpos[:3], dtype=np.float64)
                renderer.update_scene(env.env.data, camera=cam)
                writer.append_data(renderer.render())
                if data.done:
                    print(f"Done at step {step}")
                    break
    finally:
        renderer.close()
        env.env.stop()
        if log_f:
            log_f.close()

    print(f"Saved video: {args.record_path}")
    if args.deploy_log:
        print(f"Saved log: {args.deploy_log}")
    if not args.no_web_encode:
        try:
            print(f"Web-compatible copy: {write_web_compatible_mp4(args.record_path)}")
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"[warn] web encode skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

