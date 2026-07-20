#!/usr/bin/env python
"""Play Stage1/Stage2 policy in HybridEnv with left-leg highlight."""

from __future__ import annotations

import argparse
import pickle

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

from musclemimic.proknee import MuscleProKneeHybridEnv
from musclemimic.proknee.models import MuscleProKneeStudent, MuscleProKneeTeacher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2")
    parser.add_argument("--policy", required=True, help="Stage1/Stage2 .pt checkpoint path")
    parser.add_argument("--kind", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument("--dataset-group", default="KIT_KINESIS_TESTING_MOTIONS")
    parser.add_argument("--motion-path", nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--deterministic-oracle", action="store_true")
    parser.add_argument("--cam-distance", type=float, default=6.0)
    parser.add_argument("--cam-elevation", type=float, default=-20.0)
    parser.add_argument("--cam-azimuth", type=float, default=90.0)
    parser.add_argument("--record", action="store_true", help="Headless record to mp4 (no GUI needed)")
    parser.add_argument("--record-path", default="outputs/proknee_stage1_play.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--execute-policy",
        action="store_true",
        help=(
            "Experimentally inject the Stage1/2 output into the MuJoCo state. "
            "Default is oracle rollout only, which is the valid Stage1 evaluation mode."
        ),
    )
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _highlight_left_leg(model: mujoco.MjModel, joint_ids: list[int]) -> int:
    left_body_ids = {int(model.jnt_bodyid[jid]) for jid in joint_ids}
    changed = 0
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        if bid in left_body_ids:
            rgba = model.geom_rgba[gid].copy()
            model.geom_rgba[gid] = np.array([0.92, 0.12, 0.12, rgba[3]], dtype=model.geom_rgba.dtype)
            changed += 1
    return changed


def main() -> int:
    args = parse_args()
    ckpt = load_pickle(args.policy)
    env = MuscleProKneeHybridEnv(
        args.checkpoint,
        dataset_group=args.dataset_group,
        rel_dataset_path=args.motion_path,
        history_len=int(ckpt.get("history_len", 30)),
        deterministic_oracle=args.deterministic_oracle,
        apply_teacher_action=args.execute_policy,
        target_mode=str(ckpt.get("target_mode", "residual")),
        pd_override=bool(ckpt.get("dagger_pd_override", False)) if args.execute_policy else False,
        pd_kp=tuple(ckpt.get("pd_kp", (300.0, 200.0, 200.0, 120.0))),
        pd_kd=tuple(ckpt.get("pd_kd", (30.0, 20.0, 20.0, 12.0))),
        pd_torque_limit=float(ckpt.get("pd_torque_limit", 120.0)),
        pd_gain_scale=float(ckpt.get("pd_gain_scale", 1.0)),
        max_prosthesis_qpos_step=ckpt.get("max_prosthesis_qpos_step", None),
        prosthesis_muscle_scale=float(ckpt.get("prosthesis_muscle_scale", 1.0)),
    )
    data = env.reset()

    if args.kind == "stage1":
        model = MuscleProKneeTeacher(action_dim=int(ckpt["action_dim"]))

        @jax.jit
        def predict(params, obs, priv, hist):
            del hist
            pred, _ = model.apply(params, obs, priv)
            return pred

    else:
        model = MuscleProKneeStudent(action_dim=int(ckpt["action_dim"]))

        @jax.jit
        def predict(params, obs, priv, hist):
            pred, _ = model.apply(params, obs, priv, hist, mode="student")
            return pred

    joint_ids = [j.joint_id for j in env.audit.joints]
    changed = _highlight_left_leg(env.env.model, joint_ids)
    print(f"[play] highlighted {changed} geoms for left prosthesis side.")

    if args.record:
        print(f"[play] recording headless video -> {args.record_path}")
        renderer = mujoco.Renderer(env.env.model, width=args.width, height=args.height)
        frames: list[np.ndarray] = []
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = args.cam_distance
        cam.elevation = args.cam_elevation
        cam.azimuth = args.cam_azimuth

        mse_sum = 0.0
        for step in range(1, args.steps + 1):
            obs = jnp.asarray(data.obs[None, :], dtype=jnp.float32)
            priv = jnp.asarray(data.priv_info[None, :], dtype=jnp.float32)
            hist = jnp.asarray(data.proprio_hist[None, :, :], dtype=jnp.float32)
            target = jnp.asarray(data.oracle_target[None, :], dtype=jnp.float32)
            pred = predict(ckpt["params"], obs, priv, hist)
            teacher_action = np.asarray(pred)[0] if args.execute_policy else None
            mse_sum += float(jnp.mean(jnp.square(pred - target)))
            data = env.step(teacher_action=teacher_action)
            if data.done:
                data = env.reset()

            root = np.asarray(env.env.data.qpos[:3], dtype=np.float64)
            cam.lookat[:] = root
            renderer.update_scene(env.env.data, camera=cam)
            frames.append(renderer.render())
            if step % 500 == 0:
                print(f"[play] step={step} running_target_mse={mse_sum / step:.6f}", flush=True)

        imageio.mimsave(args.record_path, frames, fps=args.fps, quality=8)
        print(f"[play] video saved: {args.record_path}")
        return 0

    print("[play] close viewer window or press ESC to quit.")
    with mujoco.viewer.launch_passive(env.env.model, env.env.data) as viewer:
        viewer.cam.distance = args.cam_distance
        viewer.cam.elevation = args.cam_elevation
        viewer.cam.azimuth = args.cam_azimuth

        mse_sum = 0.0
        for step in range(1, args.steps + 1):
            obs = jnp.asarray(data.obs[None, :], dtype=jnp.float32)
            priv = jnp.asarray(data.priv_info[None, :], dtype=jnp.float32)
            hist = jnp.asarray(data.proprio_hist[None, :, :], dtype=jnp.float32)
            target = jnp.asarray(data.oracle_target[None, :], dtype=jnp.float32)
            pred = predict(ckpt["params"], obs, priv, hist)
            teacher_action = np.asarray(pred)[0] if args.execute_policy else None
            mse_sum += float(jnp.mean(jnp.square(pred - target)))
            data = env.step(teacher_action=teacher_action)
            if data.done:
                data = env.reset()
            try:
                viewer.cam.lookat[:] = np.asarray(env.env.data.qpos[:3], dtype=np.float64)
            except Exception:
                pass
            viewer.sync()
            if not viewer.is_running():
                break
            if step % 500 == 0:
                print(f"[play] step={step} running_target_mse={mse_sum / step:.6f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
