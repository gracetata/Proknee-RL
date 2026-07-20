"""Soft-switch evaluation viewer.

Keys (must focus the MuJoCo viewer window first):
  1 -> walking reference  (soft switch; physics is not reset)
  2 -> running reference  (soft switch; physics is not reset)

Behavior:
  * Reference trajectory loops forever. When the current reference reaches its
    end, we wrap subtraj_step_no back to 0 without touching qpos/qvel, so the
    agent keeps moving smoothly under the same gait.
  * On key press, only the trajectory index is swapped; the agent keeps its
    current physical state, producing a smooth gait transition.
  * On env `done` (e.g. the policy falls), we *do* reset the environment, but
    we preserve the currently selected reference.
  * Camera tracks the agent root while staying at a fixed, user-tunable zoom.

Reuses the official eval setup so everything else matches `fullbody/eval.py`.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=True ")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from musclemimic.algorithms import PPOJax
from musclemimic.runner.eval_utils import (
    align_agent_state,
    apply_temporal_params,
    configure_goal_visualization,
    load_checkpoint,
)
from loco_mujoco.task_factories import TaskFactory


DEFAULT_WALK = "KIT/314/walking_medium09_poses"
DEFAULT_RUN = "KIT/314/walking_run09_poses"


def _extract_mujoco_env(env):
    if hasattr(env, "model") and hasattr(env, "data"):
        return env, env.model, env.data
    curr = env
    while hasattr(curr, "env"):
        curr = curr.env
        if hasattr(curr, "model") and hasattr(curr, "data"):
            return curr, curr.model, curr.data
    raise RuntimeError("Could not access MuJoCo model/data from environment")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live soft-switch walking/running viewer.")
    parser.add_argument("--path", required=True, help="Checkpoint path (local dir)")
    parser.add_argument("--walking", default=DEFAULT_WALK, help="Walking motion relative path (no .npz)")
    parser.add_argument("--running", default=DEFAULT_RUN, help="Running motion relative path (no .npz)")
    parser.add_argument("--n_steps", type=int, default=10_000_000, help="Max simulation steps")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions; else take the mean")
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--cam_distance", type=float, default=6.0, help="Camera distance (meters)")
    parser.add_argument("--cam_elevation", type=float, default=-20.0, help="Camera elevation (degrees)")
    parser.add_argument("--cam_azimuth", type=float, default=90.0, help="Camera azimuth (degrees)")
    parser.add_argument(
        "--loop_from",
        type=int,
        default=150,
        help="When a reference trajectory ends, loop back to this step (skip the stand-up head).",
    )
    parser.add_argument(
        "--start_step_walk",
        type=int,
        default=150,
        help="When switching to WALKING (key 1/W), start the reference at this step to skip the initial standing frames.",
    )
    parser.add_argument(
        "--start_step_run",
        type=int,
        default=150,
        help="When switching to RUNNING (key 2/R), start the reference at this step.",
    )
    parser.add_argument(
        "--no_termination",
        action="store_true",
        help="Never reset on done=True (avoids 'screen flash' after soft-switch).",
    )
    args = parser.parse_args()

    # ---- 1. Load checkpoint & build config ----
    config, agent_state, _ = load_checkpoint(args.path)
    OmegaConf.set_struct(config, False)

    env_name = config.experiment.env_params.get("env_name")
    config.experiment.env_params["headless"] = False

    motions = [args.walking, args.running]
    config.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path = motions
    config.experiment.task_factory.params.amass_dataset_conf.dataset_group = None

    class _ArgsShim:
        no_render = False
        record = False
        mujoco_viewer = True
        use_mujoco = True
        viser_viewer = False

    configure_goal_visualization(config, _ArgsShim(), "GoalTrajMimicv2", is_mjx_env="Mjx" in env_name)

    apply_temporal_params(config)

    play_env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    apply_eval_terminal_defaults(play_env_params, config, strict_termination=False)
    if "Mjx" in play_env_params.get("env_name", ""):
        play_env_params["env_name"] = play_env_params["env_name"].replace("Mjx", "")

    # Start deterministically from walking, step 0.
    th_params = play_env_params.setdefault("th_params", {})
    th_params["random_start"] = False
    th_params["fixed_start_conf"] = [0, 0]
    th_params["start_from_random_step"] = False

    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    merged_params = {**play_env_params, **task_params}

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(**merged_params)

    agent_conf = PPOJax.init_agent_conf(env, config)
    agent_state = align_agent_state(agent_state, agent_conf)

    train_state = agent_state.train_state
    if not args.stochastic:
        train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf

    def sample_actions(ts, obs, _rng):
        y, updates = agent_conf.network.apply(
            {"params": ts.params, "run_stats": ts.run_stats}, obs, mutable=["run_stats"]
        )
        ts = ts.replace(run_stats=updates["run_stats"])
        pi, _ = y
        a = pi.sample(seed=_rng)
        return a, ts

    plcy_call = jax.jit(sample_actions)
    rng = jax.random.key(args.eval_seed)

    obs = env.reset()
    mj_env, model, data = _extract_mujoco_env(env)

    # ---- 2. Soft-switch helpers ----
    # env._additional_carry.traj_state is a flax struct; we rebuild it (immutable).
    current_idx = {"i": 0}  # 0 = walking, 1 = running

    _I32 = np.int32
    START_STEP = {0: int(args.start_step_walk), 1: int(args.start_step_run)}
    LOOP_STEP = int(args.loop_from)

    def _make_traj_state(traj_state, *, traj_no, subtraj_step_no=0, subtraj_step_no_init=0):
        return traj_state.replace(
            traj_no=np.asarray(int(traj_no), dtype=_I32),
            subtraj_step_no=np.asarray(int(subtraj_step_no), dtype=_I32),
            subtraj_step_no_init=np.asarray(int(subtraj_step_no_init), dtype=_I32),
        )

    def _clamp_step(traj_no: int, step: int) -> int:
        """Make sure `step` is a valid frame for this trajectory (0 <= step < len-1)."""
        last = int(mj_env.th.last_step_idx(int(traj_no)))
        if last <= 0:
            return 0
        return max(0, min(int(step), last - 1))

    def set_reference(new_idx: int) -> None:
        """Swap to a different reference trajectory WITHOUT touching physics."""
        carry = mj_env._additional_carry
        start = _clamp_step(new_idx, START_STEP.get(new_idx, 0))
        new_ts = _make_traj_state(carry.traj_state, traj_no=int(new_idx), subtraj_step_no=start)
        mj_env._additional_carry = carry.replace(traj_state=new_ts)
        current_idx["i"] = int(new_idx)
        print(f"[live] reference switched -> idx={new_idx}, start_step={start}")

    def maybe_loop_reference() -> None:
        """If the current reference reached its last frame, loop back to LOOP_STEP
        (not 0) so we skip the mocap's stand-up head and stay in the gait."""
        carry = mj_env._additional_carry
        traj_state = carry.traj_state
        try:
            traj_no = int(traj_state.traj_no)
            step_no = int(traj_state.subtraj_step_no)
        except Exception:
            return
        last = int(mj_env.th.last_step_idx(traj_no))
        if step_no >= last:
            loop_step = _clamp_step(traj_no, LOOP_STEP)
            new_ts = _make_traj_state(traj_state, traj_no=traj_no, subtraj_step_no=loop_step)
            mj_env._additional_carry = carry.replace(traj_state=new_ts)

    # ---- 3. Key callback (thread-safe via a mailbox) ----
    pending = {"idx": None}
    lock = threading.Lock()

    def key_cb(keycode):
        # GLFW keycodes: '1'=49, '2'=50; 'W'=87, 'R'=82 (upper-case as keys).
        try:
            kc = int(keycode)
        except Exception:
            kc = None
        # Print every received key so you can verify the viewer has focus.
        print(f"[live] key received: {keycode!r} (int={kc})")
        if kc in (49, 87):  # '1' or 'W'
            with lock:
                pending["idx"] = 0
            print("[live] -> walking (soft)")
        elif kc in (50, 82):  # '2' or 'R'
            with lock:
                pending["idx"] = 1
            print("[live] -> running (soft)")

    print("=== Soft-switchable evaluation (loops forever) ===")
    print(f"Walking reference (key 1 or W): {args.walking}")
    print(f"Running reference (key 2 or R): {args.running}")
    print("IMPORTANT: click on the MuJoCo viewer WINDOW first so it gets focus.")
    print("Then press 1 / W for walking, 2 / R for running. ESC to quit.")

    # ---- 4. Main loop with tracking camera ----
    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        viewer.cam.distance = args.cam_distance
        viewer.cam.elevation = args.cam_elevation
        viewer.cam.azimuth = args.cam_azimuth

        step_count = 0
        while viewer.is_running() and step_count < args.n_steps:
            # Apply pending soft-switch.
            with lock:
                req = pending["idx"]
                pending["idx"] = None
            if req is not None and req != current_idx["i"]:
                set_reference(int(req))

            # Loop reference if it ended (prevents the goal from sticking at last frame).
            maybe_loop_reference()

            # Policy step.
            rng, _rng = jax.random.split(rng)
            action, train_state = plcy_call(train_state, obs, _rng)
            action = jnp.atleast_2d(action)
            obs, _reward, _absorbing, done, _info = env.step(action)

            # Camera track: follow root xyz but keep user-specified distance/angle.
            try:
                viewer.cam.lookat[:] = np.asarray(data.qpos[:3], dtype=np.float64)
            except Exception:
                pass

            viewer.sync()
            step_count += 1

            if done and not args.no_termination:
                # Fell or drifted out: reset physics, but keep selected reference.
                selected = current_idx["i"]
                env.th.random_start = False
                env.th.fixed_start_conf = [int(selected), _clamp_step(selected, START_STEP.get(selected, 0))]
                obs = env.reset()
                # _additional_carry was rebuilt by reset; ensure selection sticks.
                set_reference(selected)

    print("Exited viewer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
