"""Unified rollout loop for locomotion evaluation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from musclemimic.evaluation.types import RolloutBuffer, StepResult

if TYPE_CHECKING:
    from musclemimic.evaluation.adapters import BaseEvalAdapter


def infer_done_reason(
    *,
    done: bool,
    absorbing: bool,
    info: dict,
    step: int,
    max_steps: int,
    traj_length: int,
    root_z: float | None,
    fall_height_threshold: float = 0.5,
    contact_explosion_threshold: float = 5000.0,
    peak_vgrf: float | None = None,
) -> str:
    if root_z is not None and (math.isnan(root_z) or math.isinf(root_z)):
        return "nan"
    if peak_vgrf is not None and peak_vgrf > contact_explosion_threshold:
        return "contact_explosion"
    if not done:
        if step >= max_steps - 1 or step >= traj_length - 1:
            return "completed"
        return "completed"
    if bool(info.get("terminated", False)):
        handler = str(info.get("terminal_handler", ""))
        if "root" in handler.lower():
            return "root_error_exceeded"
        if "site" in handler.lower():
            return "site_error_exceeded"
        return "site_error_exceeded"
    if absorbing and root_z is not None and root_z < fall_height_threshold:
        return "fall"
    if step >= traj_length - 1:
        return "completed"
    if step >= max_steps - 1:
        return "timeout"
    return "unknown"


def run_rollout(adapter: "BaseEvalAdapter", max_steps: int | None = None) -> RolloutBuffer:
    motion_path = adapter.motion_path
    traj_len = int(adapter.get_traj_length())
    limit = int(max_steps if max_steps is not None else traj_len)
    limit = max(1, min(limit, traj_len))

    adapter.reset(motion_path)
    buffer = RolloutBuffer(motion_path=motion_path, dt=float(adapter.dt), traj_length=traj_len)

    for step in range(limit):
        result: StepResult = adapter.step()
        contact = adapter.get_contact_forces()
        torques = adapter.get_joint_torques()
        root_pos = adapter.get_root_pos()
        root_quat = adapter.get_root_quat()
        ref = result.reference

        raw_l = float(contact.get("left_vertical_GRF_raw", contact.get("left_vertical_GRF", 0.0)))
        raw_r = float(contact.get("right_vertical_GRF_raw", contact.get("right_vertical_GRF", 0.0)))
        filt_l = float(contact.get("left_vertical_GRF", raw_l))
        filt_r = float(contact.get("right_vertical_GRF", raw_r))
        peak_v = max(abs(raw_l), abs(raw_r), abs(filt_l), abs(filt_r))
        buffer.append_step(
            t=step * buffer.dt,
            qpos=adapter.get_qpos(),
            qvel=adapter.get_qvel(),
            root_pos=root_pos,
            root_quat=root_quat,
            ref_root_pos=ref.get("root_pos"),
            ref_qpos=ref.get("qpos"),
            site_pos=adapter.get_site_pos(),
            ref_site_pos=ref.get("site_pos"),
            left_grf=filt_l,
            right_grf=filt_r,
            left_grf_raw=raw_l,
            right_grf_raw=raw_r,
            left_grf_world=contact.get("left_foot_GRF_world", np.zeros(3)),
            right_grf_world=contact.get("right_foot_GRF_world", np.zeros(3)),
            contact_left=bool(contact.get("left_contact_bool", False)),
            contact_right=bool(contact.get("right_contact_bool", False)),
            joint_torque=torques.get("joint_torque", np.zeros(0)),
            prosthesis_tau=torques.get("prosthesis_tau", np.zeros(0)),
            reward=float(result.reward),
            done=bool(result.done),
            info=result.info,
        )
        if adapter.save_video:
            adapter.render(record=True)

        if bool(result.done):
            break

    root_z = float(buffer.root_pos[-1][2]) if buffer.root_pos else None
    peak_vgrf = 0.0
    if buffer.left_grf:
        peak_vgrf = max(max(buffer.left_grf), max(buffer.right_grf))

    buffer.done_reason = infer_done_reason(
        done=bool(buffer.done_flags[-1]) if buffer.done_flags else False,
        absorbing=bool(buffer.info_steps[-1].get("terminated", False)) if buffer.info_steps else False,
        info=buffer.info_steps[-1] if buffer.info_steps else {},
        step=buffer.steps,
        max_steps=limit,
        traj_length=traj_len,
        root_z=root_z,
        peak_vgrf=peak_vgrf,
    )
    if buffer.done_reason == "completed" and buffer.steps < int(0.99 * traj_len):
        if buffer.info_steps and buffer.info_steps[-1].get("terminated"):
            pass
        elif buffer.steps < traj_len - 1 and not buffer.done_flags[-1]:
            buffer.done_reason = "completed"
    return buffer
