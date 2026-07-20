"""Locomotion evaluation orchestration."""

from __future__ import annotations

import subprocess
import traceback
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from musclemimic.distill.config import load_fullbody_config, motion_list_from_group, repo_root
from musclemimic.evaluation.adapters import build_adapter, resolve_n_steps
from musclemimic.evaluation.logger import (
    build_run_config_payload,
    make_timestamped_eval_dir,
    motion_result_paths,
    safe_motion_name,
    save_config_yaml,
    write_motion_artifacts,
)
from musclemimic.evaluation.light_summaries import load_or_build_rollout_analysis_summaries
from musclemimic.evaluation.metrics import compute_all_metrics, flatten_metrics_for_csv
from musclemimic.evaluation.motion_taxonomy import classify_motion_type
from musclemimic.evaluation.plotter import plot_dataset_summary, plot_per_motion
from musclemimic.evaluation.report import (
    write_failed_motions_csv,
    write_metrics_per_motion_csv,
    write_metrics_summary_csv,
)
from musclemimic.evaluation.rollout_loop import run_rollout
from musclemimic.evaluation.types import EvalConfig, MotionResult
from musclemimic.evaluation.video import finalize_rollout_video


def git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root()),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def resolve_motion_list(
    *,
    motion_path: str | None,
    motion_list_file: str | None,
    dataset_group: str | None,
) -> list[str]:
    if motion_path:
        return [motion_path]
    if motion_list_file:
        lines = Path(motion_list_file).read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    if dataset_group:
        return motion_list_from_group(dataset_group)
    raise ValueError("Provide one of motion_path, motion_list, or dataset_group")


def default_eval_force_from_hydra(hydra_cfg) -> dict[str, Any]:
    eval_block = OmegaConf.to_container(hydra_cfg.get("eval", {}), resolve=True) if hydra_cfg else {}
    force = eval_block.get("force", {}) if isinstance(eval_block, dict) else {}
    if not force:
        force = {
            "left_foot_geoms": ["l_bofoot_col1", "l_bofoot_col2"],
            "right_foot_geoms": ["r_foot_col1", "r_bofoot_col1"],
            "contact_threshold": 1.0,
            "grf_filter": "moving_average",
            "grf_filter_window": 5,
        }
    return force


class LocomotionEvalRunner:
    def __init__(self, eval_config: EvalConfig, motions: list[str], cli_args: dict[str, Any] | None = None):
        self.eval_config = eval_config
        self.motions = motions
        self.cli_args = cli_args or {}
        base = eval_config.output_dir
        self.eval_dir = make_timestamped_eval_dir(base) if not (base / "config.yaml").exists() else base
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        if eval_config.git_commit is None:
            eval_config.git_commit = git_commit_hash()
        save_config_yaml(self.eval_dir / "config.yaml", build_run_config_payload(eval_config, self.cli_args))
        self.results: list[MotionResult] = []

    def run_all(self) -> list[MotionResult]:
        for motion in self.motions:
            try:
                self.results.append(self.run_single(motion))
            except Exception as exc:
                self.results.append(
                    MotionResult(
                        motion_path=motion,
                        motion_type=classify_motion_type(motion),
                        safe_name=safe_motion_name(motion),
                        success=False,
                        metrics={"done_reason": {"value": "error", "reason": str(exc)}},
                        output_dir=motion_result_paths(self.eval_dir, motion),
                        error_message=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    )
                )
        self._write_reports()
        return self.results

    def run_single(self, motion_path: str) -> MotionResult:
        motion_dir = motion_result_paths(self.eval_dir, motion_path)
        motion_dir.mkdir(parents=True, exist_ok=True)
        adapter = build_adapter(self.eval_config, motion_path)
        try:
            traj_len = adapter.get_traj_length()
            max_steps = resolve_n_steps(self.eval_config.n_steps, traj_len)
            buffer = run_rollout(adapter, max_steps=max_steps)
            meta = adapter.meta_for_metrics()
            metrics = compute_all_metrics(
                buffer,
                meta=meta,
                force_reference_path=self.eval_config.force_reference_path,
            )
            metrics["done_reason"] = buffer.done_reason
            metrics["motion_type"] = classify_motion_type(motion_path)
            metrics["controller_type"] = self.eval_config.controller_type
            metrics["env_type"] = self.eval_config.env_type
            metrics["checkpoint_path"] = self.eval_config.checkpoint_path
            success = bool(metrics.get("official_imitation", {}).get("success", {}).get("value", 0))

            video_path = None
            if self.eval_config.save_video:
                video_path = finalize_rollout_video(
                    env=adapter.env,
                    motion_path=motion_path,
                    motion_dir=motion_dir,
                    success=success,
                    controller_type=self.eval_config.controller_type,
                )

            write_motion_artifacts(
                eval_config=self.eval_config,
                motion_dir=motion_dir,
                buffer=buffer,
                metrics=metrics,
                video_src=video_path,
            )

            analysis_dir = motion_dir / "analysis"
            load_or_build_rollout_analysis_summaries(
                analysis_dir,
                motion_dir / "rollout_data.npz",
                motion_path,
                config_name=self.eval_config.config_name,
                include_symmetry=True,
            )
            try:
                from musclemimic.evaluation.composite_score import (
                    run_composite_scoring_on_motion_dir,
                )

                report = run_composite_scoring_on_motion_dir(
                    motion_dir,
                    write_radar=bool(self.eval_config.save_plots),
                )
                self._warn_on_control_quality(report, motion_path)
            except Exception:
                pass

            if self.eval_config.save_plots:
                plot_per_motion(buffer, motion_dir)

            return MotionResult(
                motion_path=motion_path,
                motion_type=metrics["motion_type"],
                safe_name=safe_motion_name(motion_path),
                success=success,
                metrics=metrics,
                output_dir=motion_dir,
            )
        finally:
            adapter.close()

    def _warn_on_control_quality(self, report: dict[str, Any] | None, motion_path: str) -> None:
        if not report:
            return
        cq = (report.get("dimensions") or {}).get("control_quality")
        if not isinstance(cq, dict):
            return
        subs = cq.get("subscores") or {}
        missing = [sid for sid, item in subs.items() if isinstance(item, dict) and item.get("missing")]
        score = float(cq.get("score", 0.0) or 0.0)
        if missing:
            print(
                f"WARNING: control_quality missing submetrics for {motion_path}: {missing}. "
                "Check analysis/section4_torque/torque_metrics_summary.json."
            )
        elif score <= 0.0:
            raw = {sid: item.get("raw") for sid, item in subs.items() if isinstance(item, dict)}
            print(f"WARNING: control_quality scored 0 for {motion_path}; raw submetrics={raw}")

    def _write_reports(self) -> None:
        write_metrics_per_motion_csv(self.eval_dir / "metrics_per_motion.csv", self.results)
        write_failed_motions_csv(self.eval_dir / "failed_motions.csv", self.results)
        per_motion = []
        for r in self.results:
            row = flatten_metrics_for_csv(r.metrics)
            row.update({"motion_path": r.motion_path, "motion_type": r.motion_type, "success": int(r.success)})
            per_motion.append(row)
        write_metrics_summary_csv(self.eval_dir / "metrics_summary.csv", per_motion)
        if self.eval_config.save_plots:
            plot_dataset_summary(per_motion, self.eval_dir / "summary_plots")
