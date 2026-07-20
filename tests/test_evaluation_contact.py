"""Unit tests for foot contact / GRF extraction."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from musclemimic.evaluation.contact_extractor import (
    FootContactConfig,
    FootContactExtractor,
    analyze_stance_grf_profile,
    contact_force_to_world,
    resolve_geom_ids,
    segment_stance_phases,
)


@pytest.fixture
def tiny_model():
    xml = """
    <mujoco model="test">
      <worldbody>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        <body name="left" pos="0 0 0.05">
          <geom name="l_foot" type="sphere" size="0.05" friction="1 0.005 0.0001"/>
        </body>
        <body name="right" pos="0.15 0 0.05">
          <geom name="r_foot" type="sphere" size="0.05" friction="1 0.005 0.0001"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def test_resolve_geom_ids_ok(tiny_model):
    mapping = resolve_geom_ids(tiny_model, ("l_foot",), "left")
    assert "l_foot" in mapping


def test_resolve_geom_ids_raises_with_candidates(tiny_model):
    with pytest.raises(ValueError, match="Foot-related geoms"):
        resolve_geom_ids(tiny_model, ("missing_geom",), "left")


def test_contact_force_to_world_uses_z_component():
    # Contact x-axis aligned with world +z (floor–foot contact in prior probe).
    frame = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    f_world = contact_force_to_world(np.array([120.0, 0.0, 0.0]), frame)
    assert abs(f_world[2] - 120.0) < 1e-6
    assert abs(f_world[0]) < 1e-6 and abs(f_world[1]) < 1e-6


def test_filtered_differs_from_raw_with_filter(tiny_model):
    data = mujoco.MjData(tiny_model)
    for _ in range(50):
        mujoco.mj_step(tiny_model, data)
    cfg = FootContactConfig(
        left_foot_geoms=("l_foot",),
        right_foot_geoms=("r_foot",),
        grf_filter="moving_average",
        grf_filter_window=7,
    )
    ext = FootContactExtractor(tiny_model, cfg, dt=0.01)
    raw_vals = []
    filt_vals = []
    for _ in range(10):
        mujoco.mj_step(tiny_model, data)
        o = ext.extract(data)
        raw_vals.append(float(o["left_vertical_GRF_raw"]))
        filt_vals.append(float(o["left_vertical_GRF"]))
    assert "left_vertical_GRF" in o
    assert "left_vertical_GRF_raw" in o


def test_stance_peak_analysis_synthetic():
    t = np.linspace(0, 4 * np.pi, 400)
    v = np.maximum(0.0, 500.0 * np.sin(t) ** 2)
    contact = v > 50.0
    profile = analyze_stance_grf_profile(v, contact)
    assert profile["n_stances"] >= 2
    assert profile["peak_count"] is not None


def test_segment_stance_phases():
    mask = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0], dtype=bool)
    phases = segment_stance_phases(mask, min_steps=3)
    assert phases == [(2, 5), (7, 11)]
