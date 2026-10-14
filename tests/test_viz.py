# SPDX-License-Identifier: AGPL-3.0-or-later
"""rerun 可视化的集成测试 —— 不弹窗口，写 .rrd 到临时目录。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from d435i_person.camera import Frame
from d435i_person.geometry import Intrinsics, pointcloud
from d435i_person.pose import NUM_KEYPOINTS, PoseInstance
from d435i_person.track import PersonTracker
from d435i_person.viz import (
    ALARM_COLOR,
    ALARM_ORIGIN,
    DEFAULT_METRIC_GROUPS,
    PALETTE,
    RerunView,
    color_for,
    fall_alert,
    fall_alert_panel,
    person_color,
    value_line,
)
from d435i_person.viz import (
    _parse_metric_views as parse_metric_views,
)
from d435i_person.world import CameraToWorld

K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
CONF = np.full(NUM_KEYPOINTS, 0.9, dtype=np.float32)


def make_frame() -> Frame:
    rng = np.random.default_rng(0)
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    depth[0:20, 0:20] = np.nan
    return Frame(
        bgr=rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        depth=depth,
        frame_number=1,
        timestamp_ms=1.0,
        host_time=0.1,
        intrinsics=K,
        exposure_us=200.0,
        gain=32.0,
        depth_valid_ratio=0.99,
    )


def make_people() -> list:
    points = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    points[:, 0] = 310.0 + np.arange(NUM_KEYPOINTS)
    points[:, 1] = 200.0 + np.arange(NUM_KEYPOINTS) * 4
    pose = PoseInstance(
        conf=0.81,
        xyxy=(280.0, 100.0, 360.0, 430.0),
        keypoints=points,
        kpt_conf=CONF.copy(),
    )
    tracker = PersonTracker(trail_interval=0.0)
    tracker.update([pose], make_frame().depth, K, threshold=0.5, t=0.0)
    return tracker.update([pose], make_frame().depth, K, threshold=0.5, t=0.1)


def make_view(tmp_path: Path, **kwargs) -> RerunView:
    return RerunView(save_path=str(tmp_path / "test.rrd"), spawn=False, **kwargs)


def test_color_for_is_stable_and_wraps_around() -> None:
    assert color_for(1) == color_for(1)
    assert color_for(1) != color_for(2)
    assert color_for(1) == PALETTE[0]
    assert color_for(len(PALETTE) + 1) == PALETTE[0]


def test_fallen_person_turns_red_and_is_flagged() -> None:
    """摔倒：整体换警报色（比“去曲线框里找线”快得多），数值行带 ⚠ 与躯干角。"""
    person = make_people()[0]
    assert person_color(person) == color_for(person.id)
    assert fall_alert(person) == ""

    person.fall_level, person.lying = 1, True  # 倒地中（还没确认）
    assert fall_alert(person) == "倒地中"
    assert person_color(person) == color_for(person.id)

    person.fall_level = 1  # 只是挨了一下冲击（跳、快速蹲下）就不标注
    person.lying = False
    assert fall_alert(person) == ""

    person.fall_level = 2
    person.metrics["torso_angle"] = 88.6
    assert fall_alert(person) == "⚠ 摔倒 89°"
    assert person_color(person) == ALARM_COLOR
    assert "⚠ 摔倒" in value_line(person)


def test_alarm_box_replaces_coverage_in_the_default_grid() -> None:
    """默认格子：五个曲线组 + 摔倒警报，coverage 不再占格子（数据照记）。"""
    origins = [origin for origin, _, _ in parse_metric_views("all")]
    assert ALARM_ORIGIN in origins
    assert len(origins) == len(DEFAULT_METRIC_GROUPS)
    assert all("quality" not in origin for origin in origins)
    assert parse_metric_views("quality")[0][0] == "person/metrics/quality"  # 仍可单独要
    assert "person/metrics/quality" in [o for o, _, _ in parse_metric_views("all,quality")]
    assert parse_metric_views("speed,alarm") == [
        ("person/metrics/speed", "速度", "speed"),
        (ALARM_ORIGIN, "摔倒警报", "alarm"),
    ]


def test_fall_alert_panel_states() -> None:
    people = make_people()
    assert "画面内无人" in fall_alert_panel([])
    assert "✅ 正常" in fall_alert_panel(people)

    person = people[0]
    person.metrics["torso_angle"] = 88.6
    person.fall_level, person.lying, person.fall_since = 1, True, 10.0
    down = fall_alert_panel(people, now=12.5)
    assert down.startswith("# ⏳ 倒地中 #")
    assert "躯干 89°" in down and "已 2.5 s" in down

    person.fall_level, person.fall_state = 2, "摔倒"
    alarm = fall_alert_panel(people, history=["10.2s ⚠ #1 检测到摔倒"], now=12.5)
    assert alarm.startswith("# ⚠ 摔倒 #")
    assert "摔倒 ·" in alarm  # 状态名跟在 id 后面
    assert "最近：10.2s ⚠ #1 检测到摔倒" in alarm
    assert "倒地中" not in alarm.split("\n")[1]  # 确认后不再写“倒地中”


def test_fall_alert_panel_marks_multiple_people() -> None:
    first, second = make_people()[0], make_people()[0]
    second.id = first.id + 1
    for person in (first, second):
        person.fall_level = 2
        person.fall_state = "摔倒"
    panel = fall_alert_panel([first, second])
    assert "⚠ 摔倒" in panel and "（多人）" in panel
    assert len([line for line in panel.split("\n") if line.startswith("- ")]) == 2


def test_log_frame_writes_an_rrd(tmp_path: Path) -> None:
    rrd = tmp_path / "test.rrd"
    view = RerunView(save_path=str(rrd), spawn=False)
    frame = make_frame()
    view.log_frame(frame, make_people())
    assert rrd.is_file() and rrd.stat().st_size > 0


def test_log_frame_without_people(tmp_path: Path) -> None:
    rrd = tmp_path / "empty.rrd"
    view = RerunView(save_path=str(rrd), spawn=False)
    view.log_frame(make_frame(), [])
    assert rrd.stat().st_size > 0


def test_full_pipeline_entities_are_recorded(tmp_path: Path) -> None:
    frame = make_frame()
    people = make_people()
    view = RerunView(
        save_path=str(tmp_path / "full.rrd"),
        spawn=False,
        world=CameraToWorld(height=1.4, pitch_deg=10.0),
        show_world_view=True,
    )
    view.log_frame(frame, people)
    view.log_depth(frame.depth)
    positions, colors = pointcloud(frame.depth, frame.bgr, K, stride=4)
    view.log_points(positions, colors)
    view.log_people3d(people)
    assert (tmp_path / "full.rrd").stat().st_size > 0


def test_log_depth_handles_all_nan(tmp_path: Path) -> None:
    make_view(tmp_path).log_depth(np.full((480, 640), np.nan, dtype=np.float32))


def test_log_points_handles_empty_cloud(tmp_path: Path) -> None:
    make_view(tmp_path, world=CameraToWorld()).log_points(
        np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    )


def test_log_people3d_handles_people_without_joints(tmp_path: Path) -> None:
    view = make_view(tmp_path, world=CameraToWorld())
    people = make_people()
    for person in people:
        person.joints = np.zeros((0, 3), np.float32)
        person.names = []
        person.strips = []
        person.position = np.full(3, np.nan, np.float32)
        person.trail = np.zeros((0, 3), np.float32)
    view.log_people3d(people)


def test_pointcloud_colors_match_positions(tmp_path: Path) -> None:
    frame = make_frame()
    positions, colors = pointcloud(frame.depth, frame.bgr, K, stride=8)
    assert len(positions) == len(colors)
    assert positions.shape[1] == 3 and colors.shape[1] == 3


@pytest.mark.parametrize("quality", [0, 75])
def test_jpeg_quality_zero_and_positive_both_work(tmp_path: Path, quality: int) -> None:
    view = make_view(tmp_path, jpeg_quality=quality)
    view.log_frame(make_frame(), make_people())
