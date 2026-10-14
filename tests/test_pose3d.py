# SPDX-License-Identifier: AGPL-3.0-or-later
"""真深度抬 3D 的单元测试 —— 不需要相机，用的是手算得出的假深度图。"""

from __future__ import annotations

import numpy as np
import pytest

from d435i_person.geometry import Intrinsics
from d435i_person.pose import NUM_KEYPOINTS, SKELETON_17, PoseInstance
from d435i_person.pose3d import Person3D, pose_to_3d, to_display

K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
CONF = np.full(NUM_KEYPOINTS, 0.9, dtype=np.float32)


def flat_depth(value: float = 2.0) -> np.ndarray:
    return np.full((480, 640), value, dtype=np.float32)


def make_pose(points: np.ndarray | None = None, conf: np.ndarray | None = None) -> PoseInstance:
    """默认关键点：X 依次拉开、Y 固定，全都在画面内。"""
    if points is None:
        points = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        points[:, 0] = 300.0 + np.arange(NUM_KEYPOINTS)
        points[:, 1] = 240.0
    return PoseInstance(
        conf=0.77,
        xyxy=(300.0, 100.0, 340.0, 400.0),
        keypoints=points,
        kpt_conf=CONF if conf is None else conf,
    )


# ------------------------------------------------------------------- to_display
def test_to_display_keeps_only_valid() -> None:
    points = np.arange(NUM_KEYPOINTS * 3, dtype=np.float32).reshape(NUM_KEYPOINTS, 3)
    valid = np.zeros(NUM_KEYPOINTS, bool)
    valid[11] = valid[12] = True
    joints, names, strips = to_display(points, valid)
    assert joints.shape == (2, 3)
    assert names == ["left_hip", "right_hip"]
    # 只有 (11,12) 这根骨头两端都有效
    assert len(strips) == 1
    assert np.allclose(strips[0], np.stack([points[11], points[12]]))


def test_to_display_drops_bones_with_a_nan_end() -> None:
    points = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    valid = np.ones(NUM_KEYPOINTS, bool)
    valid[15] = False
    strips = to_display(points, valid)[2]
    assert len(strips) == len(SKELETON_17) - 1


# ------------------------------------------------------------------- pose_to_3d
def test_pose_to_3d_uses_true_depth() -> None:
    pose = make_pose()
    person = pose_to_3d(pose, flat_depth(3.0), K, threshold=0.5)
    assert isinstance(person, Person3D)
    assert person.valid_count == NUM_KEYPOINTS
    # 深度 3 m -> 所有关节的 Z 都是 3 m（真深度，不是模型估的）
    assert np.allclose(person.joints[:, 2], 3.0)
    assert person.distance == pytest.approx(np.linalg.norm(person.position))


def test_pose_to_3d_marks_keypoints_without_depth_invalid() -> None:
    depth = flat_depth(2.0)
    depth[238:243, 298:303] = np.nan  # 0 号关键点（300,240）的取样窗口整块没测到
    person = pose_to_3d(make_pose(), depth, K, threshold=0.5)
    assert person.valid_count == NUM_KEYPOINTS - 1
    assert "nose" not in person.names


def test_low_confidence_keypoint_needs_no_depth() -> None:
    conf = CONF.copy()
    conf[0] = 0.1
    person = pose_to_3d(make_pose(conf=conf), flat_depth(2.0), K, threshold=0.5)
    assert "nose" not in person.names
    assert person.valid_count == NUM_KEYPOINTS - 1


# ------------------------------------------------------------- person_position
def test_position_prefers_hip_midpoint() -> None:
    pose = make_pose()
    person = pose_to_3d(pose, flat_depth(2.5), K, threshold=0.5)
    # 11/12 号关键点的 x 是 300+11 和 300+12，中点就是 300+11.5
    mid_x = 300.0 + (11 + 12) / 2
    expected = np.array([(mid_x - 320.0) * 2.5 / 500.0, 0.0, 2.5], dtype=np.float32)
    assert person.position == pytest.approx(expected, abs=1e-5)


def test_position_falls_back_to_shoulders_without_hips() -> None:
    conf = CONF.copy()
    conf[11] = conf[12] = 0.1  # 髋看不见（人站在桌子后）
    pose = make_pose(conf=conf)
    person = pose_to_3d(pose, flat_depth(2.0), K, threshold=0.5)
    xs = (300.0 + 5, 300.0 + 6)
    assert person.position[0] == pytest.approx((np.mean(xs) - 320.0) * 2.0 / 500.0, abs=1e-5)


def test_position_falls_back_to_median_of_valid_joints() -> None:
    conf = CONF.copy()
    conf[[5, 6, 11, 12]] = 0.0  # 肩和髋都没了 -> 用剩下的有效关节中位数
    pose = make_pose(conf=conf)
    person = pose_to_3d(pose, flat_depth(2.0), K, threshold=0.5)
    assert person.valid_count > 0
    assert np.isfinite(person.position).all()


def test_position_falls_back_to_box_center_depth() -> None:
    conf = np.zeros(NUM_KEYPOINTS, dtype=np.float32)  # 关键点全丢
    pose = make_pose(conf=conf)
    person = pose_to_3d(pose, flat_depth(2.0), K, threshold=0.5)
    assert person.valid_count == 0
    # 框中心 x=(300+340)/2=320 正好在主点上；扫到 0.5 高度 y=250 -> y 偏移 (250-240)*2/500
    assert person.position == pytest.approx((0.0, 0.04, 2.0), abs=1e-5)


def test_position_is_nan_without_any_depth() -> None:
    conf = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
    pose = make_pose(conf=conf)
    depth = np.full((480, 640), np.nan, dtype=np.float32)
    person = pose_to_3d(pose, depth, K, threshold=0.5)
    assert np.isnan(person.position).all()
    assert np.isnan(person.distance)
