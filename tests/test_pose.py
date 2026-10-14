# SPDX-License-Identifier: AGPL-3.0-or-later
"""姿态估计的纯函数部分（骨架常量与过滤）—— 不加载 YOLO。"""

from __future__ import annotations

import numpy as np

from d435i_person.pose import (
    KEYPOINT_NAMES_17,
    NUM_KEYPOINTS,
    SKELETON_17,
    PoseInstance,
    bones,
    visible_keypoints,
)


def test_skeleton_is_zero_based_and_in_range() -> None:
    assert NUM_KEYPOINTS == 17
    for start, end in SKELETON_17:
        assert 0 <= start < NUM_KEYPOINTS
        assert 0 <= end < NUM_KEYPOINTS
        assert start != end
    # 19 根骨头，和 ultralytics 的 skeleton 一致
    assert len(SKELETON_17) == 19


def test_skeleton_connects_both_sides() -> None:
    pairs = set(SKELETON_17)
    # 左右对称：每根骨头换个左右手/脚也该在表里
    for start, end in SKELETON_17:
        mirrored = (mirror(start), mirror(end))
        assert mirrored in pairs or tuple(reversed(mirrored)) in pairs


def mirror(index: int) -> int:
    """把关键点编号换成对侧的编号（鼻子/眼/耳/肩/肘/腕/髋/膝/踝 都是成对的）。"""
    pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    for left, right in pairs:
        if index == left:
            return right
        if index == right:
            return left
    return index


def test_bones_skip_low_confidence_ends() -> None:
    keypoints = np.stack(
        [
            np.arange(NUM_KEYPOINTS, dtype=np.float32),
            np.arange(NUM_KEYPOINTS, dtype=np.float32) * 2,
        ],
        axis=1,
    )
    conf = np.full(NUM_KEYPOINTS, 0.9, dtype=np.float32)
    assert len(bones(keypoints, conf, threshold=0.5)) == len(SKELETON_17)
    conf[11] = 0.1  # 左髋丢了 -> 与它相连的骨头都不画
    drawn = bones(keypoints, conf, threshold=0.5)
    assert len(drawn) == len(SKELETON_17) - 3
    for strip in drawn:
        assert not np.allclose(strip[0], keypoints[11])
        assert not np.allclose(strip[1], keypoints[11])


def test_visible_keypoints_returns_names() -> None:
    keypoints = np.arange(NUM_KEYPOINTS * 2, dtype=np.float32).reshape(NUM_KEYPOINTS, 2)
    conf = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
    conf[0] = 0.9
    conf[16] = 0.6
    points, names = visible_keypoints(keypoints, conf, threshold=0.5)
    assert points.shape == (2, 2)
    assert names == ["nose", "right_ankle"]


def test_pose_instance_is_plain_data() -> None:
    pose = PoseInstance(
        conf=0.8,
        xyxy=(1.0, 2.0, 3.0, 4.0),
        keypoints=np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32),
        kpt_conf=np.ones(NUM_KEYPOINTS, dtype=np.float32),
    )
    assert KEYPOINT_NAMES_17[0] == "nose"
    assert pose.xyxy[2] == 3.0
