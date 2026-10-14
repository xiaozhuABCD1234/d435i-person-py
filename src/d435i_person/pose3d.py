# SPDX-License-Identifier: AGPL-3.0-or-later
"""把 2D 关键点 + **D435i 真深度**抬成 3D 骨架，并给出每个人的 3D 位置。

和 hik-person-py 的区别就在深度来源：那边没有深度相机，深度是 ``yolo26n-depth``
从单张图里估的（绝对尺度会飘，还得靠 fov 猜内参）；这里直接取对齐后的深度图，
公制、无模型开销，内参也是设备标定值。于是 3D 关键点是真的在测距，而不是在"看结构"。

坐标系是相机坐标系 RDF（X 右 / Y 下 / Z 前，单位米），和 rerun 的
``ViewCoordinates.RIGHT_HAND_Y_DOWN`` 一致。

纯净函数，便于单测；几何计算都在 ``geometry`` 里。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import (
    DEPTH_PATCH,
    Intrinsics,
    deproject,
    depth_valid_mask,
    lift_keypoints,
    sample_depth,
)
from .pose import KEYPOINT_NAMES_17, SKELETON_17, PoseInstance

# 用于估"这个人在哪"的对称关节对，按可靠性排序：髋比肩更稳（肩会被背包/抬手带偏）。
POSITION_PAIRS: tuple[tuple[int, int], ...] = ((11, 12), (5, 6))
# 框中心取深度失败时，再沿框的竖直线扫这几个相对高度（0=上边缘，1=下边缘）。
POSITION_SCAN = (0.5, 0.35, 0.65, 0.25, 0.75)


@dataclass
class Person3D:
    """一个人的 3D 表示：关节/骨头 + 位置（相机坐标系，米）。"""

    joints: np.ndarray  # (M, 3)
    names: list[str]
    strips: list[np.ndarray]
    position: np.ndarray  # (3,) 相机坐标；测不到就是 nan
    valid: np.ndarray  # (17,) 哪些关键点这一次是有效的

    @property
    def distance(self) -> float:
        """到相机光心的距离（米），测不到返回 nan。"""
        return (
            float(np.linalg.norm(self.position))
            if np.isfinite(self.position).all()
            else float("nan")
        )

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid))


# 表格/日志里要报的"人体具体坐标"：成对的关键点取中点，单点就用它自己。
# 顺序就是展示顺序：髀（人体质心附近，最有用）-> 肩 -> 踝（接地位置）-> 鼻（头顶高度）。
LANDMARKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hip", ("left_hip", "right_hip")),
    ("shoulder", ("left_shoulder", "right_shoulder")),
    ("ankle", ("left_ankle", "right_ankle")),
    ("nose", ("nose",)),
)


def landmarks(person, spec: tuple[tuple[str, tuple[str, ...]], ...] = LANDMARKS) -> dict:
    """从 3D 骨架里按名字取几个代表性点（相机坐标系，米）。

    ``person`` 只要带 ``joints`` / ``names`` 就行（``Person3D`` 与 ``TrackedPerson``
    都满足）。缺掉的关节直接不出现在结果里，不会用 nan 占位。
    """
    out: dict[str, np.ndarray] = {}
    joints = np.asarray(person.joints)
    names = list(person.names)
    if not len(joints):
        return out
    for label, keys in spec:
        found = [joints[names.index(key)] for key in keys if key in names]
        if not found:
            continue
        point = np.mean(found, axis=0).astype(np.float32)
        if np.isfinite(point).all():
            out[label] = point
    return out


def to_display(points3d: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[str], list]:
    """``(17,3)`` + 有效掩码 -> ``(关节 (M,3), 名字, 骨头线段列表)``。

    缺一个端点的骨头不连（宁可少画，也不把点甩到 nan 位置）。
    """
    points3d = np.asarray(points3d)
    joints = points3d[valid]
    names = [KEYPOINT_NAMES_17[i] for i in range(len(valid)) if valid[i]]
    strips = [
        np.stack([points3d[start], points3d[end]]).astype(np.float32)
        for start, end in SKELETON_17
        if valid[start] and valid[end]
    ]
    return joints, names, strips


def person_position(
    lifted: np.ndarray,
    valid: np.ndarray,
    box: tuple[float, float, float, float],
    depth: np.ndarray,
    intrinsics: Intrinsics,
    *,
    patch: int = DEPTH_PATCH,
) -> np.ndarray:
    """估一个人的 3D 位置（相机坐标，米），按可靠性依次退化：

    1. 双侧髋中点（最稳，靠近人体质心）
    2. 双侧肩中点
    3. 所有有效关节的中位数（只露半身、侧面时用）
    4. 框中心附近的深度（关键点全丢时兜底；沿框竖线扫几个高度取第一个有效值）

    全失败返回 (nan, nan, nan)。纯函数。
    """
    for left, right in POSITION_PAIRS:
        if valid[left] and valid[right]:
            return np.asarray((lifted[left] + lifted[right]) * 0.5, dtype=np.float32)

    if valid.any():
        return np.asarray(np.median(lifted[valid], axis=0), dtype=np.float32)

    x1, y1, x2, y2 = box
    center_x = (x1 + x2) * 0.5
    for ratio in POSITION_SCAN:
        d = sample_depth(depth, center_x, y1 + (y2 - y1) * ratio, patch)
        if np.isfinite(d) and d > 0:
            return np.asarray(
                deproject(center_x, y1 + (y2 - y1) * ratio, d, intrinsics), dtype=np.float32
            )
    return np.full(3, np.nan, dtype=np.float32)


def pose_to_3d(
    pose: PoseInstance,
    depth: np.ndarray,
    intrinsics: Intrinsics,
    *,
    threshold: float,
    patch: int = DEPTH_PATCH,
) -> Person3D:
    """一个人 -> ``Person3D``；不做滤波（滤波见 ``track``）。"""
    lifted = lift_keypoints(pose.keypoints, depth, intrinsics, patch=patch)
    valid = depth_valid_mask(lifted, pose.kpt_conf, threshold)
    joints, names, strips = to_display(lifted, valid)
    return Person3D(
        joints=joints,
        names=names,
        strips=strips,
        position=person_position(lifted, valid, pose.xyxy, depth, intrinsics, patch=patch),
        valid=valid,
    )
