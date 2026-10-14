# SPDX-License-Identifier: AGPL-3.0-or-later
"""几何工具（内参/深度换算/反投影/点云）的单元测试 —— 不需要相机。"""

from __future__ import annotations

import numpy as np
import pytest

from d435i_person.geometry import (
    Intrinsics,
    deproject,
    depth_to_meters,
    depth_valid_mask,
    lift_keypoints,
    pointcloud,
    sample_depth,
)

# 640×480、fx=fy=500 的假内参：手算坐标方便
K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


def flat_depth(value: float = 2.0, *, width: int = 640, height: int = 480) -> np.ndarray:
    return np.full((height, width), value, dtype=np.float32)


# --------------------------------------------------------------------- 深度换算
def test_depth_to_meters_scales_and_nans_invalid() -> None:
    raw = np.array([[1000, 0], [250, 65535]], dtype=np.uint16)
    depth = depth_to_meters(raw, 0.001)
    assert depth[0, 0] == pytest.approx(1.0)
    assert np.isnan(depth[0, 1])  # 0 = 没测到，必须是 nan 而不是 0 米
    assert depth[1, 0] == pytest.approx(0.25)
    assert depth[1, 1] == pytest.approx(65.535)
    assert depth.dtype == np.float32


def test_depth_to_meters_respects_device_scale() -> None:
    raw = np.array([[10000]], dtype=np.uint16)
    assert depth_to_meters(raw, 0.0001)[0, 0] == pytest.approx(1.0)


# ----------------------------------------------------------------------- 内参
def test_intrinsics_from_fov_90_degrees() -> None:
    intrinsics = Intrinsics.from_fov(640, 480, 90.0)
    assert intrinsics.fx == pytest.approx(320.0)
    assert intrinsics.cx == pytest.approx(320.0)
    assert intrinsics.width == 640


def test_intrinsics_scaled_to_doubles_pixels() -> None:
    scaled = K.scaled_to(1280, 960)
    assert scaled.fx == pytest.approx(1000.0)
    assert scaled.cy == pytest.approx(480.0)
    # 同尺寸时是恒等变换（返回自己，省一次分配）
    assert K.scaled_to(640, 480) is K


def test_intrinsics_fov_of_d435i_depth_default() -> None:
    # 实测 D435i 深度 640x480: fx=384.06 -> 水平 FOV 约 79.6°
    real = Intrinsics(fx=384.06, fy=384.06, cx=321.03, cy=234.35, width=640, height=480)
    horizontal, vertical = real.fov_deg()
    assert horizontal == pytest.approx(79.6, abs=0.2)
    assert vertical == pytest.approx(64.0, abs=0.3)


# ------------------------------------------------------------------- 反投影
def test_deproject_center_is_on_the_optical_axis() -> None:
    assert deproject(320.0, 240.0, 2.0, K) == pytest.approx((0.0, 0.0, 2.0))


def test_deproject_offsets_scale_with_depth() -> None:
    x, y, z = deproject(420.0, 140.0, 2.0, K)
    assert (x, y, z) == pytest.approx((0.4, -0.4, 2.0))


@pytest.mark.parametrize("bad", [(np.nan, 240.0, 2.0), (320.0, np.inf, 2.0), (320.0, 240.0, 0.0)])
def test_deproject_rejects_invalid(bad) -> None:
    assert all(np.isnan(v) for v in deproject(*bad, K))


# ------------------------------------------------------------------- 取深度
def test_sample_depth_takes_median_of_patch() -> None:
    depth = flat_depth(2.0, width=11, height=11)
    depth[5, 5] = 100.0  # 前景/背景边界上的单个离群值
    assert sample_depth(depth, 5.0, 5.0, patch=3) == pytest.approx(2.0)
    # 窗口里大多数像素都变了 -> 中位数跟着走
    depth[4:7, 4:7] = 5.0
    assert sample_depth(depth, 5.0, 5.0, patch=3) == pytest.approx(5.0)


def test_sample_depth_ignores_nan_and_out_of_range() -> None:
    depth = flat_depth(np.nan, width=4, height=4)
    assert np.isnan(sample_depth(depth, 2.0, 2.0))
    assert np.isnan(sample_depth(depth, -5.0, 2.0))
    assert np.isnan(sample_depth(depth, np.nan, 2.0))


# ------------------------------------------------------------------- 抬 3D
def test_lift_keypoints_maps_pixels_to_camera_frame() -> None:
    keypoints = np.array([[320.0, 240.0], [420.0, 140.0]], dtype=np.float32)
    lifted = lift_keypoints(keypoints, flat_depth(2.0), K)
    assert lifted[0] == pytest.approx((0.0, 0.0, 2.0), abs=1e-5)
    assert lifted[1] == pytest.approx((0.4, -0.4, 2.0), abs=1e-5)


def test_lift_keypoints_nans_where_depth_missing() -> None:
    depth = flat_depth(2.0)
    depth[238:243, 318:323] = np.nan  # 把 5×5 的取样窗口整块挖掉
    lifted = lift_keypoints(np.array([[320.0, 240.0], [420.0, 140.0]]), depth, K)
    assert np.isnan(lifted[0]).all()
    assert np.isfinite(lifted[1]).all()


def test_valid_mask_needs_both_confidence_and_depth() -> None:
    lifted = np.array([[0.0, 0.0, 2.0], [np.nan, np.nan, np.nan]], dtype=np.float32)
    conf = np.array([0.4, 0.9], dtype=np.float32)
    mask = depth_valid_mask(lifted, conf, 0.5)
    assert list(mask) == [False, False]
    conf = np.array([0.9, 0.9], dtype=np.float32)
    assert list(depth_valid_mask(lifted, conf, 0.5)) == [True, False]


# ------------------------------------------------------------------- 点云
def test_pointcloud_shapes_colors_and_limits() -> None:
    depth = flat_depth(2.0, width=8, height=8)
    depth[0, 0] = np.nan  # 无效
    depth[7, 7] = 9.0  # 超距离
    color = np.zeros((8, 8, 3), np.uint8)
    color[..., 0] = 10  # B
    color[..., 2] = 200  # R

    positions, colors = pointcloud(depth, color, K_scaled(), stride=1, max_distance=4.0)
    assert positions.shape == (62, 3)  # 64 - 无效 - 超距离
    assert colors.shape == (62, 3)
    assert positions.dtype == np.float32
    # BGR -> RGB：原来的 B=10 现在落在最后一个通道
    assert colors[0].tolist() == [200, 0, 10]


def test_pointcloud_stride_subsamples() -> None:
    depth = flat_depth(2.0, width=8, height=8)
    color = np.zeros((8, 8, 3), np.uint8)
    full, _ = pointcloud(depth, color, K_scaled(), stride=1)
    quarter, _ = pointcloud(depth, color, K_scaled(), stride=2)
    assert len(quarter) == 16
    assert len(full) == 64


def K_scaled() -> Intrinsics:
    """8×8 图上也用得上的内参（主点在图中心附近）。"""
    return Intrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0, width=8, height=8)
