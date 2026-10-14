# SPDX-License-Identifier: AGPL-3.0-or-later
"""相机层的纯函数部分（分辨率解析、帧容器、懒加载）—— 不需要插相机。"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from d435i_person.camera import Camera, DeviceInfo, Frame, parse_size
from d435i_person.geometry import Intrinsics

K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


# ------------------------------------------------------------------ 分辨率解析
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("640x480", (640, 480)),
        ("848X480", (848, 480)),
        ("1280*720", (1280, 720)),
        ("1920,1080", (1920, 1080)),
        (" 640 x 480 ", (640, 480)),
    ],
)
def test_parse_size_accepts_common_spellings(text: str, expected: tuple[int, int]) -> None:
    assert parse_size(text, (1, 1)) == expected


def test_parse_size_falls_back_to_default() -> None:
    assert parse_size("", (640, 480)) == (640, 480)
    assert parse_size(None, (848, 480)) == (848, 480)


@pytest.mark.parametrize("bad", ["640", "640x480x30", "aa x bb", "0x480", "640x-1"])
def test_parse_size_rejects_junk(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_size(bad, (640, 480))


# ------------------------------------------------------------------ 帧容器
def test_frame_exposes_size_from_image() -> None:
    frame = Frame(
        bgr=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=np.full((480, 640), 2.0, dtype=np.float32),
        frame_number=7,
        timestamp_ms=1234.5,
        host_time=1.0,
        intrinsics=K,
        exposure_us=200.0,
        gain=32.0,
        depth_valid_ratio=0.93,
    )
    assert (frame.width, frame.height) == (640, 480)


def test_device_info_is_readable() -> None:
    text = str(
        DeviceInfo(
            index=0,
            name="Intel RealSense D435I",
            serial="117322075393",
            firmware="5.17.3.10",
            usb="3.2",
            product_line="D400",
        )
    )
    assert "D435I" in text and "117322075393" in text and "3.2" in text


# ------------------------------------------------------------------ 懒加载
def test_camera_module_does_not_import_pyrealsense2_at_import_time() -> None:
    """pyrealsense2 必须延迟导入：这样没相机/没装 SDK 的机器照样能跑单测与 lint。"""
    code = "import sys, d435i_person.camera; print('pyrealsense2' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_configure_before_start_is_enforced() -> None:
    camera = Camera()
    camera._pipe = object()  # 假装已经启动
    with pytest.raises(Exception, match="pipeline 已经启动"):
        camera.configure()
    camera._pipe = None


def test_bad_exposure_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="exposure_mode"):
        Camera().configure(exposure_mode="auto")
