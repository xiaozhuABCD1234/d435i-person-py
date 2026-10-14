# SPDX-License-Identifier: AGPL-3.0-or-later
"""像素格式名与颜色转换。

D435i 的彩色流直接要 ``rs.format.bgr8``（相机支持），所以本包内部统一是 OpenCV 的
**BGR** 顺序，和 hik-person-py 的约定一致；只有进 rerun 前要换一次 RGB。

深度是 z16：1 个单位 = depth scale 米（D435i 是 0.001），换算在
``geometry.depth_to_meters`` 里做，无效值统一变 nan。
"""

from __future__ import annotations

import cv2
import numpy as np

# pyrealsense2 的 rs.format 枚举值 -> 名字（rs.format 是 int 枚举，这里只列常用的）。
# 现状是从 pyrealsense2 自己取出名字（str(format) 的末段），这份表只是兜底。
FORMAT_NAMES = {
    0: "any",
    1: "z16",
    2: "disparity16",
    3: "xyz32f",
    6: "y8",
    7: "y16",
    8: "raw8",
    9: "rgb8",
    10: "bgr8",
    11: "rgba8",
    12: "bgra8",
    16: "yuyv",
    18: "uyvy",
    19: "raw16",
}


def format_name(value) -> str:
    """rs.format -> 名字；认不出来就退回 ``str()`` 的末段。"""
    text = str(value)
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return FORMAT_NAMES.get(int(value), text)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """BGR -> RGB。

    rerun 的 rr.Image 把 3 通道数据当 RGB 解读，所以本包内部的 BGR 帧在进 rerun 前
    必须换一次，否则红蓝互换。
    """
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
