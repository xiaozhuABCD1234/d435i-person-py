# SPDX-License-Identifier: AGPL-3.0-or-later
"""相机自检：不加载 YOLO，直接看 D435i 的流配置、内参、深度质量与帧率。

    uv run python tools/check_camera.py                  # 默认跑 60 帧
    uv run python tools/check_camera.py --frames 120 --save out/
    uv run python tools/check_camera.py --exposure-mode manual --exposure-us 8000

会打印：

* 枚举到的设备（名字/序列号/固件/USB 代次）
* 生效的流配置、内参、FOV、depth scale
* 实测帧率、有效深度占比、深度分位数、曝光/增益
* ``--save`` 时把最后一帧的彩色/深度存成 PNG（深度用 Turbo 伪彩）

出问题时的排查顺序：设备在不在 -> 是不是 USB3 -> 有效深度占比 -> 帧率。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from d435i_person.camera import Camera, RsError, list_devices, parse_size
from d435i_person.config import filter_spec, load_settings
from d435i_person.geometry import pointcloud


def main() -> int:
    parser = argparse.ArgumentParser(description="D435i 相机自检（不跑 YOLO）")
    parser.add_argument("--config", default=None, help="读 config.toml 里的 [camera]/[filters]")
    parser.add_argument("--frames", type=int, default=60, help="跑多少帧")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--color-size", default=None)
    parser.add_argument("--depth-size", default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--exposure-mode", default=None, choices=("sdk", "manual"))
    parser.add_argument("--exposure-us", type=float, default=None)
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--save", default="", help="把最后一帧存成 PNG 的目录")
    parser.add_argument("--stride", type=int, default=4, help="点云抽稀步长（统计点数用）")
    args = parser.parse_args()

    settings, warnings = load_settings(args.config) if args.config else (None, [])
    for warning in warnings:
        print(f"warning: {warning}")

    def pick(name: str, fallback):
        value = getattr(args, name)
        if value is not None:
            return value
        return getattr(settings, name) if settings is not None else fallback

    print("== 设备 ==")
    try:
        devices = list_devices()
    except RsError as exc:
        print(f"error: {exc}")
        return 2
    if not devices:
        print("没有找到 RealSense 设备")
        return 1
    for device in devices:
        print(f"  {device}")

    try:
        with Camera(
            serial=args.serial or (settings.serial if settings else "") or None,
            index=settings.index if settings else 0,
            color_size=parse_size(pick("color_size", "640x480"), (640, 480)),
            depth_size=parse_size(pick("depth_size", "640x480"), (640, 480)),
            fps=int(pick("fps", 30)),
        ) as camera:
            camera.configure(
                exposure_mode=pick("exposure_mode", "sdk"),
                exposure_us=float(pick("exposure_us", 0.0)),
                gain=float(pick("gain", 0.0)),
                filters=filter_spec(settings) if settings else {},
            )
            camera.start()
            for note in camera.notes:
                print(f"  相机: {note}")

            print("== 生效配置 ==")
            for key, value in camera.settings().items():
                print(f"  {key}: {value}")

            if camera._auto:
                settled, converged = camera.wait_auto_settled()
                print(f"  AE {'已收敛' if converged else '未收敛'}: {settled}")

            print(f"== 跑 {args.frames} 帧 ==")
            ratios: list[float] = []
            depths: list[float] = []
            cloud_sizes: list[int] = []
            timestamps: list[float] = []
            frame = None
            t0 = time.monotonic()
            for _ in range(args.frames):
                frame = camera.read()
                if frame is None:
                    continue
                ratios.append(frame.depth_valid_ratio)
                finite = frame.depth[np.isfinite(frame.depth)]
                if finite.size:
                    depths.append(float(np.median(finite)))
                timestamps.append(frame.timestamp_ms)
                positions, _ = pointcloud(
                    frame.depth, frame.bgr, frame.intrinsics, stride=args.stride
                )
                cloud_sizes.append(len(positions))
            elapsed = time.monotonic() - t0

            if frame is None:
                print("error: 一帧都没读到（检查 USB3 口/线，或试 initial_reset）")
                return 1

            span = (timestamps[-1] - timestamps[0]) / 1000.0 if len(timestamps) > 1 else 0.0
            print(f"  主机侧帧率: {len(ratios) / max(elapsed, 1e-6):.2f} fps")
            if span > 0:
                print(f"  设备时间戳侧帧率: {(len(timestamps) - 1) / span:.2f} fps")
            print(f"  有效深度: {np.mean(ratios) * 100:.1f}% 中位深度 {np.median(depths):.3f} m")
            print(f"  点云点数（stride={args.stride}）: {int(np.median(cloud_sizes))}")
            print(f"  曝光/增益: {frame.exposure_us:.0f} us / {frame.gain:.2f}")
            print(
                f"  内参: fx={frame.intrinsics.fx:.2f} fy={frame.intrinsics.fy:.2f} "
                f"cx={frame.intrinsics.cx:.2f} cy={frame.intrinsics.cy:.2f}"
            )

            if args.save:
                import cv2

                out = Path(args.save)
                out.mkdir(parents=True, exist_ok=True)
                color_path = out / "color.png"
                depth_path = out / "depth.png"
                cv2.imwrite(str(color_path), frame.bgr)
                # 深度：nan -> 0，再按可见范围归一化，好看清前后关系
                finite = frame.depth[np.isfinite(frame.depth)]
                low, high = np.percentile(finite, [5, 95]) if finite.size else (0.0, 1.0)
                scaled = np.nan_to_num(frame.depth, nan=0.0)
                scaled = np.clip((scaled - low) / max(high - low, 1e-6), 0, 1)
                cv2.imwrite(
                    str(depth_path),
                    cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
                )
                print(
                    f"  已保存 {color_path} 与 {depth_path}（深度 {low:.2f}-{high:.2f} m 归一化）"
                )
    except RsError as exc:
        print(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("interrupted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
