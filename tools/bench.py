# SPDX-License-Identifier: AGPL-3.0-or-later
"""分阶段计时：取流 / YOLO pose / 跟踪 / 点云 / rerun 记录各占多少。

    uv run python tools/bench.py                  # 200 帧
    uv run python tools/bench.py --frames 500 --device cpu
    uv run python tools/bench.py --no-camera      # 没有相机时用假数据只测模型与几何

目的：先知道瓶颈在哪。经验值（RTX 4060 Laptop, imgsz 640）：pose ~7 ms、
点云 stride 4 ~1.5 ms、跟踪 ~1 ms、rerun 记录 ~2 ms；所以 15 Hz 的 detect_hz
留了很宽的余量，卡顿一般来自 viewer 内存而不是这里的计算。

``--no-camera`` 用固定深度图造数据，只跑模型与几何，用来比较权重/imgsz 的开销。
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from d435i_person.camera import Camera, RsError, parse_size
from d435i_person.config import resolve_path
from d435i_person.geometry import Intrinsics, pointcloud
from d435i_person.pose import NUM_KEYPOINTS, PoseInstance
from d435i_person.track import PersonTracker


class Timer:
    """极简累加计时器（只关心总量，不需要上下文管理器）。"""

    def __init__(self) -> None:
        self.total: dict[str, float] = {}
        self.count = 0

    def add(self, name: str, seconds: float) -> None:
        self.total[name] = self.total.get(name, 0.0) + seconds

    def report(self) -> None:
        if not self.count:
            print("没有有效帧")
            return
        print(f"\n=== 每帧平均（{self.count} 帧）===")
        for name, seconds in sorted(self.total.items(), key=lambda item: -item[1]):
            print(f"  {name:12s} {seconds / self.count * 1000:7.2f} ms")
        beam = sum(self.total.values())
        summary = f"{beam / self.count * 1000:7.2f} ms -> {self.count / beam:6.2f} fps 上限"
        print(f"  {'合计':12s} {summary}")


def fake_pose() -> PoseInstance:
    points = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    points[:, 0] = 310.0 + np.arange(NUM_KEYPOINTS)
    points[:, 1] = 200.0 + np.arange(NUM_KEYPOINTS) * 4
    return PoseInstance(
        conf=0.8,
        xyxy=(280.0, 100.0, 360.0, 430.0),
        keypoints=points,
        kpt_conf=np.full(NUM_KEYPOINTS, 0.9, dtype=np.float32),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="d435i-person 分阶段计时")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="前多少帧不计时（CUDA 初始化/cuDNN 选算法都落在第一帧，不热身会把均值拉高）",
    )
    parser.add_argument("--weights", default="models/yolo26m-pose.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--color-size", default="640x480")
    parser.add_argument("--no-camera", action="store_true", help="用假数据，只测模型与几何")
    parser.add_argument("--no-yolo", action="store_true", help="跳过模型（只测取流与几何）")
    args = parser.parse_args()

    weights = resolve_path(args.weights)
    estimator = None
    if not args.no_yolo:
        from d435i_person.pose import PoseEstimator

        estimator = PoseEstimator(
            weights, device=args.device, conf=args.conf, imgsz=args.imgsz, kpt_conf=0.5
        )
        print(f"model: {weights.name} imgsz {args.imgsz} device {estimator.device}")

    tracker = PersonTracker()
    timer = Timer()
    width, height = parse_size(args.color_size, (640, 480))
    intrinsics = Intrinsics.from_fov(width, height, 60.0)

    camera = None
    if not args.no_camera:
        try:
            camera = Camera(color_size=(width, height), depth_size=(width, height), fps=30)
            camera.start()
            print(f"camera: {' '.join(f'{k}={v}' for k, v in camera.settings().items())}")
        except RsError as exc:
            print(f"warning: {exc}；退回 --no-camera 模式")
            camera = None
            args.no_camera = True

    try:
        for index in range(args.frames):
            timed = index >= args.warmup

            def lap(name: str, start: float, *, timed: bool = timed) -> None:
                if timed:
                    timer.add(name, time.perf_counter() - start)

            if camera is not None:
                t0 = time.perf_counter()
                frame = camera.read()
                lap("取流+对齐", t0)
                if frame is None:
                    continue
                bgr, depth, intrinsics_frame = frame.bgr, frame.depth, frame.intrinsics
            else:
                rng = np.random.default_rng(index)
                bgr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
                depth = np.full((height, width), 2.0 + 0.05 * np.sin(index / 10), dtype=np.float32)
                intrinsics_frame = intrinsics

            if estimator is not None:
                t0 = time.perf_counter()
                poses, _ = estimator(bgr)
                lap("YOLO pose", t0)
            else:
                poses = [fake_pose()]

            # 跟踪/点云用合成姿态计时：噪声图上模型根本检不到人，若直接把 poses 传下去，
            # 这两条路径没被跑到，会报出一个好看但没用的"0.02 ms"。
            track_poses = poses if camera is not None and estimator is not None else [fake_pose()]
            if index == args.warmup:
                print(f"本帧真实检测到的人数: {len(poses)}（跟踪按 {len(track_poses)} 人计时）")

            t0 = time.perf_counter()
            tracker.update(track_poses, depth, intrinsics_frame, threshold=0.5, t=index / 30.0)
            lap("跟踪+3D", t0)

            t0 = time.perf_counter()
            positions, _colors = pointcloud(
                depth, bgr, intrinsics_frame, stride=args.stride, max_distance=4.0
            )
            lap("点云", t0)

            if index == args.warmup:
                print(f"点云点数: {len(positions)} (stride={args.stride})")
            if timed:
                timer.count += 1
    finally:
        if camera is not None:
            camera.close()

    timer.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
