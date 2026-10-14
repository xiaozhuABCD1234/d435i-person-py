# SPDX-License-Identifier: AGPL-3.0-or-later
"""d435i-person: Intel RealSense D435i -> YOLO 姿态估计 -> 真深度 3D 骨架 -> rerun。

参数都放在 config.toml（见 `--show-config` 看生效值）；命令行只用来临时覆盖。

    uv run d435i-person --list              # 枚举 RealSense 设备
    uv run d435i-person                      # 按 config.toml 跑
    uv run d435i-person --show-config        # 打印合并后的生效参数
    uv run d435i-person --duration 30
    uv run d435i-person --no-rerun --max-frames 300
    uv run d435i-person --world --world-height 1.2 --world-pitch 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields

import numpy as np

from .camera import Camera, RsError, list_devices, parse_size
from .config import (
    EXPOSURE_MODES,
    Settings,
    apply_cli,
    default_config_path,
    filter_spec,
    load_settings,
    resolve_path,
)
from .fall import FallConfig
from .geometry import pointcloud
from .track import MetricsConfig, PersonTracker
from .world import CameraToWorld


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="d435i-person",
        description="Intel RealSense D435i -> YOLO 姿态估计 -> 真深度 3D 骨架 -> rerun。"
        "参数默认读 config.toml，命令行只做临时覆盖",
    )
    p.add_argument(
        "--config",
        default=str(default_config_path()),
        help="参数文件（默认项目根的 config.toml）",
    )
    p.add_argument("--show-config", action="store_true", help="打印生效参数后退出")
    p.add_argument("--list", action="store_true", help="枚举 RealSense 设备后退出")

    # 下面这些 default=None 表示"命令行没给"，值交给 config.toml / 兜底默认值
    p.add_argument("--serial", default=None, help="覆盖 [camera].serial（空 = 第一台）")
    p.add_argument("--index", type=int, default=None, help="覆盖 [camera].index")
    p.add_argument("--color-size", default=None, help="覆盖 [camera].color_size，如 640x480")
    p.add_argument("--depth-size", default=None, help="覆盖 [camera].depth_size，如 640x480")
    p.add_argument("--fps", type=int, default=None, help="覆盖 [camera].fps")
    p.add_argument(
        "--exposure-mode",
        default=None,
        choices=EXPOSURE_MODES,
        help="覆盖 [camera].exposure_mode：sdk（相机自动）/manual",
    )
    p.add_argument("--exposure-us", type=float, default=None, help="覆盖 [camera].exposure_us")
    p.add_argument("--gain", type=float, default=None, help="覆盖 [camera].gain")
    p.add_argument(
        "--brightness",
        type=float,
        default=None,
        help="覆盖 [camera].brightness（AE 目标，-1 = 不动）",
    )
    p.add_argument(
        "--laser-power", type=float, default=None, help="覆盖 [camera].laser_power（-1 = 不动）"
    )
    p.add_argument(
        "--emitter",
        dest="emitter_enabled",
        type=int,
        default=None,
        help="覆盖 [camera].emitter_enabled：1 开发射器 / 0 关（-1 = 不动）",
    )
    p.add_argument(
        "--depth-preset", type=int, default=None, help="覆盖 [camera].depth_preset（-1 = 不动）"
    )
    p.add_argument(
        "--read-timeout-ms", type=int, default=None, help="覆盖 [camera].read_timeout_ms"
    )
    p.add_argument(
        "--spatial",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 [filters].spatial（深度浮雕滤镜）",
    )
    p.add_argument(
        "--temporal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 [filters].temporal（深度时域滤镜）",
    )
    p.add_argument(
        "--hole-filling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 [filters].hole_filling（孔洞填充）",
    )

    p.add_argument("--weights", default=None, help="覆盖 [detector].weights")
    p.add_argument("--conf", type=float, default=None, help="覆盖 [detector].conf")
    p.add_argument("--imgsz", type=int, default=None, help="覆盖 [detector].imgsz")
    p.add_argument("--device", default=None, help="覆盖 [detector].device：''/cuda:0/cpu")
    p.add_argument("--kpt-conf", type=float, default=None, help="覆盖 [detector].kpt_conf")

    p.add_argument("--detect-hz", type=float, default=None, help="覆盖 [run].detect_hz")
    p.add_argument("--duration", type=float, default=None, help="覆盖 [run].duration")
    p.add_argument("--max-frames", type=int, default=None, help="覆盖 [run].max_frames")
    p.add_argument("--save-dir", default=None, help="覆盖 [run].save_dir（存标注图）")
    p.add_argument("--jsonl", default=None, help="覆盖 [output].jsonl（每帧人的 JSON Lines）")
    p.add_argument("--output-hz", type=float, default=None, help="覆盖 [output].hz")

    rerun_group = p.add_mutually_exclusive_group()
    rerun_group.add_argument(
        "--rerun",
        dest="rerun_enabled",
        action="store_true",
        default=None,
        help="开 rerun viewer",
    )
    rerun_group.add_argument(
        "--no-rerun",
        dest="rerun_enabled",
        action="store_false",
        default=None,
        help="不开 rerun，只打印（3D/点云/深度也不再记录）",
    )
    p.add_argument(
        "--rerun-save",
        default=None,
        help="覆盖 [rerun].save：写成 .rrd 且不弹窗口（默认不写文件）",
    )
    p.add_argument(
        "--rerun-memory-limit",
        default=None,
        help='覆盖 [rerun].memory_limit，如 "512MB"/"75%%"',
    )
    p.add_argument(
        "--rerun-keep-history",
        action="store_true",
        default=None,
        help="覆盖 [rerun].keep_history=true：保留时间轴（内存会一直涨）",
    )
    p.add_argument(
        "--rerun-jpeg-quality",
        type=int,
        default=None,
        help="覆盖 [rerun].jpeg_quality：1-100，0 = 送原图（更吃内存）",
    )
    p.add_argument(
        "--coords-view",
        dest="rerun_show_coords",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 [rerun].show_coords：右上角加一组 X/Y/Z 坐标曲线",
    )
    p.add_argument(
        "--count-view",
        dest="rerun_show_count",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 [rerun].show_count：右上角再放一条人数曲线（默认关）",
    )
    p.add_argument(
        "--metrics-view",
        dest="rerun_show_metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 [rerun].show_metrics：人体指标曲线（速度/身高/肩宽/步频…，默认开）",
    )
    p.add_argument(
        "--values-view",
        dest="rerun_show_values",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 [rerun].show_values：每人一行极简数值（默认开）",
    )
    p.add_argument(
        "--metrics-views",
        default=None,
        help="覆盖 [rerun].metrics_views：指标曲线分哪几个框，如 speed,size,gait"
        "（默认 all = 全部组；也可以直接写单个指标名，如 height）",
    )
    p.add_argument(
        "--metrics-layout",
        default=None,
        choices=["grid", "stack", "single"],
        help="覆盖 [rerun].metrics_layout：格子/一框一行/全塞回一个框",
    )
    p.add_argument(
        "--fall",
        dest="fall_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 [fall].enabled：开不开摔倒检测（默认开）",
    )
    p.add_argument(
        "--metrics",
        dest="metrics_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 [metrics].enabled：算不算人体指标（速度/步态/尺寸）",
    )

    pose3d_group = p.add_mutually_exclusive_group()
    pose3d_group.add_argument(
        "--pose3d",
        dest="pose3d_enabled",
        action="store_true",
        default=None,
        help="覆盖 [pose3d].enabled=true：记 3D 骨架/轨迹/点云（默认开）",
    )
    pose3d_group.add_argument(
        "--no-pose3d",
        dest="pose3d_enabled",
        action="store_false",
        default=None,
        help="覆盖 [pose3d].enabled=false：只记 2D（人还是带 id），省 viewer 内存",
    )
    p.add_argument(
        "--pose3d-kalman",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 [pose3d].kalman：3D 关键点卡尔曼滤波（默认开）",
    )
    p.add_argument(
        "--pose3d-depth-view",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 [pose3d].depth_view：左列再放一块对齐深度图",
    )
    p.add_argument("--iou-threshold", type=float, default=None, help="覆盖 [track].iou_threshold")
    p.add_argument(
        "--match-max-distance",
        type=float,
        default=None,
        help="覆盖 [track].match_max_distance（米，0 = 只看 IoU）",
    )
    p.add_argument("--trail-length", type=int, default=None, help="覆盖 [track].trail_length")
    p.add_argument("--trail-interval", type=float, default=None, help="覆盖 [track].trail_interval")

    points_group = p.add_mutually_exclusive_group()
    points_group.add_argument(
        "--points",
        dest="points_enabled",
        action="store_true",
        default=None,
        help="覆盖 [points].enabled=true：记彩色点云",
    )
    points_group.add_argument(
        "--no-points",
        dest="points_enabled",
        action="store_false",
        default=None,
        help="覆盖 [points].enabled=false：不记点云（viewer 内存降一个量级）",
    )
    p.add_argument("--points-hz", type=float, default=None, help="覆盖 [points].hz")
    p.add_argument("--points-stride", type=int, default=None, help="覆盖 [points].stride")
    p.add_argument(
        "--points-max-distance",
        type=float,
        default=None,
        help="覆盖 [points].max_distance（米）",
    )

    world_group = p.add_mutually_exclusive_group()
    world_group.add_argument(
        "--world",
        dest="world_enabled",
        action="store_true",
        default=None,
        help="覆盖 [world].enabled=true：启用世界系（地面系）坐标",
    )
    world_group.add_argument(
        "--no-world",
        dest="world_enabled",
        action="store_false",
        default=None,
        help="覆盖 [world].enabled=false：只用相机坐标系",
    )
    p.add_argument(
        "--world-height", type=float, default=None, help="覆盖 [world].height（光心离地高度，米）"
    )
    p.add_argument(
        "--world-pitch", type=float, default=None, help="覆盖 [world].pitch_deg（镜头下俯，度）"
    )
    p.add_argument("--world-yaw", type=float, default=None, help="覆盖 [world].yaw_deg")
    p.add_argument("--world-roll", type=float, default=None, help="覆盖 [world].roll_deg")
    return p


def print_settings(settings: Settings, config_path: str) -> None:
    print(f"# {config_path}")
    for spec in fields(settings):
        print(f"{spec.name} = {getattr(settings, spec.name)!r}")


def _person_record(person, world: CameraToWorld | None) -> dict:
    """一个人 -> 一行 JSON。世界坐标只在 [world] 开着时才有。"""
    position = person.position
    record = {
        "id": int(person.id),
        "conf": round(float(person.conf), 4),
        "distance": None if not np.isfinite(person.distance) else round(person.distance, 4),
        "cam": [round(float(v), 4) for v in position] if np.isfinite(position).all() else None,
        "world": None,
        "speed": round(float(person.speed_ground), 4),
        "state": person.state,
        "velocity": [round(float(v), 4) for v in person.velocity],
        # 摔倒状态：level 0 正常 / 1 疑似（有冲击、正在倒地）/ 2 摔倒（报警）
        "fall": {
            "level": int(getattr(person, "fall_level", 0)),
            "state": getattr(person, "fall_state", ""),
            "lying": bool(getattr(person, "lying", False)),
            "since": getattr(person, "fall_since", None),
        },
        # 人体尺寸/步态：身高/肩宽/髀宽/腿长/步频/步幅…缺项的键不会出现
        "metrics": {key: round(float(value), 4) for key, value in person.metrics.items()},
        "box": [round(float(v), 2) for v in person.box],
        "valid_keypoints": int(person.valid_count),
        "age": int(person.age),
    }
    if world is not None and np.isfinite(position).all():
        record["world"] = [round(float(v), 4) for v in world.apply(position)]
    return record


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings, config_warnings = load_settings(args.config)
    except (OSError, ValueError) as exc:  # TOMLDecodeError 也是 ValueError
        print(f"error: 读参数文件失败: {exc}", file=sys.stderr)
        return 2
    settings = apply_cli(settings, args)
    for warning in config_warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if args.show_config:
        print_settings(settings, args.config)
        return 0

    if settings.exposure_mode not in EXPOSURE_MODES:
        print(
            f"error: [camera].exposure_mode = {settings.exposure_mode!r} 不认识，"
            f"可选 {EXPOSURE_MODES}",
            file=sys.stderr,
        )
        return 2

    if args.list:
        try:
            devices = list_devices()
        except RsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if not devices:
            print("没有找到 RealSense 设备（确认插在 USB3 口、udev 规则已装）")
            return 1
        for device in devices:
            print(device)
        return 0

    try:
        color_size = parse_size(settings.color_size, (640, 480))
        depth_size = parse_size(settings.depth_size, (640, 480))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    weights_path = resolve_path(settings.weights)
    if not weights_path.is_file():
        print(
            f"error: 找不到权重 {weights_path}（[detector].weights = {settings.weights!r}）",
            file=sys.stderr,
        )
        return 2

    try:
        from .pose import PoseEstimator

        estimator = PoseEstimator(
            weights_path,
            device=settings.device or None,
            conf=settings.conf,
            imgsz=settings.imgsz,
            kpt_conf=settings.kpt_conf,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    world = (
        CameraToWorld(
            height=settings.world_height,
            pitch_deg=settings.world_pitch_deg,
            yaw_deg=settings.world_yaw_deg,
            roll_deg=settings.world_roll_deg,
        )
        if settings.world_enabled
        else None
    )

    view = None
    if settings.rerun_enabled:
        from .viz import RerunView

        rrd = settings.rerun_save or None
        view = RerunView(
            spawn=not rrd,
            save_path=rrd,
            kpt_conf=settings.kpt_conf,
            keep_history=settings.rerun_keep_history,
            memory_limit=settings.rerun_memory_limit,
            server_memory_limit=settings.rerun_server_memory_limit,
            jpeg_quality=settings.rerun_jpeg_quality,
            port=settings.rerun_port,
            show_depth=settings.pose3d_enabled and settings.pose3d_depth_view,
            show_pose3d=settings.pose3d_enabled,
            show_points=settings.points_enabled,
            eye=settings.pose3d_eye,
            look_at=settings.pose3d_look_at,
            eye_up=settings.pose3d_eye_up,
            world=world,
            show_world_view=world is not None and settings.world_view,
            world_pointcloud=settings.world_pointcloud,
            show_coords=settings.rerun_show_coords,
            show_count=settings.rerun_show_count,
            show_metrics=settings.rerun_show_metrics,
            show_values=settings.rerun_show_values,
            metrics_views=settings.rerun_metrics_views,
            metrics_layout=settings.rerun_metrics_layout,
        )
        if rrd:
            print(f"rerun 记录 -> {rrd}")
            print(f"  打开方式: rerun {rrd}   （或拖进 rerun viewer）")
        else:
            print(
                f"rerun: viewer 已启动（记忆历史 {settings.rerun_keep_history}，"
                f"内存上限 {settings.rerun_memory_limit}，图像 JPEG {settings.rerun_jpeg_quality}）"
            )
            print("  不要窗口就用 --no-rerun")

    tracker = PersonTracker(
        kalman=settings.pose3d_kalman,
        process_noise=settings.pose3d_kalman_process,
        measurement_noise=settings.pose3d_kalman_measurement,
        iou_threshold=settings.iou_threshold,
        match_max_distance=settings.match_max_distance,
        max_missed=settings.max_missed,
        max_missed_kpt=settings.max_missed_kpt,
        trail_length=settings.trail_length,
        trail_interval=settings.trail_interval,
        metrics=MetricsConfig(
            enabled=settings.metrics_enabled,
            # 开了 [world] 就用世界系的"上"——否则镜头一俯仰，身高和腿长就发虚
            up=tuple(world.up_in_camera()) if world is not None else (0.0, -1.0, 0.0),
            speed_smoothing=settings.speed_smoothing,
            speed_deadband=settings.speed_deadband,
            gait_window=settings.gait_window,
            gait_lift_k=settings.gait_lift_k,
            walk_speed=settings.walk_speed,
            head_offset=settings.head_offset,
            height_min=settings.height_min,
            height_max=settings.height_max,
            stand_cos=settings.stand_cos,
            fall=FallConfig(
                enabled=settings.fall_enabled,
                impact_speed=settings.fall_impact_speed,
                drop_height=settings.fall_drop_height,
                impact_window=settings.fall_impact_window,
                lying_angle=settings.fall_lying_angle,
                lying_height=settings.fall_lying_height,
                lying_time=settings.fall_lying_time,
                confirm_window=settings.fall_confirm_window,
                recover_angle=settings.fall_recover_angle,
                recover_time=settings.fall_recover_time,
            ),
        ),
    )

    save_dir = resolve_path(settings.save_dir) if settings.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = resolve_path(settings.jsonl) if settings.jsonl else None
    jsonl_handle = None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = jsonl_path.open("a", encoding="utf-8")
        print(f"JSON Lines -> {jsonl_path}（追加写；每行一帧里的人）")

    print(f"config: {args.config}")
    print(
        f"run   : 姿态检测 {settings.detect_hz:g} Hz（相机 {settings.fps} fps 上限）, "
        f"imgsz {settings.imgsz}, conf {settings.conf:g}, kpt_conf {settings.kpt_conf:g}, "
        f"曝光 {settings.exposure_mode}"
    )
    print(
        f"3D    : pose3d {settings.pose3d_enabled}, 卡尔曼 {settings.pose3d_kalman}"
        + (
            f"，点云 {settings.points_hz:g} Hz stride {settings.points_stride}"
            if settings.points_enabled
            else "，点云关"
        )
    )
    if world is not None:
        print(f"world : 开（{world.describe()}），原点在相机正下方的地面点")

    period = 1.0 / max(settings.detect_hz, 0.1)
    # 帧到达时刻正好压在周期边界上时纯靠抖动决定去留，会白丢约一半的帧
    # （实测 15/15 与 30/15 都会），所以留一点容差。倍数在 config.toml 里。
    min_gap = period * settings.rate_tolerance
    period_points = 1.0 / settings.points_hz if settings.points_hz > 0 else 0.0
    period_output = 1.0 / settings.output_hz if settings.output_hz > 0 else 0.0
    processed = skipped = 0
    last_count = -1
    t_start = t_last = t_last_points = t_last_output = t_last_save = time.monotonic()

    try:
        with Camera(
            serial=settings.serial or None,
            index=settings.index,
            color_size=color_size,
            depth_size=depth_size,
            fps=settings.fps,
        ) as cam:
            cam.configure(
                exposure_mode=settings.exposure_mode,
                exposure_us=settings.exposure_us,
                gain=settings.gain,
                brightness=settings.brightness,
                laser_power=settings.laser_power,
                emitter_enabled=settings.emitter_enabled,
                depth_preset=settings.depth_preset,
                filters=filter_spec(settings),
            )
            cam.start()
            for item in cam.applied:
                print(f"option: {item}")
            for note in cam.notes:
                print(f"相机: {note}")

            if settings.exposure_mode == "sdk":
                settled, converged = cam.wait_auto_settled()
                tail = "" if converged else "（还在变，未收敛）"
                if settled:
                    print(
                        f"曝光/增益: 相机自动 -> 曝光 {settled['exposure_us']:.0f} us，"
                        f"增益 {settled['gain']:.2f}{tail}"
                    )
                else:
                    print("曝光/增益: 相机自动")
            else:
                print(
                    f"曝光/增益: 固定 {settings.exposure_us:g} us / {settings.gain:g}"
                    "（0 = 用设备当前值）"
                )

            camera_settings = cam.settings()
            print("camera: " + ", ".join(f"{k}={v}" for k, v in camera_settings.items()))
            print(
                f"model : {estimator.weights} task=pose classes={estimator.names} "
                f"device={estimator.device}"
            )
            print(f"running: pose {settings.detect_hz:g} Hz, Ctrl-C 退出")

            while True:
                frame = cam.read(timeout_ms=settings.read_timeout_ms)
                if frame is None:
                    continue

                now = time.monotonic()
                # 还没到检测时刻：这帧只用来保持取流节奏，不做推理
                # （注意不能把这个判断放在 read 之前，否则会先把等到的那一帧丢掉）
                if now - t_last < min_gap:
                    skipped += 1
                    continue
                t_last = now

                # 不在这里 result.plot()：rerun 侧用结构化框/骨架重画，只有存盘时才
                # 需要标注图，到那时再 estimator.annotate(result) 按需生成。
                poses, result = estimator(frame.bgr)
                processed += 1

                # 真深度直接把 2D 关键点抬成 3D：跟踪、轨迹、世界坐标都在这一步出结果
                people = tracker.update(
                    poses,
                    frame.depth,
                    frame.intrinsics,
                    threshold=settings.kpt_conf,
                    t=frame.host_time,
                )
                live = [person for person in people if person.missed == 0]

                if view is not None:
                    # 送相机原图；框/关键点/骨架由 rerun 用结构化数据画
                    view.log_frame(frame, people)
                    if settings.pose3d_enabled:
                        if settings.pose3d_depth_view:
                            view.log_depth(frame.depth)
                        view.log_people3d(people)

                count = len(live)
                if count and count != last_count:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] person x{count} "
                        f"({', '.join(f'#{d.id} {d.conf:.2f} {d.distance:.2f}m' for d in live)})"
                    )
                last_count = count

                if (
                    view is not None
                    and settings.points_enabled
                    and now - t_last_points >= period_points
                ):
                    t_last_points = now
                    positions, colors = pointcloud(
                        frame.depth,
                        frame.bgr,
                        frame.intrinsics,
                        stride=settings.points_stride,
                        max_distance=settings.points_max_distance,
                        min_distance=settings.points_min_distance,
                    )
                    view.log_points(positions, colors)

                if jsonl_handle is not None and now - t_last_output >= period_output:
                    t_last_output = now
                    jsonl_handle.write(
                        json.dumps(
                            {
                                "t": round(frame.host_time, 4),
                                "frame": frame.frame_number,
                                "device_timestamp_ms": round(frame.timestamp_ms, 3),
                                "count": count,
                                "people": [_person_record(person, world) for person in live],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    jsonl_handle.flush()

                if save_dir and count and now - t_last_save >= settings.save_min_interval:
                    import cv2

                    t_last_save = now
                    path = save_dir / time.strftime("person_%Y%m%d_%H%M%S.jpg")
                    cv2.imwrite(str(path), estimator.annotate(result))
                    print(f"saved {path}")

                if processed % 50 == 0:
                    print(
                        f"... {processed} 帧, "
                        f"{processed / (now - t_start):.1f} fps 端到端, "
                        f"{skipped} 帧跳推理, 最近一帧 {frame.width}x{frame.height}, "
                        f"有效深度 {frame.depth_valid_ratio * 100:.0f}%, 人 {count}"
                    )

                if settings.max_frames and processed >= settings.max_frames:
                    break
                if settings.duration and now - t_start >= settings.duration:
                    break
    except KeyboardInterrupt:
        print("\ninterrupted")
    except RsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    elapsed = time.monotonic() - t_start
    print(f"done: {processed} 帧 / {elapsed:.1f}s ({processed / max(elapsed, 1e-6):.2f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
