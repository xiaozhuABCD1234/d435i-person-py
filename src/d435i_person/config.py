# SPDX-License-Identifier: AGPL-3.0-or-later
"""config.toml 的读取与命令行覆盖。

优先级：**命令行 > config.toml > 代码里的兜底默认值**。

`Settings` 里那些默认值只是"连配置文件都没有"时的兜底；正常参数都应当写在
config.toml 里。字段用 `[section].key` 的扁平映射描述，写错的键名会给出警告而不是
被静默忽略 —— 参数文件最怕的就是改了没生效还不知道。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/d435i_person/config.py -> 项目根
DEFAULT_CONFIG_NAME = "config.toml"
# sdk = 相机自己的自动曝光/增益；manual = 固定用 exposure_us / gain
EXPOSURE_MODES: tuple[str, ...] = ("sdk", "manual")


@dataclass
class Settings:
    """全部可调参数。"""

    # ---------------------------------------------------------------- [camera]
    serial: str = ""
    index: int = 0
    color_size: str = "640x480"
    depth_size: str = "640x480"
    fps: int = 30
    exposure_mode: str = "sdk"
    # manual 模式：0 = 用设备当前值（曝光/增益不支持"0 就是关掉"的语义）
    exposure_us: float = 0.0
    gain: float = 0.0
    # sdk 模式下 AE 的目标亮度（写彩色 sensor 的 Brightness）；-1 = 不动
    brightness: float = -1.0
    # 红外发射器 / 激光功率 / 深度视觉预设；-1 一律表示"不动这项"
    laser_power: float = -1.0
    emitter_enabled: int = -1
    depth_preset: int = -1
    read_timeout_ms: int = 5000

    # --------------------------------------------------------------- [filters]
    # 后处理滤镜（分辨率不变的那几个）。默认全关：D435i 原生深度对关键点已经够用，
    # 滤镜会模糊边缘，开之前先看 [pose3d] 的效果。
    spatial: bool = False
    temporal: bool = False
    hole_filling: bool = False

    # -------------------------------------------------------------- [detector]
    weights: str = "models/yolo26m-pose.pt"  # 兜底默认；正常参数写在 config.toml
    conf: float = 0.35
    imgsz: int = 640
    device: str = ""
    kpt_conf: float = 0.5

    # ------------------------------------------------------------------- [run]
    detect_hz: float = 15.0
    rate_tolerance: float = 0.9
    duration: float = 0.0
    max_frames: int = 0
    save_dir: str = ""
    save_min_interval: float = 1.0

    # ---------------------------------------------------------------- [output]
    # 非空则把每帧的人（id/距离/相机坐标/世界坐标）按 JSON Lines 追加写进去
    jsonl: str = ""
    output_hz: float = 5.0

    # ----------------------------------------------------------------- [rerun]
    rerun_enabled: bool = True
    rerun_save: str = ""
    rerun_keep_history: bool = False
    rerun_memory_limit: str = "512MB"
    rerun_server_memory_limit: str = "512MB"
    rerun_jpeg_quality: int = 75
    rerun_port: int = 9876
    # 右上角面板：默认显示"人体坐标"（坐标表 + 每个人的 X/Y/Z 曲线），
    # 人数曲线改成默认关（数据照样在 person/count 与 JSONL 里）。
    rerun_show_coords: bool = True
    rerun_show_count: bool = False

    # ---------------------------------------------------------------- [pose3d]
    pose3d_enabled: bool = True
    pose3d_kalman: bool = True
    pose3d_kalman_process: float = 10.0
    # 观测噪声（m）。D435i 的真深度比单目估计稳得多，可以比 hik 小一个量级。
    pose3d_kalman_measurement: float = 0.02
    pose3d_depth_view: bool = True
    # 3D 视图的固定视角（相机坐标系，米）；不固定的话 rerun 会按数据包围盒自动取景
    pose3d_eye: tuple[float, float, float] = (0.0, 0.0, -1.5)
    pose3d_look_at: tuple[float, float, float] = (0.0, 0.0, 2.5)
    pose3d_eye_up: tuple[float, float, float] = (0.0, -1.0, 0.0)

    # ----------------------------------------------------------------- [track]
    iou_threshold: float = 0.3
    # >0 时配对还要满足 3D 距离门限（米），防止两人交叉时换 id；0 = 只看 IoU
    match_max_distance: float = 0.0
    max_missed: int = 5
    max_missed_kpt: int = 3
    trail_length: int = 300
    trail_interval: float = 0.1

    # ---------------------------------------------------------------- [points]
    points_enabled: bool = True
    points_hz: float = 10.0
    points_stride: int = 4
    points_max_distance: float = 4.0
    points_min_distance: float = 0.05

    # ----------------------------------------------------------------- [world]
    world_enabled: bool = False
    world_height: float = 1.0
    world_pitch_deg: float = 0.0
    world_yaw_deg: float = 0.0
    world_roll_deg: float = 0.0
    world_pointcloud: bool = True
    world_view: bool = True


# (section, key) -> Settings 字段名
CONFIG_FIELDS: dict[tuple[str, str], str] = {
    ("camera", "serial"): "serial",
    ("camera", "index"): "index",
    ("camera", "color_size"): "color_size",
    ("camera", "depth_size"): "depth_size",
    ("camera", "fps"): "fps",
    ("camera", "exposure_mode"): "exposure_mode",
    ("camera", "exposure_us"): "exposure_us",
    ("camera", "gain"): "gain",
    ("camera", "brightness"): "brightness",
    ("camera", "laser_power"): "laser_power",
    ("camera", "emitter_enabled"): "emitter_enabled",
    ("camera", "depth_preset"): "depth_preset",
    ("camera", "read_timeout_ms"): "read_timeout_ms",
    ("filters", "spatial"): "spatial",
    ("filters", "temporal"): "temporal",
    ("filters", "hole_filling"): "hole_filling",
    ("detector", "weights"): "weights",
    ("detector", "conf"): "conf",
    ("detector", "imgsz"): "imgsz",
    ("detector", "device"): "device",
    ("detector", "kpt_conf"): "kpt_conf",
    ("run", "detect_hz"): "detect_hz",
    ("run", "rate_tolerance"): "rate_tolerance",
    ("run", "duration"): "duration",
    ("run", "max_frames"): "max_frames",
    ("run", "save_dir"): "save_dir",
    ("run", "save_min_interval"): "save_min_interval",
    ("output", "jsonl"): "jsonl",
    ("output", "hz"): "output_hz",
    ("rerun", "enabled"): "rerun_enabled",
    ("rerun", "save"): "rerun_save",
    ("rerun", "keep_history"): "rerun_keep_history",
    ("rerun", "memory_limit"): "rerun_memory_limit",
    ("rerun", "server_memory_limit"): "rerun_server_memory_limit",
    ("rerun", "jpeg_quality"): "rerun_jpeg_quality",
    ("rerun", "port"): "rerun_port",
    ("rerun", "show_coords"): "rerun_show_coords",
    ("rerun", "show_count"): "rerun_show_count",
    ("pose3d", "enabled"): "pose3d_enabled",
    ("pose3d", "kalman"): "pose3d_kalman",
    ("pose3d", "kalman_process"): "pose3d_kalman_process",
    ("pose3d", "kalman_measurement"): "pose3d_kalman_measurement",
    ("pose3d", "depth_view"): "pose3d_depth_view",
    ("pose3d", "eye"): "pose3d_eye",
    ("pose3d", "look_at"): "pose3d_look_at",
    ("pose3d", "eye_up"): "pose3d_eye_up",
    ("track", "iou_threshold"): "iou_threshold",
    ("track", "match_max_distance"): "match_max_distance",
    ("track", "max_missed"): "max_missed",
    ("track", "max_missed_kpt"): "max_missed_kpt",
    ("track", "trail_length"): "trail_length",
    ("track", "trail_interval"): "trail_interval",
    ("points", "enabled"): "points_enabled",
    ("points", "hz"): "points_hz",
    ("points", "stride"): "points_stride",
    ("points", "max_distance"): "points_max_distance",
    ("points", "min_distance"): "points_min_distance",
    ("world", "enabled"): "world_enabled",
    ("world", "height"): "world_height",
    ("world", "pitch_deg"): "world_pitch_deg",
    ("world", "yaw_deg"): "world_yaw_deg",
    ("world", "roll_deg"): "world_roll_deg",
    ("world", "pointcloud"): "world_pointcloud",
    ("world", "view"): "world_view",
}

# 命令行能覆盖的字段（argparse 里 default=None 表示"没给"；没有对应开关的字段
# 会因为 args 上没有这个属性而被跳过）
CLI_FIELDS: tuple[str, ...] = tuple(dict.fromkeys(CONFIG_FIELDS.values()))


def default_config_path() -> Path:
    return PROJECT_ROOT / DEFAULT_CONFIG_NAME


def _coerce(value: Any, target: type, attr: str) -> Any:
    """把 TOML 读出来的值转成 Settings 字段期望的类型，类型不对就报错。"""
    if target is bool:
        if not isinstance(value, bool):
            raise TypeError(f"需要 true/false，收到 {value!r}")
        return value
    if target is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"需要数字，收到 {value!r}")
        return float(value)
    if target is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"需要整数，收到 {value!r}")
        return value
    if target is tuple:
        if (
            not isinstance(value, list)
            or len(value) != 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
        ):
            raise TypeError(f"需要 3 个数字的数组，收到 {value!r}")
        return tuple(float(v) for v in value)
    return str(value)


def load_settings(path: Path | str | None = None) -> tuple[Settings, list[str]]:
    """读配置文件；返回 ``(设置, 警告列表)``。文件不存在就用兜底默认值。"""
    resolved = Path(path) if path else default_config_path()
    settings = Settings()
    warnings: list[str] = []

    if not resolved.is_file():
        warnings.append(f"没找到配置文件 {resolved}，用代码里的兜底默认值")
        return settings, warnings

    with resolved.open("rb") as handle:
        data = tomllib.load(handle)

    for (section, key), attr in CONFIG_FIELDS.items():
        table = data.get(section)
        if not isinstance(table, dict) or key not in table:
            continue
        target = type(getattr(settings, attr))
        try:
            setattr(settings, attr, _coerce(table[key], target, attr))
        except (TypeError, ValueError) as exc:
            warnings.append(f"[{section}].{key} 不合法（{exc}），已忽略")

    for section, table in data.items():
        if not isinstance(table, dict):
            warnings.append(f"{section!r} 不是表（table），已忽略")
            continue
        for key in table:
            if (section, key) not in CONFIG_FIELDS:
                warnings.append(f"未知配置项 [{section}].{key}，已忽略")

    return settings, warnings


def apply_cli(settings: Settings, args) -> Settings:
    """把命令行显式给出的值盖到配置上（argparse 的 ``default=None`` = 没给）。"""
    for attr in CLI_FIELDS:
        value = getattr(args, attr, None)
        if value is not None:
            setattr(settings, attr, list(value) if isinstance(value, list) else value)
    return settings


def resolve_path(value: str) -> Path:
    """相对路径按项目根解析，避免受当前工作目录影响。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def filter_spec(settings: Settings) -> dict[str, bool]:
    """给 ``Camera.configure(filters=...)`` 用的字典。"""
    return {
        "spatial": settings.spatial,
        "temporal": settings.temporal,
        "hole_filling": settings.hole_filling,
    }
