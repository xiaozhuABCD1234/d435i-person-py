# SPDX-License-Identifier: AGPL-3.0-or-later
"""D435i 取流：pyrealsense2 直连，深度对齐到彩色，输出一帧（彩色 BGR + 米制深度）。

为什么不用 ``rs.pointcloud`` / ``realsense-viewer``：本包只要"彩色 + 对齐深度 + 内参"，
自己拼 pipeline 最省事，而且对齐之后深度图就是彩色相机视角的一张针孔图像，
3D 相关的东西全都能在 numpy 里算（见 ``geometry``）。

几个实测/文档确认过的坑：

* 彩色流直接要 ``rs.format.bgr8``（D435i 支持），省掉一次 cv2 转换，本包内部统一 BGR。
* ``align(rs.stream.color)`` 之后深度图分辨率 = **彩色**流分辨率，内参要用彩色的
  （``Intrinsics.scaled_to`` 只是兜底，正常是恒等变换）。
* z16 的 1 不等于 1 米，得乘 ``depth_scale``（D435i 是 0.001）。这个值从设备读，
  不写死。
* ``wait_for_frames`` 超时抛 ``RuntimeError``，这里翻译成返回 None，交给上层 continue。
* 后处理滤镜只用**不改变分辨率**的那几个（spatial / temporal / hole_filling）；
  decimation 会改分辨率，内参与 3D 全得跟着重算，本包用 numpy 抽稀代替。
* 设置不支持的 option 会抛 ``RuntimeError``；``_set_option`` 把它变成一条 note，
  而不是让整个程序起不来（换机型/固件时这点很关键）。
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass

import numpy as np

from .geometry import DEFAULT_DEPTH_SCALE, Intrinsics, depth_to_meters


class RsError(RuntimeError):
    """RealSense SDK 相关的错误（没装包、没插相机、打不开设备）。"""


def _rs():
    """延迟导入 pyrealsense2：没装 SDK 时也能 import 本模块跑单测。"""
    try:
        import pyrealsense2 as rs
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise RsError(
            "没找到 pyrealsense2。装法：`uv sync`（Linux x86_64 + py3.12 有现成 manylinux wheel）"
        ) from exc
    return rs


def parse_size(text: str | None, default: tuple[int, int]) -> tuple[int, int]:
    """``"640x480"`` -> ``(640, 480)``；空/None -> default。"""
    if not text:
        return default
    cleaned = str(text).strip().lower().replace("*", "x").replace(",", "x")
    parts = [p for p in cleaned.split("x") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f'分辨率要写成 "宽x高"（如 640x480），收到 {text!r}')
    width, height = (int(p) for p in parts)
    if width <= 0 or height <= 0:
        raise ValueError(f"分辨率必须是正数，收到 {text!r}")
    return width, height


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    serial: str
    firmware: str
    usb: str
    product_line: str

    def __str__(self) -> str:
        return (
            f"[{self.index}] {self.name}  sn={self.serial}  fw={self.firmware}  "
            f"usb={self.usb}  {self.product_line}"
        )


def list_devices() -> list[DeviceInfo]:
    """枚举当前插着的 RealSense 设备（相当于 ``rs-enumerate-devices -s``）。"""
    rs = _rs()
    devices: list[DeviceInfo] = []
    for index, device in enumerate(rs.context().query_devices()):

        def info(key: str, device=device) -> str:
            try:
                return str(device.get_info(key))
            except Exception:  # 老固件可能没有某些字段
                return ""

        devices.append(
            DeviceInfo(
                index=index,
                name=info(rs.camera_info.name),
                serial=info(rs.camera_info.serial_number),
                firmware=info(rs.camera_info.firmware_version),
                usb=info(rs.camera_info.usb_type_descriptor),
                product_line=info(rs.camera_info.product_line),
            )
        )
    return devices


@dataclass
class Frame:
    """一帧：彩色 BGR + 已对齐到彩色的米制深度 + 内参。"""

    bgr: np.ndarray
    depth: np.ndarray  # (H, W) float32 米，无效是 nan
    frame_number: int
    timestamp_ms: float  # 设备硬件时钟（毫秒），未与主机同步
    host_time: float  # time.monotonic()
    intrinsics: Intrinsics
    exposure_us: float
    gain: float
    depth_valid_ratio: float  # 有效深度像素占比（抽稀统计）

    @property
    def width(self) -> int:
        return int(self.bgr.shape[1])

    @property
    def height(self) -> int:
        return int(self.bgr.shape[0])


class Camera:
    """一个 pipeline 同时开彩色 + 深度，返回对齐好的帧。

    用法固定是 **先 configure()、再 start()、最后 read()**：

        with Camera(serial="...") as cam:
            cam.configure(exposure_mode="sdk")
            cam.start()
            frame = cam.read()

    ``with`` 只负责退出时关流，不会自动开流（否则 configure 就没机会跑了）。
    """

    def __init__(
        self,
        *,
        serial: str | None = None,
        index: int = 0,
        color_size: tuple[int, int] = (640, 480),
        depth_size: tuple[int, int] = (640, 480),
        fps: int = 30,
    ):
        self.serial = serial or None
        self.index = int(index)
        self.color_size = tuple(color_size)
        self.depth_size = tuple(depth_size)
        self.fps = int(fps)
        self.notes: list[str] = []
        self.applied: list[str] = []
        self._pipe = None
        self._align = None
        self._depth_sensor = None
        self._color_sensor = None
        self._depth_scale = DEFAULT_DEPTH_SCALE
        self._intrinsics: Intrinsics | None = None
        self._filters: list = []
        self._filter_names: list[str] = []
        self._auto = False

    # ------------------------------------------------------------------ 配置
    def _set_option(self, sensor, option, value, label: str) -> bool:
        """写一个 option，不支持就记一条 note（而不是抛异常）。"""
        try:
            sensor.set_option(option, value)
        except RuntimeError as exc:
            self.notes.append(f"{label} 不受支持，跳过（{exc}）")
            return False
        with self._suppress():
            actual = sensor.get_option(option)
            self.applied.append(f"{label}={actual:g}")
        return True

    @staticmethod
    def _suppress():
        return contextlib.suppress(Exception)

    def configure(
        self,
        *,
        exposure_mode: str = "sdk",
        exposure_us: float = 0.0,
        gain: float = 0.0,
        brightness: float = -1.0,
        laser_power: float = -1.0,
        emitter_enabled: int = -1,
        depth_preset: int = -1,
        filters: dict | None = None,
    ) -> None:
        """设置曝光/增益/激光/后处理滤镜。只能在 ``start()`` 之前调。

        语义和 hik-person 一致：``exposure_mode="sdk"`` 交给相机自己的 AE；
        ``"manual"`` 用 ``exposure_us`` / ``gain``（0 = 用设备当前值）。
        ``-1`` 一律表示"不动这项"。
        """
        rs = _rs()
        if self._pipe is not None:
            raise RsError("pipeline 已经启动，configure() 必须在 start() 之前调用")

        self._auto = exposure_mode == "sdk"
        if exposure_mode not in ("sdk", "manual"):
            raise ValueError(f"exposure_mode 只能是 sdk 或 manual，收到 {exposure_mode!r}")

        ctx = rs.context()
        device = self._pick_device(ctx)
        self.notes.append(
            f"设备: {device.get_info(rs.camera_info.name)} "
            f"sn={device.get_info(rs.camera_info.serial_number)} "
            f"fw={device.get_info(rs.camera_info.firmware_version)} "
            f"usb={device.get_info(rs.camera_info.usb_type_descriptor)}"
        )
        if str(device.get_info(rs.camera_info.usb_type_descriptor)) != "3.2":
            self.notes.append(
                "USB 不是 3.2：带宽不足会让深度/彩色掉到低帧率或低分辨率，尽量插 USB3 口"
            )

        self._depth_sensor = device.first_depth_sensor()
        try:
            self._color_sensor = device.first_color_sensor()
        except Exception:  # pragma: no cover - 个别机型/固件没有独立彩色 sensor
            self._color_sensor = None
            self.notes.append("设备没有独立的彩色 sensor 句柄，曝光/增益走默认值")

        # 深度：发射器 / 激光功率 / 视觉预设（"-1 = 不动"）
        if emitter_enabled >= 0:
            self._set_option(
                self._depth_sensor,
                rs.option.emitter_enabled,
                int(emitter_enabled),
                "emitter_enabled",
            )
        if laser_power >= 0:
            self._set_option(self._depth_sensor, rs.option.laser_power, laser_power, "laser_power")
        if depth_preset >= 0:
            self._set_option(
                self._depth_sensor, rs.option.visual_preset, depth_preset, "depth visual_preset"
            )
            with self._suppress():
                name = self._depth_sensor.get_option_value_description(
                    rs.option.visual_preset, depth_preset
                )
                self.notes.append(f"深度预设 {depth_preset} = {name}")

        # 彩色：自动曝光（sdk）或固定曝光/增益（manual）
        if self._color_sensor is not None:
            self._set_option(
                self._color_sensor,
                rs.option.enable_auto_exposure,
                1 if self._auto else 0,
                "color enable_auto_exposure",
            )
            if not self._auto:
                if exposure_us and exposure_us > 0:
                    self._set_option(
                        self._color_sensor, rs.option.exposure, exposure_us, "color exposure"
                    )
                if gain and gain > 0:
                    self._set_option(self._color_sensor, rs.option.gain, gain, "color gain")
            elif brightness >= 0:
                # 彩色 sensor 的 Brightness 相当于 AE 的目标亮度（等价 hik 的 AE 目标灰度）
                self._set_option(
                    self._color_sensor, rs.option.brightness, brightness, "color brightness"
                )

        # 后处理滤镜（分辨率不变的那几个）
        spec = filters or {}
        self._filters = []
        self._filter_names = []
        for key, factory, label in (
            ("spatial", rs.spatial_filter, "spatial"),
            ("temporal", rs.temporal_filter, "temporal"),
            ("hole_filling", rs.hole_filling_filter, "hole_filling"),
        ):
            if spec.get(key):
                self._filters.append(factory())
                self._filter_names.append(label)
        if spec.get("decimation"):
            self.notes.append(
                "decimation 会改变深度分辨率（内参/3D 都得跟着重算），本包不启用；"
                "点云用 [points].stride 抽稀代替"
            )

    def _pick_device(self, ctx):
        """有序列号就按序列号挑，否则按 index 在物理设备里数。"""
        rs = _rs()
        devices = list(ctx.query_devices())
        if not devices:
            raise RsError(
                "没找到 RealSense 设备：确认相机插在 USB3 口、udev 规则已装"
                "（sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/ && 拔插相机）"
            )
        if self.serial:
            for device in devices:
                with self._suppress():
                    if str(device.get_info(rs.camera_info.serial_number)) == self.serial:
                        return device
            raise RsError(f"没有序列号为 {self.serial!r} 的设备；用 `d435i-person --list` 看有哪些")
        if not 0 <= self.index < len(devices):
            raise RsError(f"index={self.index} 越界，当前只有 {len(devices)} 台设备")
        return devices[self.index]

    # ------------------------------------------------------------------ 启停
    def start(self) -> None:
        """建 pipeline、开流、读内参与 depth scale。"""
        rs = _rs()
        if self._pipe is not None:
            return
        ctx = rs.context()
        device = self._pick_device(ctx)
        cfg = rs.config()
        cfg.enable_device(str(device.get_info(rs.camera_info.serial_number)))

        width, height = self.color_size
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, self.fps)
        dwidth, dheight = self.depth_size
        cfg.enable_stream(rs.stream.depth, dwidth, dheight, rs.format.z16, self.fps)

        self._pipe = rs.pipeline(ctx)
        profile = self._pipe.start(cfg)
        self._align = rs.align(rs.stream.color)

        self._depth_sensor = profile.get_device().first_depth_sensor()
        with self._suppress():
            self._depth_scale = float(self._depth_sensor.get_depth_scale())
        with contextlib.suppress(Exception):
            self._color_sensor = profile.get_device().first_color_sensor()

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self._intrinsics = Intrinsics(
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
            width=int(intr.width),
            height=int(intr.height),
        )

    def close(self) -> None:
        if self._pipe is not None:
            with self._suppress():
                self._pipe.stop()
            self._pipe = None
            self._align = None

    def __enter__(self) -> Camera:
        """只返回自己，**不**自动开流：``configure()`` 必须在 ``start()`` 之前调。"""
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ 读帧
    def read(self, timeout_ms: int = 5000) -> Frame | None:
        """等到一帧并对齐；超时/缺流返回 None（上层 continue 即可）。"""
        rs = _rs()
        if self._pipe is None:
            raise RsError("pipeline 还没 start()")
        try:
            frames = self._pipe.wait_for_frames(timeout_ms)
        except RuntimeError:
            return None

        aligned = self._align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        for flt in self._filters:
            depth_frame = flt.process(depth_frame)

        raw = np.asanyarray(depth_frame.get_data())
        depth = depth_to_meters(raw, self._depth_scale)
        bgr = np.asanyarray(color_frame.get_data())
        if not bgr.flags.writeable:  # SDK 缓冲是只读视图，拷一份免得后面意外踩到
            bgr = bgr.copy()

        height, width = depth.shape[:2]
        intrinsics = self._intrinsics or Intrinsics.from_fov(width, height, 60.0)
        intrinsics = intrinsics.scaled_to(width, height)

        # 有效深度占比按 4 倍抽稀统计，别为了一个日志数字遍历 30 万像素
        sample = depth[::4, ::4]
        valid_ratio = float(np.count_nonzero(np.isfinite(sample))) / max(sample.size, 1)

        return Frame(
            bgr=bgr,
            depth=depth,
            frame_number=int(color_frame.get_frame_number()),
            timestamp_ms=float(color_frame.get_timestamp()),
            host_time=time.monotonic(),
            intrinsics=intrinsics,
            exposure_us=self._option(rs.option.exposure),
            gain=self._option(rs.option.gain),
            depth_valid_ratio=valid_ratio,
        )

    def _option(self, option, sensor=None) -> float:
        target = self._color_sensor if sensor is None else sensor
        if target is None:
            return float("nan")
        try:
            return float(target.get_option(option))
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------ 状态
    def settings(self) -> dict:
        """当前生效的流配置/深度单位（打印用）。"""
        rs = _rs()
        out: dict[str, object] = {"depth_scale": f"{self._depth_scale:g} m/unit"}
        if self._pipe is None:
            return out
        with self._suppress():
            profile = self._pipe.get_active_profile()
            for name, stream in (("color", rs.stream.color), ("depth", rs.stream.depth)):
                video = profile.get_stream(stream).as_video_stream_profile()
                out[name] = (
                    f"{video.width()}x{video.height()}x{video.fps()} "
                    f"{str(video.format()).split('.')[-1]}"
                )
        if self._intrinsics is not None:
            h, v = self._intrinsics.fov_deg()
            out["intrinsics"] = (
                f"fx={self._intrinsics.fx:.1f} fy={self._intrinsics.fy:.1f} "
                f"cx={self._intrinsics.cx:.1f} cy={self._intrinsics.cy:.1f}"
            )
            out["fov"] = f"{h:.1f}°x{v:.1f}°"
        if self._filter_names:
            out["filters"] = "+".join(self._filter_names)
        return out

    def wait_auto_settled(
        self, *, timeout_s: float = 3.0, tolerance: float = 0.02, needed: int = 5
    ) -> tuple[dict[str, float], bool]:
        """等相机自己的 AE 收敛（连续 ``needed`` 帧相对变化都在 ``tolerance`` 内）。

        返回 ``({"exposure_us": ..., "gain": ...}, 是否收敛)``。manual 模式下直接返回
        当前读值，不去改任何东西。
        """
        last: dict[str, float] = {}
        stable = 0
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = self.read(timeout_ms=2000)
            if frame is None:
                continue
            current = {"exposure_us": frame.exposure_us, "gain": frame.gain}
            if last:
                change = max(abs(current[k] - last[k]) / max(abs(last[k]), 1e-6) for k in current)
                stable = stable + 1 if change <= tolerance else 0
                if stable >= needed:
                    return current, True
            last = current
        return last, False
