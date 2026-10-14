# SPDX-License-Identifier: AGPL-3.0-or-later
"""3D 关键点的卡尔曼滤波 + IoU 关联跟踪（带持久 ID 与轨迹）。

真深度虽然比单目估计稳得多，但逐帧跳变（尤其关节处的深度边缘）还是能看出来，
所以给每个人的每个关键点坐标（17×3 个标量）各配一个**恒定速度**卡尔曼滤波：

    状态 x = [位置, 速度]，F = [[1, dt], [0, 1]]，H = [1, 0]

跨帧得把"第 i 个人"对上号：前后帧用检测框 IoU 贪心配对（可选再加一个 3D 距离门限，
防止两人交叉时换脸）。配上的继续更新滤波并**沿用同一个 id**，配不上的新建轨迹，
连续多帧配不上的轨迹就删掉。id 单调递增、永不复用，所以日志里能直接当人的编号。

位置（每人一个 3D 点，用于轨迹和世界坐标）单独再配一组 3 维卡尔曼，让轨迹不至于
一格一格地跳。没有外部依赖（不用 scipy 的匈牙利算法）。纯 numpy，可单测。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .fall import FallConfig, FallDetector, torso_angle
from .geometry import Intrinsics, depth_valid_mask, lift_keypoints
from .pose import KEYPOINT_NAMES_17, PoseInstance
from .pose3d import (
    HEAD_OFFSET,
    HEIGHT_RANGE,
    STAND_COS,
    body_metrics,
    joint_of,
    person_position,
    to_display,
)

NUM_KP = len(KEYPOINT_NAMES_17)
# dt 的夹取范围（秒）：太小会让 F 退化，太大会让一帧预测飞出画面。
MIN_DT, MAX_DT = 1e-3, 0.5
# 相机系里的"上"（Y 向下，所以上 = -Y）；开了 [world] 就换成世界 Z 在相机系里的方向。
DEFAULT_UP: tuple[float, float, float] = (0.0, -1.0, 0.0)


@dataclass(frozen=True)
class MetricsConfig:
    """人体指标（速度/步态/尺寸）的参数。全部来自 config.toml 的 [metrics]。"""

    enabled: bool = True
    # "上"方向（相机系）。[world] 开着时由 CameraToWorld.up_in_camera() 给
    up: tuple[float, float, float] = DEFAULT_UP
    # 速度：卡尔曼速度状态直接读会很跳（15 Hz 下量测噪声导数化约 ±0.4 m/s），
    # 所以再走一道指数平滑；tau 越小越跟手、越大越平滑。
    speed_smoothing: float = 0.3
    # 低于它就当静止（显示与分类用，不影响曲线原始值）。
    speed_deadband: float = 0.15
    # 步态：抬脚判据 = 踝的竖直位移超过 μ + k·σ（窗口内的均值/标准差）；
    # 只有地面速度超过 walk_speed 时才计步（防止站着挪重心被算成走路）。
    gait_window: float = 4.0
    gait_lift_k: float = 0.8
    walk_speed: float = 0.25
    # 身高：鼻 -> 头顶补偿、合理区间、直立判据（见 pose3d.body_metrics）。
    head_offset: float = HEAD_OFFSET
    height_min: float = HEIGHT_RANGE[0]
    height_max: float = HEIGHT_RANGE[1]
    stand_cos: float = STAND_COS
    # 摔倒检测：独立的时序状态机（见 fall.py），[fall] 段配置。它在 [metrics].enabled
    # 关掉时也照跑——安全类功能不该因为“不看尺寸”就没了。
    fall: FallConfig = field(default_factory=FallConfig)

    def up_vector(self) -> np.ndarray:
        return np.asarray(self.up, dtype=np.float64)


class _Gait:
    """单个人的步态：每只脚一个"抬脚"状态机，阈值由滑动窗口的 μ/σ 自适应。

    信号是``(踝 - 髀中点) · 上方向``，也就是腿的伸展量：走路时它周期性地起伏。
    抬脚 = 信号越过 μ + k·σ，落地 = 回落到 μ 以下。每次"抬脚→落地"算一步，
    步频就用窗口内完成的步数除以实际跨度。比单目方案那套"猜步频"实在：
    这里测的是真深度的竖直位移。
    """

    def __init__(self, window_s: float, lift_k: float, max_steps: int = 64):
        self.window_s = float(window_s)
        self.lift_k = float(lift_k)
        self.samples: deque = deque()  # (t, (s_left, s_right))
        self.lifted = [False, False]
        self.steps: deque = deque(maxlen=max_steps)

    def update(self, t: float, signals: tuple[float, float]) -> None:
        self.samples.append((t, signals))
        while self.samples and t - self.samples[0][0] > self.window_s:
            self.samples.popleft()
        for leg in (0, 1):
            values = [s[1][leg] for s in self.samples if np.isfinite(s[1][leg])]
            current = signals[leg]
            if len(values) < 5 or not np.isfinite(current):
                continue
            mean = float(np.mean(values))
            high = mean + self.lift_k * float(np.std(values))
            low = mean + 0.2 * self.lift_k * float(np.std(values))
            if not self.lifted[leg] and current > high:
                self.lifted[leg] = True
            elif self.lifted[leg] and current < low:
                self.lifted[leg] = False
                self.steps.append(t)
        while self.steps and t - self.steps[0] > self.window_s:
            self.steps.popleft()

    def cadence(self) -> float:
        """步/分钟；步数不够或跨度太短就给 nan（不让 UI 显示一个乱数）。"""
        if len(self.steps) < 3:
            return float("nan")
        span = self.steps[-1] - self.steps[0]
        if span < 0.5:
            return float("nan")
        return float(len(self.steps) - 1) * 60.0 / span


def iou(a, b) -> float:
    """两个 ``xyxy`` 框的交并比。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    if inter <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


class ScalarKalman:
    """一维恒定速度卡尔曼：状态 ``[位置, 速度]``。"""

    def __init__(self, process_noise: float, measurement_noise: float):
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.x = np.zeros(2)
        self.p = np.eye(2)

    def init(self, z: float) -> None:
        """用第一个观测初始化：位置=观测，速度=0（速度方差先给大一点）。"""
        self.x = np.array([z, 0.0])
        self.p = np.diag([self.r, self.q * 100.0])

    def predict(self, dt: float) -> None:
        f = np.array([[1.0, dt], [0.0, 1.0]])
        q = self.q * np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]])
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q

    def update(self, z: float) -> None:
        h = np.array([1.0, 0.0])
        s = float(h @ self.p @ h) + self.r
        k = (self.p @ h) / s
        self.x = self.x + k * (z - float(h @ self.x))
        self.p = (np.eye(2) - np.outer(k, h)) @ self.p

    @property
    def position(self) -> float:
        return float(self.x[0])


@dataclass
class TrackedPerson:
    """跟踪结果：稳定的 3D 骨架 + 持久 id + 轨迹 + 人体指标（相机坐标系，米）。"""

    id: int
    conf: float
    box: tuple[float, float, float, float]
    keypoints2d: np.ndarray  # (17, 2) 本帧的原始 2D 关键点（像素）
    kpt_conf: np.ndarray  # (17,) 本帧的关键点置信度；漏检帧全置 0 不再画
    joints: np.ndarray  # (M, 3) 相机坐标
    names: list[str]
    strips: list[np.ndarray]
    position: np.ndarray  # (3,) 相机坐标，滤波后的位置；测不到是 nan
    velocity: np.ndarray  # (3,) 相机系速度（m/s，已平滑）
    speed: float  # |velocity|（m/s）
    speed_ground: float  # 垂直于"上"方向的速度分量（m/s）；"人走多快"看它
    metrics: dict[str, float]  # 人体尺寸 + 步态 + 速度，只含本帧有数的项
    trail: np.ndarray  # (T, 3) 相机坐标轨迹（旧的在前）
    age: int  # 连续/累计命中次数
    missed: int  # 最近连续漏检帧数（0 = 本帧命中了）
    valid_count: int  # 本帧有效关键点个数
    state_speed: float = 0.15  # 状态标签阀值：低于它显示"静止"
    fall_level: int = 0  # 0 正常 / 1 疑似摔倒 / 2 摔倒（报警）
    fall_state: str = ""  # 摔倒状态的中文短标签（正常时为空串）
    fall_since: float | None = None  # 当前状态的起始时刻（报警时刻看它）
    lying: bool = False  # 本帧是躺倒姿态（配 fall_level=1 就是在“倒地”过程中）

    @property
    def distance(self) -> float:
        """到相机光心的距离（米）。"""
        if not np.isfinite(self.position).all():
            return float("nan")
        return float(np.linalg.norm(self.position))

    @property
    def state(self) -> str:
        """极简运动状态（给短标签用）：静止 / 行走 / ?。"""
        if not np.isfinite(self.speed_ground):
            return "?"
        return "行走" if self.speed_ground >= self.state_speed else "静止"

    @property
    def label(self) -> str:
        return f"#{self.id} {self.conf:.2f}"


@dataclass
class _Track:
    id: int
    box: tuple[float, float, float, float]
    filters: list[list[ScalarKalman]] = field(default_factory=list)
    pos_filters: list[ScalarKalman] = field(default_factory=list)
    # 本帧（或最近一帧）观测到的原始位置，只用于配对时的 3D 距离门限
    position: np.ndarray = field(default_factory=lambda: np.full(3, np.nan, np.float32))
    seen: np.ndarray = field(default_factory=lambda: np.zeros(NUM_KP, bool))
    missed_kpt: np.ndarray = field(default_factory=lambda: np.zeros(NUM_KP, int))
    conf: float = 0.0
    keypoints2d: np.ndarray = field(default_factory=lambda: np.zeros((NUM_KP, 2), np.float32))
    kpt_conf: np.ndarray = field(default_factory=lambda: np.zeros(NUM_KP, np.float32))
    # 位置滤波是否已经被观测初始化过：没初始化过就不能报告 (0,0,0)（那是原点，不是"不知道"）
    pos_initialized: bool = False
    # 速度（m/s，已指数平滑）与步态状态机；没滤波时用后向差分算速度
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    speed: float = 0.0
    speed_ground: float = 0.0
    prev_position: np.ndarray = field(default_factory=lambda: np.full(3, np.nan, np.float32))
    prev_t: float = 0.0
    gait: _Gait = field(default_factory=lambda: _Gait(4.0, 0.8))
    fall: FallDetector = field(default_factory=FallDetector)
    missed: int = 0
    age: int = 0
    last_t: float = 0.0
    last_trail_t: float = -1e9
    trail: deque = field(default_factory=deque)
    raw: np.ndarray | None = None  # 最近一次观测（kalman 关掉时直接用它）


class PersonTracker:
    """把 pose 的 2D 关键点 + 真深度变成"带 id 的 3D 人"。"""

    def __init__(
        self,
        *,
        kalman: bool = True,
        process_noise: float = 10.0,
        measurement_noise: float = 0.02,
        iou_threshold: float = 0.3,
        match_max_distance: float = 0.0,
        max_missed: int = 5,
        max_missed_kpt: int = 3,
        trail_length: int = 300,
        trail_interval: float = 0.1,
        metrics: MetricsConfig | None = None,
    ):
        self.kalman = bool(kalman)
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.iou_threshold = float(iou_threshold)
        # >0 时，配对还要满足 3D 距离不超过这个值（米）；0 = 只看 IoU
        self.match_max_distance = float(match_max_distance)
        self.max_missed = int(max_missed)
        self.max_missed_kpt = int(max_missed_kpt)
        self.trail_length = max(0, int(trail_length))
        self.trail_interval = max(0.0, float(trail_interval))
        self.metrics = metrics or MetricsConfig()
        self.up = self.metrics.up_vector()
        self.tracks: list[_Track] = []
        self._next_id = 1

    # ------------------------------------------------------------------ 关联
    def _match(self, boxes, positions) -> dict[int, int]:
        """贪心 IoU 配对，返回 ``{检测下标: 轨迹下标}``（互斥）。"""
        pairs = []
        for ti, track in enumerate(self.tracks):
            for bi, box in enumerate(boxes):
                score = iou(track.box, box)
                if score < self.iou_threshold:
                    continue
                if self.match_max_distance > 0:
                    here, there = track.position, positions[bi]
                    too_far = (
                        np.isfinite(here).all()
                        and np.isfinite(there).all()
                        and float(np.linalg.norm(here - there)) > self.match_max_distance
                    )
                    if too_far:
                        continue
                pairs.append((score, ti, bi))
        pairs.sort(key=lambda item: item[0], reverse=True)

        used_tracks: set[int] = set()
        matched: dict[int, int] = {}
        for _score, ti, bi in pairs:
            if ti in used_tracks or bi in matched:
                continue
            used_tracks.add(ti)
            matched[bi] = ti
        return matched

    # ------------------------------------------------------------- 单条轨迹
    def _new_track(self, pose: PoseInstance, lifted, ok, position, t: float) -> _Track:
        filters = [[ScalarKalman(self.q, self.r) for _ in range(3)] for _ in range(NUM_KP)]
        for i in range(NUM_KP):
            if ok[i]:
                for axis in range(3):
                    filters[i][axis].init(float(lifted[i, axis]))
        pos_filters = [ScalarKalman(self.q, self.r) for _ in range(3)]
        initialized = bool(np.isfinite(position).all())
        if initialized:
            for axis in range(3):
                pos_filters[axis].init(float(position[axis]))
        track = _Track(
            id=self._next_id,
            box=tuple(pose.xyxy),
            filters=filters,
            pos_filters=pos_filters,
            seen=ok.copy(),
            missed_kpt=np.where(ok, 0, 1),
            conf=float(pose.conf),
            keypoints2d=np.asarray(pose.keypoints, dtype=np.float32).copy(),
            kpt_conf=np.asarray(pose.kpt_conf, dtype=np.float32).copy(),
            position=np.asarray(position, dtype=np.float32).copy(),
            pos_initialized=initialized,
            prev_position=np.asarray(position, dtype=np.float32).copy(),
            prev_t=t,
            gait=_Gait(self.metrics.gait_window, self.metrics.gait_lift_k),
            fall=FallDetector(self.metrics.fall),
            missed=0,
            age=1,
            last_t=t,
            trail=deque(maxlen=max(self.trail_length, 1)),
            raw=lifted.copy(),
        )
        self._next_id += 1
        self._append_trail(track, position, t)
        return track

    def _advance(self, track: _Track, pose: PoseInstance, lifted, ok, position, t: float) -> None:
        dt = min(max(t - track.last_t, MIN_DT), MAX_DT)
        for i in range(NUM_KP):
            for axis in range(3):
                filt = track.filters[i][axis]
                filt.predict(dt)
                if ok[i]:
                    filt.update(float(lifted[i, axis]))

        # 位置滤波：第一次拿到有效位置就整组 init（三个轴一起，否则会出现"x 轴已 init、
        # y/z 轴还在默认状态"的混搭，滤波值会莫名偏小）；之后只 predict/update。
        if not track.pos_initialized and np.isfinite(position).all():
            for axis in range(3):
                track.pos_filters[axis].init(float(position[axis]))
            track.pos_initialized = True
        else:
            for axis in range(3):
                filt = track.pos_filters[axis]
                filt.predict(dt)
                if np.isfinite(position[axis]):
                    filt.update(float(position[axis]))
        if np.isfinite(position).all():
            # 配对用的位置：拿到就用新的，没拿到就保留上一次已知的（短暂丢深度不该换 id）
            track.position = np.asarray(position, dtype=np.float32).copy()
        self._update_speed(track, position, dt, t)

        track.seen |= ok
        track.missed_kpt = np.where(ok, 0, track.missed_kpt + 1)
        track.box = tuple(pose.xyxy)
        track.conf = float(pose.conf)
        track.keypoints2d = np.asarray(pose.keypoints, dtype=np.float32).copy()
        track.kpt_conf = np.asarray(pose.kpt_conf, dtype=np.float32).copy()
        track.missed = 0
        track.age += 1
        track.last_t = t
        track.raw = lifted.copy()
        self._append_trail(track, position, t)

    def _freeze(self, track: _Track, t: float) -> None:
        """没配上的轨迹：只预测，不更新，计漏检。"""
        dt = min(max(t - track.last_t, MIN_DT), MAX_DT)
        for i in range(NUM_KP):
            for axis in range(3):
                track.filters[i][axis].predict(dt)
        for axis in range(3):
            track.pos_filters[axis].predict(dt)
        # 没新观测就不重算速度，但让 EMA 朝"保持原速度"衰减（卡尔曼本身也是这个假设）
        self._update_speed(track, np.full(3, np.nan, np.float32), dt, t)
        track.missed += 1
        track.missed_kpt += 1
        track.last_t = t

    def _update_speed(self, track: _Track, position: np.ndarray, dt: float, t: float) -> None:
        """速度估计：优先用位置卡尔曼的速度状态，退回后向差分；再走一道 EMA。

        为什么要 EMA：15 Hz 下量测噪声（r≈0.02 m）导数化后会产生 ±0.4 m/s 的抖动，
        直接显示会看到速度在 0 和 1 之间乱跳。tau = 0 就关闭平滑。
        """
        if position is not None and np.isfinite(position).all():
            if self.kalman and track.pos_initialized:
                raw = np.array([filt.x[1] for filt in track.pos_filters], dtype=np.float64)
            elif np.isfinite(track.prev_position).all() and dt > 0:
                raw = (np.asarray(position, np.float64) - track.prev_position) / dt
            else:
                raw = np.zeros(3)
            raw = np.where(np.isfinite(raw), raw, 0.0)
            tau = self.metrics.speed_smoothing
            alpha = 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)
            track.velocity = ((1.0 - alpha) * track.velocity + alpha * raw).astype(np.float32)
            track.prev_position = np.asarray(position, dtype=np.float32).copy()
            track.prev_t = t
        speed = float(np.linalg.norm(track.velocity))
        vertical = float(np.asarray(track.velocity, np.float64) @ self.up)
        track.speed = speed
        track.speed_ground = float(np.sqrt(max(speed**2 - vertical**2, 0.0)))

    def _append_trail(self, track: _Track, position: np.ndarray, t: float) -> None:
        """按 trail_interval 限流地记一个轨迹点（0 长度配置就完全不留轨迹）。"""
        if not self.trail_length or not np.isfinite(position).all():
            return
        if t - track.last_trail_t < self.trail_interval:
            return
        track.last_trail_t = t
        track.trail.append(np.asarray(position, dtype=np.float32))

    # ------------------------------------------------------------------ 输出
    def _position_of(self, track: _Track, raw_position: np.ndarray) -> np.ndarray:
        if not self.kalman:
            return np.asarray(raw_position, dtype=np.float32)
        if not track.pos_initialized:  # 从来没测到过位置 -> nan，而不是原点
            return np.full(3, np.nan, dtype=np.float32)
        return np.array([filt.position for filt in track.pos_filters], dtype=np.float32)

    def _display(self, track: _Track, raw_position: np.ndarray, t: float) -> TrackedPerson:
        if self.kalman:
            show = track.seen & (track.missed_kpt <= self.max_missed_kpt)
            points = np.full((NUM_KP, 3), np.nan, dtype=np.float32)
            for i in range(NUM_KP):
                if show[i]:
                    for axis in range(3):
                        points[i, axis] = track.filters[i][axis].position
            joints, names, strips = to_display(points, show)
        else:
            show = track.seen & (track.missed_kpt <= self.max_missed_kpt)
            raw = track.raw if track.raw is not None else np.full((NUM_KP, 3), np.nan, np.float32)
            joints, names, strips = to_display(raw, show)
        position = self._position_of(track, raw_position)
        metrics = self._metrics(track, joints, names, position, t)
        trail = np.asarray(track.trail, dtype=np.float32).reshape(-1, 3)
        # 漏检的轨迹：2D 关键点已经过期，别画（框留着，好看出"人刚刚还在这"）
        kpt_conf = track.kpt_conf if track.missed == 0 else np.zeros(NUM_KP, dtype=np.float32)
        return TrackedPerson(
            id=track.id,
            conf=track.conf,
            box=track.box,
            keypoints2d=track.keypoints2d,
            kpt_conf=kpt_conf,
            joints=joints,
            names=names,
            strips=strips,
            position=position,
            velocity=np.asarray(track.velocity, dtype=np.float32).copy(),
            speed=track.speed,
            speed_ground=track.speed_ground,
            metrics=metrics,
            trail=trail,
            age=track.age,
            missed=track.missed,
            valid_count=int(np.count_nonzero(show)),
            state_speed=self.metrics.speed_deadband,
            fall_level=track.fall.level,
            fall_state=track.fall.state_name,
            fall_since=track.fall.since,
            lying=track.fall.lying,
        )

    def _metrics(
        self, track: _Track, joints, names, position: np.ndarray, t: float
    ) -> dict[str, float]:
        """本帧的人体指标：摔倒状态 + 尺寸（真深度）+ 步态 + 速度。缺项的不出键。"""
        config = self.metrics
        out: dict[str, float] = {}
        fall = self._update_fall(track, joints, names, position, t)
        out.update(fall)
        if not config.enabled:
            return out
        out.update(
            body_metrics(
                joints,
                names,
                up=self.up,
                head_offset=config.head_offset,
                height_range=(config.height_min, config.height_max),
                stand_cos=config.stand_cos,
            )
        )
        out["speed"] = track.speed
        out["speed_ground"] = track.speed_ground

        # 步态：用"踝相对髀中点沿上方向的伸展量"做为周期信号。
        hip = joint_of(joints, names, "left_hip")
        hip_r = joint_of(joints, names, "right_hip")
        if hip is not None and hip_r is not None:
            hip_mid = (hip + hip_r) * 0.5
            signals = []
            for name in ("left_ankle", "right_ankle"):
                ankle = joint_of(joints, names, name)
                signals.append(
                    float((ankle - hip_mid) @ self.up) if ankle is not None else float("nan")
                )
            track.gait.update(t, (signals[0], signals[1]))
            if track.speed_ground >= config.walk_speed:
                cadence = track.gait.cadence()
                if np.isfinite(cadence) and cadence > 0:
                    out["cadence"] = cadence  # 步/分钟
                    out["stride"] = track.speed_ground * 60.0 / cadence  # 米/步
        return out

    def _update_fall(self, track: _Track, joints, names, position, t: float) -> dict[str, float]:
        """跑一帧摔倒状态机，返回要记录的曲线值（torso_angle / fall_level）。

        信号都是实测的：高度 = 躯干中心沿"上"方向的分量（髀中点优先，见
        ``pose3d.person_position``），竖直速度 = 位置卡尔曼的速度状态向"上"投影。
        躺着时关键点常被遮，所以角度测不到就给 nan，靠高度兜底判据。
        """
        detector = track.fall
        if not detector.config.enabled:
            return {}
        hip = joint_of(joints, names, "left_hip")
        hip_r = joint_of(joints, names, "right_hip")
        shoulder = joint_of(joints, names, "left_shoulder")
        shoulder_r = joint_of(joints, names, "right_shoulder")
        angle = float("nan")
        if (
            hip is not None
            and hip_r is not None
            and shoulder is not None
            and shoulder_r is not None
        ):
            torso = (shoulder + shoulder_r) * 0.5 - (hip + hip_r) * 0.5
            angle = torso_angle(torso, self.up)
        height = float("nan")
        if position is not None and np.isfinite(position).all():
            height = float(np.asarray(position, np.float64) @ self.up)
        v_down = -float(np.asarray(track.velocity, np.float64) @ self.up)
        detector.update(t, height=height, v_down=v_down, angle=angle)
        out: dict[str, float] = {"fall_level": float(detector.level)}
        if np.isfinite(detector.angle):
            out["torso_angle"] = float(detector.angle)
        return out

    # ------------------------------------------------------------------ 入口
    def update(
        self,
        poses: list[PoseInstance],
        depth: np.ndarray,
        intrinsics: Intrinsics,
        *,
        threshold: float,
        t: float,
    ) -> list[TrackedPerson]:
        """``poses`` 是本帧的 2D 姿态，``depth`` 是同一帧的对齐深度（米）。

        返回按 id 排序的 ``TrackedPerson``（含刚被删掉前一帧还存在的轨迹都算，
        没命中的轨迹会带着 ``missed>0`` 继续出现在列表里，方便画"刚刚还在"的人）。
        """
        lifted_all = np.full((len(poses), NUM_KP, 3), np.nan, dtype=np.float32)
        valid_all = np.zeros((len(poses), NUM_KP), dtype=bool)
        positions = np.full((len(poses), 3), np.nan, dtype=np.float32)
        boxes = []
        for i, pose in enumerate(poses):
            lifted = lift_keypoints(pose.keypoints, depth, intrinsics)
            lifted_all[i] = lifted
            valid_all[i] = depth_valid_mask(lifted, pose.kpt_conf, threshold)
            positions[i] = person_position(lifted, valid_all[i], pose.xyxy, depth, intrinsics)
            boxes.append(tuple(pose.xyxy))

        matched = self._match(boxes, positions)
        new_tracks: list[tuple[_Track, np.ndarray]] = []
        for bi, pose in enumerate(poses):
            if bi in matched:
                self._advance(
                    self.tracks[matched[bi]],
                    pose,
                    lifted_all[bi],
                    valid_all[bi],
                    positions[bi],
                    t,
                )
            else:
                # 新轨迹这一帧就是它的"命中帧"，不能走下面的 _freeze
                new_tracks.append(
                    (
                        self._new_track(pose, lifted_all[bi], valid_all[bi], positions[bi], t),
                        positions[bi],
                    )
                )

        matched_tracks = set(matched.values())
        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                self._freeze(track, t)
        self.tracks.extend(track for track, _ in new_tracks)

        # 每个人的本帧观测位置（没命中的给 nan，显示时走卡尔曼预测值）
        raw_by_id = {self.tracks[ti].id: positions[bi] for bi, ti in matched.items()}
        raw_by_id.update({track.id: raw for track, raw in new_tracks})
        empty = np.full(3, np.nan, dtype=np.float32)

        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]
        out = [self._display(track, raw_by_id.get(track.id, empty), t) for track in self.tracks]
        out.sort(key=lambda person: person.id)
        return out
