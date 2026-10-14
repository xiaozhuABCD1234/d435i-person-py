# SPDX-License-Identifier: AGPL-3.0-or-later
"""rerun 可视化：2D 原图/框/骨架、对齐深度、相机系 3D 骨架/点云、世界系视图、指标曲线与摔倒警报。

记录到这些实体：

===================================  =======================================
``camera/image``                     彩色原图（框/骨架由下面的结构化实体画）
``camera/image/boxes``               框的 Boxes2D（标签带 id 与置信度）
``camera/image/keypoints``           2D 关键点（带名称）
``camera/image/skeleton``            2D 骨架连线
``camera/depth``                     对齐深度（DepthImage，悬停看米数）
``camera/pose3d``                    相机系 3D 视图根（RIGHT_HAND_Y_DOWN）
``camera/pose3d/points``             彩色点云
``camera/pose3d/{joints,skeleton}``  3D 骨架（真深度）
``camera/pose3d/trails``             每个人的运动轨迹
``camera/pose3d/positions``          每个人的 3D 位置（标签 = id + 距离）
``world/...``                       世界系 3D 视图根（RIGHT_HAND_Z_UP，可选）
``world/{points,joints,skeleton,trails,positions}``  世界系副本（刚体变换后）
``person/metrics/<组>/<指标>/<id>``  人体指标曲线（按组拆框：速度/尺寸/长度/步态/摔倒/可信度）
``person/values``                    极简数值行（每人一行短文本，报警时带 ⚠ 摔倒）
``person/coords/<id>/{x,y,z}``       每个人的坐标曲线（相机系，米，默认关）
``person/count``                     人数（标量，默认不占面板）
``person/events``                    人数变化日志
``person/fall_light``                摔倒报警灯（一格 2D：整格变色 + 格内文字；红=摔倒会闪）
``person/fall_alert``                报警灯的纯文本副本（markdown，可选中复制）
``person/fall``                      摔倒状态变化日志（只在跳变时记一条）
===================================  =======================================

右上角默认是**按组拆开的指标曲线框**（速度 / 尺寸 / 长度 / 步态 / 可信度，用
``metrics_views`` 选组、``metrics_layout`` 选摆法）+ 一列极简数值行（每人一行，
不是大段表格），用 ``show_metrics`` / ``show_values`` / ``show_coords`` /
``show_count`` 控制。每个人的颜色由 id 决定，所以同一 id 在 2D、3D 以及各个
曲线框里都是同一种颜色（曲线颜色是显式声明的，不靠路径哈希）。

进 rerun 前统一 `bgr_to_rgb()`：`rr.Image` 默认按 RGB 解读三通道数据，直接喂 BGR
会红蓝互换。
"""

from __future__ import annotations

import contextlib
import sys
from collections import deque

import numpy as np

from .colors import bgr_to_rgb
from .fall import FALLEN, NORMAL, SUSPECT
from .pose import bones, visible_keypoints

# 按 id 轮转的颜色（RGB）。挑的是互相区分度高、在深色背景上也看得清的几种。
PALETTE: tuple[tuple[int, int, int], ...] = (
    (90, 220, 120),
    (255, 180, 60),
    (120, 180, 255),
    (255, 120, 160),
    (180, 140, 255),
    (120, 230, 230),
    (240, 220, 90),
    (150, 255, 140),
)
POSITION_RADIUS = 0.04  # 米：3D 位置球半径
JOINT_RADIUS = 0.02  # 米：关节球半径
ALARM_COLOR = (255, 64, 64)  # 摔倒报警色（2D/3D 都换成它，比“看曲线”快得多）

# 摔倒报警灯：占「摔倒警报」那一格（2D 视图：整格一块颜色 + 白字）。
# 用颜色而不是表情符号：颜色是屏幕上最先被看见的东西，也不会有字体缺字的问题。
LIGHT_ORIGIN = "person/fall_light"
LIGHT_STATE = f"{LIGHT_ORIGIN}/state"  # 铺满整格的色块（2×2 纯色图）
LIGHT_TEXT = f"{LIGHT_ORIGIN}/text"  # 格内文字（2D 标签：屏幕像素大小，格子小也看得清）
LIGHT_COLORS: dict[int, tuple[int, int, int]] = {
    NORMAL: (48, 200, 96),  # 绿：正常
    SUSPECT: (255, 176, 32),  # 琥珀：疑似倒地
    FALLEN: (255, 64, 64),  # 红：摔倒（跟“整个人变红”同一个色）
}
LIGHT_IDLE_COLOR = (90, 100, 120)  # 画面内无人：暗蓝灰（故意不用绿——没人在 ≠ 一切正常）
LIGHT_TEXT_COLOR = (255, 255, 255)
LIGHT_FLASH_DIM = 0.35  # 闪烁的暗相位保留多少亮度（还看得出是红灯）
# 色块怎么画：这个版本的 rr.Boxes2D 只画边框（没有 fill_mode），Points2D 只能画圆盘，
# 真正“铺满整格的矩形”只能用纯色小图（试出来的：1×1 的图 rerun 会画成白屏，2×2 正常）。
# 图占 [0,SIZE]²，所以坐标就用图像自己的 [0,SIZE]²：即使 visual_bounds 取不到
# （老版本退回自动取景），文字也还在色块中间。
LIGHT_FILL_SIZE = 2
# 文字锚点（图像坐标）：rerun 的 2D 标签以锚点为中心、向上展开，所以锚点给图中心稍下一点，
# 几行字才落在色块中间；x 给图中心才是左右居中（靠边会被格子裁掉）。
LIGHT_TEXT_ANCHOR = (LIGHT_FILL_SIZE / 2, 0.935)
ALERT_TEXT_ORIGIN = "person/fall_alert"  # 纯文本副本（可选中/可复制，记着不花什么钱）


def color_for(person_id: int) -> tuple[int, int, int]:
    """同一个 id 永远同一个颜色（id 从 1 开始，单调递增）。"""
    return PALETTE[(int(person_id) - 1) % len(PALETTE)]


def person_color(person) -> tuple[int, int, int]:
    """报警时整个人变红：视觉上比“某个框里多一条线”快得多。"""
    return ALARM_COLOR if getattr(person, "fall_level", 0) >= 2 else color_for(person.id)


def fall_alert(person) -> str:
    """数值行里的摔倒标注：确认了才写“⚠ 摔倒”（带躯干角，方便一眼判断是不是真的躺下了），
    只是倒地中的写“倒地中”，仅挨一下冲击（跳、快速蹲下）不写——别把正常动作显示成警报。"""
    level = getattr(person, "fall_level", 0)
    if level >= 2:
        angle = person.metrics.get("torso_angle")
        if angle is not None and np.isfinite(angle):
            return f"⚠ 摔倒 {angle:.0f}°"
        return "⚠ 摔倒"
    if level == 1 and getattr(person, "lying", False):
        return "倒地中"
    return ""


# 极简数值行的字段顺序与单位（键 -> (短标签, 单位, 小数位)）
VALUE_FIELDS: tuple[tuple[str, str, str, int], ...] = (
    ("height", "高", "m", 2),
    ("shoulder_width", "肩宽", "m", 2),
    ("hip_width", "髀宽", "m", 2),
    ("cadence", "步频", "步/分", 0),
    ("stride", "步幅", "m", 2),
)

# 指标分组：一个框里塞十几条线根本看不清，按"什么时候会一起看"拆成几个视图。
# 键 = 实体路径里的那一段（person/metrics/<键>/<指标>/<id>），也是 [rerun].metrics_views
# 里能选的组名（也可以用单个指标名，比如只要看 height 就写 height）。
METRIC_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("speed", "速度", ("speed", "speed_ground")),
    ("size", "尺寸（高/肩/髀）", ("height", "shoulder_width", "hip_width")),
    ("length", "长度（躯干/腿/臂展）", ("torso", "leg", "arm_span")),
    ("gait", "步态（步频/步幅）", ("cadence", "stride")),
    ("fall", "摔倒（躯干角/等级）", ("torso_angle", "fall_level")),
    ("quality", "可信度", ("coverage",)),
)

# 摔倒警报：不是曲线（曲线要去找），而是报警灯——整格一块颜色 + 白字。
# 默认占一个格子（取代原来的“可信度”）：报警要一眼看到。coverage 曲线照记，
# 想要回那个格子就把它写进 metrics_views（"all,quality"）。
ALARM_GROUP = "alarm"
ALARM_ORIGIN = LIGHT_ORIGIN  # 那一格记的是报警灯（person/fall_light）
ALARM_LABEL = "摔倒警报"

# 指标 -> 组；没列进去的新指标落到 "other"（数据照记，只是没预置视图）。
METRIC_GROUP_OF: dict[str, str] = {
    metric: group for group, _, metrics in METRIC_GROUPS for metric in metrics
}

# 默认格子（顺序就是摆的顺序）：五个曲线组 + 摔倒警报框。
DEFAULT_METRIC_GROUPS: tuple[str, ...] = (
    "speed",
    "size",
    "length",
    "gait",
    "fall",
    ALARM_GROUP,
)


def metric_path(metric: str, person_id: int) -> str:
    """指标曲线的实体路径：按组分层，TimeSeriesView 才能一组一个框。"""
    group = METRIC_GROUP_OF.get(metric, "other")
    return f"person/metrics/{group}/{metric}/{person_id}"


def _parse_metric_views(spec: str | None) -> list[tuple[str, str, str]]:
    """把 ``--metrics-views`` 解析成 [(origin, 图名, 组名)]。

    ``"all"``/空 = 默认那几格（五个曲线组 + 摔倒警报，**不含 coverage**）；否则用
    逗号/空格分隔的组名、单个指标名或 ``alarm``，比如 ``speed,size,alarm``。
    选到单个指标时就只给它一个框（origin 落到那一级）。
    """
    labels = {group: label for group, label, _ in METRIC_GROUPS}
    known = {**labels, ALARM_GROUP: ALARM_LABEL}
    defaults = list(DEFAULT_METRIC_GROUPS)
    is_all = spec is None or spec.strip().lower() in {"all", "*"}
    explicit = [] if is_all else spec.replace(",", " ").split()

    def entry_of(token: str) -> tuple[str, str, str] | None:
        """组名 -> (origin, 图名, 组名)；不是组名就 None（下面再试指标名）。"""
        if token == ALARM_GROUP:
            return (ALARM_ORIGIN, ALARM_LABEL, ALARM_GROUP)
        if token in labels:
            return (f"person/metrics/{token}", labels[token], token)
        return None

    picked: list[tuple[str, str, str]] = []
    unknown: list[str] = []
    for token in explicit or defaults:
        if token in {"all", "*"}:
            # 注意是拼 entry，不是 extend(组名字符串)——那样会把名字按字符拆散
            picked.extend(entry for group in defaults if (entry := entry_of(group)))
            continue
        entry = entry_of(token)
        if entry is not None:
            picked.append(entry)
        elif token in METRIC_GROUP_OF:
            group = METRIC_GROUP_OF[token]
            picked.append((f"person/metrics/{group}/{token}", token, group))
        else:
            unknown.append(token)
    if unknown:
        # 名字打错就静静退回默认布局，太容易让人以为自己没改对，影响一下
        options = " ".join(known) + " 或单个指标名（如 height）"
        print(f"[viz] 认不出的指标组：{', '.join(unknown)}；可选：{options}", file=sys.stderr)
    return picked


def value_line(person) -> str:
    """一个人一行短文本：只列本帧真有的指标，不占地方也看不到 nan。"""
    parts = [f"#{person.id}"]
    alert = fall_alert(person)
    if alert:
        parts.append(alert)
    if np.isfinite(person.distance):
        parts.append(f"距 {person.distance:.2f} m")
    if np.isfinite(person.speed_ground):
        parts.append(f"速 {person.speed_ground:.2f} m/s")
        parts.append(person.state)
    for key, label, unit, digits in VALUE_FIELDS:
        value = person.metrics.get(key)
        if value is not None and np.isfinite(value):
            parts.append(f"{label} {value:.{digits}f} {unit}")
    return " · ".join(parts)


def _fall_detail(person, now: float | None = None, *, with_state: bool = True) -> str:
    """纯文本副本里的一行（长写法）：谁、什么状态、躯干多横、多远、已经多久了。

    ``with_state=False`` 时省掉状态名（标题里已经写了，明细行重复就啰嗦）。
    """
    level = int(getattr(person, "fall_level", 0))
    parts = [f"#{person.id}"]
    if with_state:
        parts.append(person.fall_state if level >= 2 else "倒地中")
    angle = person.metrics.get("torso_angle")
    if angle is not None and np.isfinite(angle):
        parts.append(f"躯干 {angle:.0f}°")
    if np.isfinite(person.distance):
        parts.append(f"距 {person.distance:.1f} m")
    since = getattr(person, "fall_since", None)
    if since is not None and now is not None:
        parts.append(f"已 {max(now - since, 0.0):.1f} s")
    return " · ".join(parts)


def _light_detail(person, now: float | None = None) -> str:
    """报警灯格内的一行（短写法）：``#1 · 躯干 90° · 4.1 m · 0.7 s``。

    格内文字是居中画的、不会自动折行，太长就左右被格子裁掉，所以这里去掉“距/已”这类字，
    只留数字与单位。
    """
    parts = [f"#{person.id}"]
    angle = person.metrics.get("torso_angle")
    if angle is not None and np.isfinite(angle):
        parts.append(f"躯干 {angle:.0f}°")
    if np.isfinite(person.distance):
        parts.append(f"{person.distance:.1f} m")
    since = getattr(person, "fall_since", None)
    if since is not None and now is not None:
        parts.append(f"{max(now - since, 0.0):.1f} s")
    return " · ".join(parts)


def fall_light(
    people, history=(), now: float | None = None, *, flash_hz: float = 2.0
) -> tuple[tuple[int, int, int], list[str]]:
    """报警灯：返回 (颜色, 文字行)。纯函数，不依赖 rerun，方便单测。

    四档颜色：红（摔倒）/ 琥珀（疑似倒地）/ 绿（正常）/ 暗蓝灰（画面内无人）。
    ``flash_hz`` > 0 时红灯按这个频率明暗交替——报警灯就该闪；其余档位常亮。
    """
    alive = list(people)
    alarmed = [p for p in alive if int(getattr(p, "fall_level", 0)) >= FALLEN]
    down = [
        p
        for p in alive
        if int(getattr(p, "fall_level", 0)) == SUSPECT and getattr(p, "lying", False)
    ]
    if alarmed:
        who = "、".join(f"#{p.id}" for p in alarmed)
        head = f"摔倒 {who}" + ("（多人）" if len(alarmed) > 1 else "")
        color = LIGHT_COLORS[FALLEN]
    elif down:
        head = "疑似倒地 " + "、".join(f"#{p.id}" for p in down)
        color = LIGHT_COLORS[SUSPECT]
    elif alive:
        head, color = "正常", LIGHT_COLORS[NORMAL]
    else:
        head, color = "画面内无人", LIGHT_IDLE_COLOR
    if (
        color is LIGHT_COLORS[FALLEN]
        and flash_hz > 0
        and now is not None
        and int(now * flash_hz) % 2
    ):
        color = tuple(int(c * LIGHT_FLASH_DIM) for c in color)  # 暗相位：暗下去但还看得出是红的
    lines = [head, *(_light_detail(p, now) for p in [*alarmed, *down])]
    # 格内只放最近一次状态变化（多了会把格子占满，完整的在纯文本副本里）
    lines += [f"最近 {t:.1f}s {short}" for t, short, _ in list(history)[-1:]]
    return color, lines


def fall_alert_panel(people, history=(), now: float | None = None) -> str:
    """报警灯的纯文本副本：标题 + 报警的人各一行 + 最近几条状态变化。

    报警灯（2D 色块）是给眼睛看的，这份文本是给“选中/复制/搜索”用的——两者内容同源。
    单独拎出来也是为了可测（不依赖 rerun）。
    """
    alive = list(people)
    alarmed = [p for p in alive if int(getattr(p, "fall_level", 0)) >= FALLEN]
    down = [
        p
        for p in alive
        if int(getattr(p, "fall_level", 0)) == SUSPECT and getattr(p, "lying", False)
    ]
    if alarmed:
        who = "、".join(f"#{p.id}" for p in alarmed)
        head = f"# 摔倒 {who}" + ("（多人）" if len(alarmed) > 1 else "")
    elif down:
        head = "# 疑似倒地 " + "、".join(f"#{p.id}" for p in down)
    elif alive:
        head = "正常"
    else:
        head = "画面内无人"
    lines = [head, *(f"- {_fall_detail(person, now)}" for person in [*alarmed, *down])]
    if history:
        lines += ["", "最近：" + "；".join(f"{t:.1f}s {long}" for t, _, long in history)]
    return "\n".join(lines)


def _blueprint_module():
    """blueprint API 的位置在 rerun 0.23 前后变过：优先 rerun.blueprint，退回 rerun。"""
    try:
        import rerun.blueprint as rrb
    except ImportError:  # pragma: no cover - 老版本
        import rerun as rrb

    return rrb


class RerunView:
    def __init__(
        self,
        app_id: str = "d435i-person",
        *,
        spawn: bool = True,
        save_path: str | None = None,
        kpt_conf: float = 0.5,
        keep_history: bool = False,
        memory_limit: str = "512MB",
        server_memory_limit: str = "512MB",
        jpeg_quality: int = 75,
        port: int = 9876,
        show_depth: bool = True,
        show_pose3d: bool = True,
        show_points: bool = True,
        eye: tuple[float, float, float] = (0.0, 0.0, -1.5),
        look_at: tuple[float, float, float] = (0.0, 0.0, 2.5),
        eye_up: tuple[float, float, float] = (0.0, -1.0, 0.0),
        world=None,
        show_world_view: bool = False,
        world_pointcloud: bool = True,
        show_coords: bool = True,
        show_count: bool = False,
        show_metrics: bool = True,
        show_values: bool = True,
        metrics_views: str = "all",
        metrics_layout: str = "grid",
        fall_light_flash: float = 2.0,
    ):
        """跑起来就弹 viewer（除非给了 save_path）。

        viewer 只当显示器：不给 save_path 就不往磁盘写任何东西。
        keep_history=False（默认）时图像/框/关键点用 static 记录，viewer 只保留最新一份，
        没有时间轴可回拖，内存不随帧数涨；memory_limit 是兼底硬上限。
        """
        import rerun as rr

        self.rr = rr
        self.kpt_conf = kpt_conf
        self.keep_history = keep_history
        self.jpeg_quality = int(jpeg_quality)
        self.show_depth = show_depth
        self.show_pose3d = show_pose3d
        self.show_points = show_points
        self.eye = tuple(eye)
        self.look_at = tuple(look_at)
        self.eye_up = tuple(eye_up)
        self.world = world
        self.show_world_view = bool(show_world_view and world is not None)
        self.world_pointcloud = bool(world_pointcloud and world is not None)
        self.show_coords = bool(show_coords)
        self.show_count = bool(show_count)
        self.show_metrics = bool(show_metrics)
        self.show_values = bool(show_values)
        # 想少几个框：--metrics-views speed,height （组名或单个指标名，逗号/空格分隔）
        self.metric_views = _parse_metric_views(metrics_views)
        # grid = 2 行多列的格子；stack = 一框一行地堆；single = 全塞回一个框
        self.metrics_layout = (
            metrics_layout if metrics_layout in {"grid", "stack", "single"} else "grid"
        )
        # 报警灯闪烁频率（Hz）：红灯明暗交替；0 = 常亮
        self.fall_light_flash = max(float(fall_light_flash), 0.0)
        self._last_count: int | None = None
        self._image_size: tuple[int, int] | None = None
        self._coords_logged: set[str] = set()
        self._styled_series: set[str] = set()  # 已经声明过颜色的曲线（只声明一次）
        self._fall_levels: dict[int, int] = {}  # 每个人的摔倒等级，用于只在跳变时记事件
        self._fall_history: deque = deque(maxlen=3)  # 报警灯里附的“最近”几条 (时刻, 短, 长)

        rr.init(app_id, spawn=False)
        if save_path:
            rr.save(str(save_path))
        elif spawn:
            # rerun 默认 memory_limit="75%"（本机约 23 GB），这里显式压住
            rr.spawn(
                port=port,
                memory_limit=memory_limit,
                server_memory_limit=server_memory_limit,
                hide_welcome_screen=True,
            )

    # ---------------------------------------------------------------- 布局
    def _send_blueprint(self, width: int, height: int) -> None:
        """布局 + 把 2D 视图的取景钉死在图像范围内。

        不设 visual_bounds 时，rerun 按"视图内数据的包围盒"自动取景：框/关键点一旦
        伸出画面（人半出画就是这样），取景就跟着放大缩小，看起来像画面在自动缩放。
        """
        rrb = _blueprint_module()
        view_args: dict = {"origin": "camera/image", "name": "彩色 + 骨架"}
        with contextlib.suppress(AttributeError, TypeError):
            view_args["visual_bounds"] = rrb.VisualBounds2D(x_range=[0, width], y_range=[0, height])
        try:
            left = rrb.Spatial2DView(**view_args)
            if self.show_depth:
                # 左列：上面原图、下面对齐深度
                left = rrb.Vertical(
                    left,
                    rrb.Spatial2DView(origin="camera/depth", name="对齐深度"),
                    row_shares=[3, 2],
                )

            right_panels: list = []
            shares: list[int] = []
            top: list = []
            top_shares: list[int] = []
            if self.show_values:
                # 极小的一段：每人一行短文本（速度/身高/肩宽…）
                top.append(rrb.TextDocumentView(origin="person/values", name="数值"))
                top_shares.append(1)
            if self.show_metrics:
                # 指标曲线按组拆框：一个框里十几条线根本看不清
                top.append(self._metrics_panel(rrb))
                top_shares.append(5)
            if self.show_coords:
                top.append(rrb.TimeSeriesView(origin="person/coords", name="坐标"))
                top_shares.append(2)
            if self.show_count:
                top.append(rrb.TimeSeriesView(origin="person/count", name="人数"))
                top_shares.append(1)
            if top:
                panel = top[0] if len(top) == 1 else rrb.Vertical(*top, row_shares=top_shares)
                right_panels.append(panel)
                shares.append(2 if len(top) <= 2 else 3)
            if self.show_pose3d:
                right_panels.append(self._make_3d_view(rrb, "camera/pose3d", "3D 骨架 / 点云"))
                shares.append(3 if not self.show_world_view else 2)
            if self.show_world_view:
                right_panels.append(self._make_3d_view(rrb, "world", "世界系"))
                shares.append(2)

            right = right_panels[0] if right_panels else None
            if len(right_panels) > 1:
                right = rrb.Vertical(*right_panels, row_shares=shares)

            library = rrb.Horizontal(left, right, column_shares=[3, 2]) if right else left
            blueprint = rrb.Blueprint(library, rrb.TimePanel(state="collapsed"))
        except AttributeError:
            # blueprint API 不可用就用 rerun 的默认布局，不影响记录本身
            return
        self.rr.send_blueprint(blueprint)

    def _metrics_panel(self, rrb):
        """指标曲线面板：单个组 / Grid 多框 / 一框一行地堆。

        ``metrics_views`` 选组（默认全上），``metrics_layout`` 定摆法。单组时直接
        给个 TimeSeriesView（不再套容器，省一层边框）。
        """
        picked = self.metric_views or _parse_metric_views("all")
        views = [
            self._alarm_view(rrb, origin, name)
            if group == ALARM_GROUP
            else rrb.TimeSeriesView(origin=origin, name=name)
            for origin, name, group in picked
        ]
        if self.metrics_layout == "single":
            # 想要回“一个大框全在里面”就 single
            return rrb.TimeSeriesView(origin="person/metrics", name="人体指标")
        if len(views) == 1:
            return views[0]
        if self.metrics_layout == "stack":
            return rrb.Vertical(*views)
        grid = getattr(rrb, "Grid", None)
        if grid is None:  # pragma: no cover - 老版本没有 Grid
            return rrb.Vertical(*views)
        return grid(*views)

    def _alarm_view(self, rrb, origin: str, name: str):
        """报警灯那一格：2D 视图，坐标系锁在色块的 [0,2]² 上。

        不锁的话 rerun 会按内容自动取景；锁住后色块就是“整格一块颜色”，与报警灯的样子一致
        （内容就是 [0,2]² 的色块，所以两种情况下画面一样）。
        """
        bounds = getattr(rrb, "VisualBounds2D", None)
        if bounds is not None:
            with contextlib.suppress(Exception):  # 老版本没有 visual_bounds
                return rrb.Spatial2DView(
                    origin=origin,
                    name=name,
                    visual_bounds=bounds(
                        x_range=[0.0, LIGHT_FILL_SIZE], y_range=[0.0, LIGHT_FILL_SIZE]
                    ),
                )
        return rrb.Spatial2DView(origin=origin, name=name)

    def _make_3d_view(self, rrb, origin: str, name: str):
        """3D 视图，眼位钉在相机坐标系的默认位置。

        不设 ``eye_controls`` 时 rerun 会按数据的包围盒自动取景，深度逐帧抖动、点云
        远近变化，画面就跟着"自动缩放"。这里把眼位/视点/上方向固定下来；想转视角在
        viewer 里鼠标拖即可。
        """
        eye_cls = getattr(rrb, "EyeControls3D", None)
        controls = None
        if eye_cls is not None:
            with contextlib.suppress(Exception):  # 老版本没有 kind/这些字段
                controls = eye_cls(
                    kind="Orbital",
                    position=list(self.eye),
                    look_target=list(self.look_at),
                    eye_up=list(self.eye_up),
                )
        if controls is None:
            return rrb.Spatial3DView(origin=origin, name=name)
        return rrb.Spatial3DView(origin=origin, name=name, eye_controls=controls)

    # ------------------------------------------------------------ 记录工具
    def _set_time(self, frame_number: int) -> None:
        # 0.23+ 用 set_time(...)，更早的版本用 set_time_sequence
        rr = self.rr
        if hasattr(rr, "set_time"):
            rr.set_time("frame", sequence=frame_number)
        else:  # pragma: no cover - 老版本
            rr.set_time_sequence("frame", frame_number)

    def _log(self, entity: str, data) -> None:
        """默认 static：不带时间戳，viewer 只保留最新一份，没有历史可回拖。

        图像/点云是内存大头（640×480 原图 0.92 MB/帧，stride=4 的点云也差不多），
        它们不留历史后 viewer 的内存就不随帧数涨；人数曲线需要时间戳，在
        ``log_frame`` 里单独 set_time 后写。
        """
        if self.keep_history:
            self.rr.log(entity, data)
        else:
            self.rr.log(entity, data, static=True)

    def _log_scalar(
        self, entity: str, value: float, *, person_id: int | None = None, name: str | None = None
    ) -> None:
        """标量曲线：**带时间戳**（跟 ``_log`` 的 static 相反），否则画不成曲线。

        样式（颜色 + 图例名）只在第一次记：不写的话 rerun 会按路径哈希分配颜色，
        同一个人在不同指标框里可能是不同颜色，而且图例显示的是完整实体路径。
        现在每个实体自己声明颜色和 ``#id`` 图例名，各框之间颜色一致。
        """
        rr = self.rr
        scalar = getattr(rr, "Scalars", None) or rr.Scalar
        rr.log(entity, scalar(float(value)))
        if entity in self._styled_series:
            return
        self._styled_series.add(entity)
        lines = getattr(rr, "SeriesLines", None)
        if lines is None:  # pragma: no cover - 老版本没有 SeriesLines
            return
        kwargs: dict = {}
        if person_id is not None:
            kwargs["colors"] = [color_for(person_id)]
        if name or person_id is not None:
            kwargs["names"] = [name or f"#{person_id}"]
        if not kwargs:
            return
        with contextlib.suppress(Exception):
            rr.log(entity, lines(**kwargs), static=True)

    def _encode(self, bgr: np.ndarray):
        """按 jpeg_quality 压过再送。

        640×480 原图每帧 0.92 MB，viewer 得把它整份留着供回拖；压成 JPEG 后每帧约
        0.05 MB。压过的只是一份显示副本，`--save-dir` 存的图仍是原分辨率。
        """
        rr = self.rr
        image = rr.Image(bgr_to_rgb(bgr))
        if self.jpeg_quality > 0 and hasattr(image, "compress"):
            return image.compress(jpeg_quality=self.jpeg_quality)
        return image

    # ---------------------------------------------------------------- 2D
    def log_frame(self, frame, people) -> None:
        """记一帧的 2D 内容：原图 + 结构化框/关键点/骨架 + 人数。

        ``frame.bgr`` 必须是**相机原图**，不要传 ``result.plot()`` 画过的图：框、
        关键点、骨架都由 rerun 用结构化数据在这里画，传标注图会让同一份信息画两遍。
        所有叠加信息都从 ``people``（带 id 的跟踪结果）来，所以 2D 上也看得到 id。
        """
        rr = self.rr
        number = frame.frame_number
        if self.keep_history:
            self._set_time(number)

        # 分辨率变了（换相机/换模式）才重发 blueprint，把取景钉到新尺寸
        height, width = frame.bgr.shape[:2]
        if (width, height) != self._image_size:
            self._image_size = (width, height)
            self._send_blueprint(width, height)

        self._log("camera/image", self._encode(frame.bgr))
        self._log_boxes(people)
        self._log_poses2d(people)

        live = [person for person in people if person.missed == 0]
        count = len(live)
        if count != self._last_count:
            self._last_count = count
            self._log(
                "person/events",
                rr.TextLog(
                    f"{count} person(s): " + ", ".join(f"#{d.id} conf={d.conf:.2f}" for d in live)
                ),
            )
        self._log_fall_events(live, frame.host_time)
        self._log_fall_light(live, frame.host_time)
        self._log_fall_alert(live, frame.host_time)

        # 标量一直带时间戳（曲线靠它们画）：人数、人体指标、以及每个人的位置坐标。
        self._set_time(number)
        self._log_scalar("person/count", count, name="人数")
        if self.show_metrics or self.show_coords:
            for person in live:
                if self.show_metrics:
                    for key, value in person.metrics.items():
                        if np.isfinite(value):
                            self._log_scalar(
                                metric_path(key, person.id), float(value), person_id=person.id
                            )
                if self.show_coords and np.isfinite(person.position).all():
                    for axis, value in zip("xyz", person.position, strict=True):
                        self._log_scalar(
                            f"person/coords/{person.id}/{axis}", float(value), person_id=person.id
                        )
        if self.show_values:
            self._log_values(live)

    def _log_boxes(self, people) -> None:
        rr = self.rr
        cls = getattr(rr, "Boxes2D", None) or rr.Box2D  # 0.14 叫 Box2D
        if not people:
            empty = np.zeros((0, 2), dtype=np.float32)
            self._log("camera/image/boxes", cls(centers=empty, sizes=empty))
            return
        xyxy = np.array([d.box for d in people], dtype=np.float32)
        centers = np.stack(
            [(xyxy[:, 0] + xyxy[:, 2]) * 0.5, (xyxy[:, 1] + xyxy[:, 3]) * 0.5], axis=1
        )
        sizes = np.stack([xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], axis=1)
        self._log(
            "camera/image/boxes",
            cls(
                centers=centers,
                sizes=sizes,
                labels=[person.label for person in people],
                colors=[person_color(person) for person in people],
            ),
        )

    def _log_poses2d(self, people) -> None:
        """2D 关键点用 Points2D（带名字），骨头用 LineStrips2D；颜色跟着 id。"""
        rr = self.rr
        points: list[np.ndarray] = []
        labels: list[str] = []
        colors: list[tuple[int, int, int]] = []
        strips: list[np.ndarray] = []
        strip_colors: list[tuple[int, int, int]] = []
        for person in people:
            color = person_color(person)
            visible, names = visible_keypoints(
                person.keypoints2d, person.kpt_conf, threshold=self.kpt_conf
            )
            if len(visible):
                points.append(visible)
                labels.extend(f"#{person.id} {name}" for name in names)
                colors.extend([color] * len(visible))
            for strip in bones(person.keypoints2d, person.kpt_conf, threshold=self.kpt_conf):
                strips.append(strip)
                strip_colors.append(color)

        self._log(
            "camera/image/keypoints",
            rr.Points2D(
                np.concatenate(points) if points else np.zeros((0, 2), dtype=np.float32),
                labels=labels,
                colors=colors or None,
                radii=3.0,
            ),
        )
        self._log("camera/image/skeleton", rr.LineStrips2D(strips, colors=strip_colors))

    def _log_fall_events(self, live, t: float) -> None:
        """摔倒状态一变就记一条 TextLog（rerun 的文本日志面板里能看到整串时间线）。

        只在跳变时记，不是每帧都记：报警从“倒地中”到“摔倒”再到“已恢复”一共就几条。
        历史存三元组 (时刻, 短写法, 长写法)：报警灯那格用短写法（格子窄，长了会被裁），
        纯文本副本用长写法。文案不带表情符号（这些字会进报警灯那一格，那格要求无表情）。
        """
        rr = self.rr
        for person in live:
            level = int(getattr(person, "fall_level", 0))
            if level == self._fall_levels.get(person.id, 0):
                continue
            self._fall_levels[person.id] = level
            angle = person.metrics.get("torso_angle")
            detail = f"（躯干 {angle:.0f}°）" if angle is not None and np.isfinite(angle) else ""
            if level >= FALLEN:
                text, short = f"#{person.id} 检测到摔倒{detail}", f"#{person.id} 摔倒"
            elif level == SUSPECT:
                text, short = f"#{person.id} 疑似摔倒 / 倒地中{detail}", f"#{person.id} 疑似"
            else:
                text, short = f"#{person.id} 已恢复正常", f"#{person.id} 恢复"
            self._fall_history.append((t, short, text))
            self._log("person/fall", rr.TextLog(text))

    def _log_fall_light(self, live, t: float) -> None:
        """摔倒报警灯：一格 2D 视图里画一块颜色 + 白字（不再是表情符号）。

        红=摔倒（还会闪）/ 琥珀=疑似倒地 / 绿=正常 / 暗蓝灰=画面内无人。闪烁靠“每帧重新
        记一次颜色”实现，所以 keep_history=False 的 static 记录也照样闪。
        """
        rr = self.rr
        points = getattr(rr, "Points2D", None)
        if points is None:  # pragma: no cover - 老版本
            return
        color, lines = fall_light(live, self._fall_history, t, flash_hz=self.fall_light_flash)
        # 纯色小图铺满整格：那一格本身就是“色块”，文字画在它里面（居中）
        self._log(
            LIGHT_STATE,
            rr.Image(np.full((LIGHT_FILL_SIZE, LIGHT_FILL_SIZE, 3), color, dtype=np.uint8)),
        )
        # 2D 标签的文字是屏幕像素大小：格子缩小也不会看不见（不像图像里的字会跟着缩）；
        # 锚点为什么在那儿见 LIGHT_TEXT_ANCHOR 的注释（居中 + 向上展开）。
        self._log(
            LIGHT_TEXT,
            points(
                positions=[list(LIGHT_TEXT_ANCHOR)],
                labels=["\n".join(lines)],
                colors=[LIGHT_TEXT_COLOR],
                radii=[0.0],
            ),
        )

    def _log_fall_alert(self, live, t: float) -> None:
        """报警灯的纯文本副本（markdown）：报警灯是给眼睛看的，这份给“选中/复制”用。

        成本几乎为零（static，只保留最新一份），但少了它就没法把状态整段复制出来。
        """
        document = getattr(self.rr, "TextDocument", None)
        if document is None:  # pragma: no cover - 老版本没有 TextDocument
            return
        media_type = getattr(self.rr, "MediaType", None)
        kwargs = {"media_type": media_type.MARKDOWN} if media_type is not None else {}
        text = fall_alert_panel(live, self._fall_history, t)
        self._log(ALERT_TEXT_ORIGIN, document(text, **kwargs))

    def _log_values(self, live) -> None:
        """极简数值行：每人一行短文本（距/速/身高/肩宽/髀宽/步频/步幅）。

        刻意不用表格：一是表格在窄窗口里要横向拖，二是"大段文字"把图像挤没了。
        每行只列本帧真有的指标（缺关节的项直接省略），所以看不到 nan。
        """
        rr = self.rr
        document = getattr(rr, "TextDocument", None)
        if document is None:  # pragma: no cover - 老版本没有 TextDocument
            return
        lines = [f"- {value_line(person)}" for person in live] or ["- 无人"]
        media_type = getattr(rr, "MediaType", None)
        kwargs = {"media_type": media_type.MARKDOWN} if media_type is not None else {}
        self._log("person/values", document("\n".join(lines), **kwargs))

    # ---------------------------------------------------------------- 深度/点云
    def log_depth(self, depth: np.ndarray, *, max_width: int = 480) -> None:
        """把对齐深度（米）送进 viewer（Turbo 伪彩，悬停看米数）。

        nan（无效深度）先换成 0，否则 rerun 侧上色时会出现一片"贴到镜头前"的假点。
        """
        rr = self.rr
        step = max(1, -(-depth.shape[1] // max_width))  # 向上取整的整除
        downsampled = np.nan_to_num(depth[::step, ::step], nan=0.0)
        downsampled = np.ascontiguousarray(downsampled, dtype=np.float32)
        kwargs: dict = {"meter": 1.0}
        colormap = getattr(getattr(rr, "components", None), "Colormap", None)
        if colormap is not None:
            kwargs["colormap"] = colormap.Turbo
        self._log("camera/depth", rr.DepthImage(downsampled, **kwargs))

    def log_points(self, positions: np.ndarray, colors: np.ndarray) -> None:
        """彩色点云（相机坐标系，米）。世界系开着就顺便再记一份变换后的。"""
        rr = self.rr
        self._log(
            "camera/pose3d/points",
            rr.Points3D(np.asarray(positions, dtype=np.float32), colors=colors, radii=0.004),
        )
        if self.world_pointcloud and len(positions):
            self._log(
                "world/points",
                rr.Points3D(
                    np.asarray(self.world.apply(positions), dtype=np.float32),
                    colors=colors,
                    radii=0.004,
                ),
            )

    # ---------------------------------------------------------------- 3D
    def log_people3d(self, people) -> None:
        """记 3D：相机系的骨架/轨迹/位置，世界系开着就再记一份（刚体变换后）。"""
        rr = self.rr
        self._log_coords("camera/pose3d", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)
        if self.world is not None:
            self._log_coords("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)

        joints: list[np.ndarray] = []
        labels: list[str] = []
        colors: list[tuple[int, int, int]] = []
        strips: list[np.ndarray] = []
        strip_colors: list[tuple[int, int, int]] = []
        trails: list[np.ndarray] = []
        trail_colors: list[tuple[int, int, int]] = []
        positions: list[np.ndarray] = []
        position_labels: list[str] = []
        position_colors: list[tuple[int, int, int]] = []

        for person in people:
            color = person_color(person)
            if len(person.joints):
                joints.append(np.asarray(person.joints, dtype=np.float32))
                labels.extend(f"#{person.id} {name}" for name in person.names)
                colors.extend([color] * len(person.joints))
            for strip in person.strips:
                strips.append(np.asarray(strip, dtype=np.float32))
                strip_colors.append(color)
            if len(person.trail) > 1:
                trails.append(np.asarray(person.trail, dtype=np.float32))
                trail_colors.append(color)
            if np.isfinite(person.position).all():
                positions.append(np.asarray(person.position, dtype=np.float32))
                alert = fall_alert(person)
                position_labels.append(
                    f"#{person.id} {person.distance:.2f} m · {person.speed_ground:.2f} m/s"
                    if np.isfinite(person.speed_ground)
                    else f"#{person.id} {person.distance:.2f} m"
                )
                if alert:
                    position_labels[-1] = f"{position_labels[-1]} · {alert}"
                position_colors.append(color)

        self._log(
            "camera/pose3d/joints",
            rr.Points3D(
                np.concatenate(joints) if joints else np.zeros((0, 3), dtype=np.float32),
                labels=labels,
                colors=colors or None,
                radii=JOINT_RADIUS,
            ),
        )
        self._log("camera/pose3d/skeleton", rr.LineStrips3D(strips, colors=strip_colors))
        self._log("camera/pose3d/trails", rr.LineStrips3D(trails, colors=trail_colors))
        self._log(
            "camera/pose3d/positions",
            rr.Points3D(
                np.stack(positions) if positions else np.zeros((0, 3), dtype=np.float32),
                labels=position_labels,
                colors=position_colors or None,
                radii=POSITION_RADIUS,
            ),
        )

        if self.world is None:
            return
        # 世界系视图：同一个刚体变换套一遍（点云在 log_points 里已经处理过）
        world_joints = [self.world.apply(points) for points in joints]
        world_strips = [self.world.apply(strip) for strip in strips]
        world_trails = [self.world.apply(trail) for trail in trails]
        world_positions = [self.world.apply(point) for point in positions]
        self._log(
            "world/joints",
            rr.Points3D(
                np.concatenate(world_joints)
                if world_joints
                else np.zeros((0, 3), dtype=np.float32),
                labels=labels,
                colors=colors or None,
                radii=JOINT_RADIUS,
            ),
        )
        self._log("world/skeleton", rr.LineStrips3D(world_strips, colors=strip_colors))
        self._log("world/trails", rr.LineStrips3D(world_trails, colors=trail_colors))
        self._log(
            "world/positions",
            rr.Points3D(
                np.stack(world_positions)
                if world_positions
                else np.zeros((0, 3), dtype=np.float32),
                labels=position_labels,
                colors=position_colors or None,
                radii=POSITION_RADIUS,
            ),
        )

    def _log_coords(self, entity: str, coordinates) -> None:
        """坐标系只发一次：不发的话 3D 视图可能倒着/转着显示。"""
        if entity in self._coords_logged:
            return
        self._coords_logged.add(entity)
        self._log(entity, coordinates)
