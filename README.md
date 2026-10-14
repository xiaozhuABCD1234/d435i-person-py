# d435i-person-py

Intel RealSense **D435i** → YOLO 姿态估计 → **真深度 3D 骨架** → 跟踪 ID/轨迹/世界坐标 →
[Rerun](https://rerun.io) 可视化。取流直接用 `pyrealsense2`。

**参数都在 [`config.toml`](config.toml) 里，代码里不写死。** 命令行只做临时覆盖。

## 和 hik-person-py 的关系

`~/Projects/hik-person-py` 是同一套骨架（config.toml + 命令行覆盖、YOLO 结构化输出、
rerun blueprint、卡尔曼 + IoU 跟踪、按需 `annotate()`），区别只在**深度从哪来**：

| | hik-person-py | 本项目 |
| --- | --- | --- |
| 相机 | HIKROBOT MVS 工业相机 | Intel RealSense D435i |
| 取流 | MVS SDK（C 接口 + 手写 Bayer 转换） | pyrealsense2（彩色 BGR8 + 对齐深度） |
| 3D 深度 | `yolo26s-depth` 单目估计（模型猜的，尺度会飘） | **对齐后的真深度**（公制、无模型开销） |
| 内参 | 按假定 FOV 猜 | 设备标定值（`fx/fy/cx/cy` 从 profile 读） |
| 附加能力 | — | 点云、IMU 可扩展、世界坐标只需静态外参 |
| 相机参数 | 曝光/增益/Bayer 位深/GenICam 特性 | 深度分辨率/发射器/激光功率/后处理滤镜 |

所以本项目里 `[pose3d]` 的卡尔曼观测噪声能比 hik 小一个量级（0.02 m vs 0.1 m）：
真深度不抖，不需要那么强的平滑。

## 快速开始

```bash
uv sync                                  # torch/cv2 命中 uv 缓存
uv run d435i-person --list               # 枚举 RealSense 设备
uv run d435i-person --show-config        # 打印合并后的生效参数
uv run d435i-person                      # 按 config.toml 跑（开 rerun 窗口）
uv run d435i-person --duration 30        # 临时覆盖某一项
uv run d435i-person --weights models/yolo26n-pose.pt --detect-hz 60   # n 模型可以真跑到 60 Hz
uv run d435i-person --coords-view --count-view                          # 坐标表 + 人数曲线都要
uv run d435i-person --count-view --no-coords-view                       # 只要人数曲线（旧布局）
uv run d435i-person --no-rerun --max-frames 300        # 只看结果，不开窗口不写文件
uv run d435i-person --world --world-height 1.35 --world-pitch 20   # 打开世界坐标系
uv run python tools/check_camera.py --save out/        # 不跑模型的相机自检
```

前置条件：D435i 插 **USB 3.x** 口、Intel udev 规则已装
（`sudo cp ~/Projects/d435i_ws/config/99-realsense-libusb.rules /etc/udev/rules.d/`，
本机已装；ROS deb **不带**这些规则）。`pyrealsense2` 的 manylinux wheel 自带一份
librealsense，与 `/opt/ros/jazzy` 里 ROS 那份互不干扰 —— 但两者**不能在同一个进程里
同时加载**（所以本包不订阅 ROS 话题）。

## 参数文件

优先级：**命令行 > `config.toml` > 代码里的兜底默认值**。用 `--config PATH` 换文件。

```toml
[camera]
serial = ""                   # 多相机时必须用 serial 指认；空 = 用 index 那台
color_size = "640x480"        # 深度会被对齐到彩色，对齐后深度分辨率 = 彩色分辨率
depth_size = "640x480"
fps = 30
exposure_mode = "sdk"         # sdk = 相机自动曝光/增益；manual = 固定用 exposure_us/gain
laser_power = -1.0            # 激光功率，-1 = 不动（关发射器会让无纹理面出现空洞）
emitter_enabled = -1          # 1 开 / 0 关，-1 = 不动
read_timeout_ms = 1000        # 60 fps 下帧间隔只有 17 ms，超时不用设得很宽

[filters]
spatial = false               # 深度后处理：默认全关，边缘更干净
temporal = false
hole_filling = false

[detector]
weights = "models/yolo26x-pose.pt"   # 只做 pose；相对路径按项目根解析（见下方"模型选择"）
conf = 0.35
imgsz = 640
kpt_conf = 0.5                # 关键点置信度阈值（2D/3D 骨架、有效点计数都受影响）

[run]
# 按**模型实测能力**设，不是按相机 fps：x 模型 64.5 ms/帧 -> 15 Hz
# （相机仍是 60 fps 取流，中间帧丢掉不推理；用 m/n 模型就拉到 30/60）
detect_hz = 15.0
save_dir = ""                 # 检测到人时存标注图

[output]
jsonl = ""                    # 非空则把人写成 JSON Lines（追加）
hz = 5.0

[track]
iou_threshold = 0.3           # 跨帧关联；越高越容易换 id，越低越容易并成一个人
match_max_distance = 0.0      # >0 时配对还要满足 3D 距离（米），防交叉换 id
trail_length = 300            # 轨迹点数上限（0 = 不留轨迹）
trail_interval = 0.1          # 轨迹采样间隔（秒）

[points]
enabled = true                # 彩色点云
stride = 4                    # 640×480 stride 4 ≈ 1.9 万点
max_distance = 4.0            # 只保留这个距离内的点（米）

[world]
enabled = false               # 需要知道安装高度/俯仰角才准
height = 1.0                  # 光心离地高度（米）
pitch_deg = 0.0               # 镜头下俯角度（正 = 向下）
```

## 数据流

```
Camera.read()                 彩色 BGR8 + align(depth -> color) 的 z16
  │                           depth_scale 从设备读（D435i = 0.001）→ float32 米，0 → nan
  ├─ 2D: PoseEstimator  YOLO pose → 17 个关键点 + 框（不做 result.plot()）
  │        └─ viz.log_frame      原图 + 结构化 Boxes2D/Points2D/LineStrips2D
  └─ 3D: PersonTracker.update()  每个人的 17 个关键点按 5×5 中值取深度 → 反投影成相机系 3D
           │                     位置：双髋中点 → 双肩中点 → 有效关节中位数 → 框中心深度
           ├─ 卡尔曼（每人每关键点 + 位置各一组恒定速度滤波）
           ├─ IoU 贪心关联 → 持久 id（单调递增、永不复用）+ 轨迹
           └─ viz.log_people3d   3D 骨架/轨迹/位置；（可选）世界系副本
```

坐标系：相机系用 **RDF**（X 右 / Y 下 / Z 前，= rerun 的 `RIGHT_HAND_Y_DOWN`），
世界系用 **FLU**（X 前 / Y 左 / Z 上，原点在相机正下方的地面点）。
D435i 的深度是沿光轴的 Z（不是斜距），所以针孔反投影直接成立：

```
X = (u - cx) * Z / fx     Y = (v - cy) * Z / fy     Z = depth
```

关键点取深度用的是 **5×5 窗口中值**，不是单像素：关节正好压在人体/背景边界上时，
单像素会被背景抢走，中值稳得多。窗口内没有有效值就是 nan，该关键点这一帧不算数
（`valid_count` 会掉），骨头只在两端都有效时才画。

## 模型选择与 60 fps

**先分清两件事：相机的 60 fps 只是取流上限，和推理帧率无关。** 配置里分开控制：
`[camera].fps = 60` 决定相机多久给一帧（日志里的"X 帧跳推理"就是被丢掉的那些），
`[run].detect_hz` 决定多久跑一次模型。所以"相机 60 fps + 推理 15 Hz"是完全正常的
组合，好处是时间戳更密、运动更连贯。

三者实测（RTX 4060 Laptop、输入 640×480、`imgsz 640`、20 帧热身后计 200 帧；
可视化开销 = JPEG 编码 + 深度/点云/坐标记录，也实测约 3.8 ms/帧）：

| 权重 | 大小 | YOLO pose | 全链路（含跟踪/点云/可视化） | 60 Hz 推理？ |
| --- | --- | --- | --- | --- |
| `yolo26n-pose.pt` | 7.9 MB | 4.4 ms | ~9.6 ms → ~104 fps | ✅ 宽松 |
| `yolo26m-pose.pt` | 49 MB | 9.7 ms | ~14.9 ms → ~67 fps | ✅ **能跑满 60 fps** |
| `yolo26x-pose.pt`（当前） | 126 MB | **64.5 ms** | **~70 ms → ~14 Hz** | ❌ 差 4 倍 |
| `yolo26x-pose.pt` @ `imgsz 512` | — | 39.0 ms | ~45 ms → ~22 Hz | ❌ |

结论：**x 模型 + 60 Hz 推理在这台机器上做不到**（60 Hz 的预算是 16.7 ms/帧，x 光
模型就要 64.5 ms）。想要真 60 fps 就用 m；可行的四种配法：

```bash
# A）要 x 的精度：相机 60 fps、推理 ~14 Hz（当前 config.toml，detect_hz 15）
uv run d435i-person

# B）要真 60 Hz 端到端：换 m
uv run d435i-person --weights models/yolo26m-pose.pt --detect-hz 60

# C）要 60 Hz 且余量充足：换 n
uv run d435i-person --weights models/yolo26n-pose.pt --detect-hz 60

# D）x 降分辨率换频率（约 22 Hz）
uv run d435i-person --imgsz 512 --detect-hz 22
```

想再往上走只有导出加速这条路（`model.export(format='engine', half=True)` 走 TensorRT，
需要额外装 tensorrt；ONNX Runtime 也能快 1.3–2×），代价是要多一套推理后端。
官方预训练都在 `ultralytics/assets` 的 release 里，`YOLO("yolo26l-pose.pt")`
会自动下载到当前目录：

```bash
cd models && uv run python -c "from ultralytics import YOLO; YOLO('yolo26l-pose.pt')"
```

## 世界坐标怎么用

人相对相机的坐标已经够多数场景用；要"人在地面上的位置"（比如多相机拼接、给底盘发人
的位置）就打开 `[world]`，只需要三个数：

1. `height`：光心离地多高。卷尺量，或把人放到已知距离看深度读数校准。
2. `pitch_deg`：镜头往下俯了几度。手机水平仪 App 贴在相机外壳上量，或者让一个已知
   身高的人站远一点，看 3D 骨架的头顶高度反推。
3. `yaw_deg`/`roll_deg`：相机朝向相对世界 X / 光轴不水平时才需要，默认 0。

外参不准只会让**绝对位置**整体偏，不影响骨架形状与相对距离。`--world-height` /
`--world-pitch` 可以临时覆盖来试。世界系实体记在 rerun 的 `world/` 下，
JSON Lines 里每人也多一个 `world` 字段。

## 跟踪与 ID

* 配对：前后帧检测框 IoU 贪心配对（`iou_threshold`），可选再加 3D 距离门限
  （`match_max_distance`，>0 时生效）——两个人交错而过时，光靠 IoU 容易换 id，
  加上"位置不能跳太远"就稳了。
* 漏检：轨迹保留 `max_missed` 帧（15 Hz 下 5 帧 ≈ 0.33 s），期间位置由卡尔曼预测，
  所以短时间内被遮挡的人不会丢 id，也不会在曲线上被数两遍。
* 单个关键点连续丢 `max_missed_kpt` 帧后不再画，人的其他部位继续画。
* id 单调递增、**永不复用**，可以直接当人的编号写进日志。

## 可视化（rerun）

| 实体 | 内容 |
| --- | --- |
| `camera/image` | 彩色原图（按 `jpeg_quality` 压过再送） |
| `camera/image/boxes` | 框（标签 = `#id conf`，颜色跟着 id） |
| `camera/image/keypoints` `skeleton` | 2D 关键点与骨架（同上色） |
| `camera/depth` | 对齐深度（Turbo 伪彩，悬停看米数） |
| `camera/pose3d` | 相机系 3D 视图根（`RIGHT_HAND_Y_DOWN`） |
| `camera/pose3d/points` | 彩色点云 |
| `camera/pose3d/joints` `skeleton` | 真深度 3D 骨架 |
| `camera/pose3d/trails` `positions` | 轨迹、每人 3D 位置（标签带距离） |
| `world/...` | 世界系视图（`[world].enabled` 时才记） |
| `person/table` | **人体坐标表**（位置表 + 髀/肩/踝/鼻，相机系与世界系） |
| `person/coords/<id>/{x,y,z}` | 每个人的坐标曲线（相机系，TimeSeriesView） |
| `person/count` | 人数标量（默认不再占面板，需要就把 `[rerun].show_count` 打开） |
| `person/events` | 人数变化日志 |

布局：左边彩色（下面对齐深度），右边 **人体坐标**（上：坐标表，下：X/Y/Z 曲线）
+ 3D 骨架/点云（+ 世界系视图）。每个人的颜色由 id 决定，2D 和 3D 里同一个 id 同色，
一眼看得出谁是谁。

坐标面板用 `[rerun].show_coords` / `show_count` 控制：两个都开也行，都关就只剩图像和 3D。
曲线是**相机坐标系**（X 右 / Y 下 / Z 前）；开了 `[world]` 后曲线仍用相机系（保持坐标轴
含义固定），世界坐标在表里看；想把曲线也换成世界系，把 `viz.py` 里
`person/coords` 改成 `person/world_coords` 即可（两组都在 .rrd 里）。

`keep_history = false`（默认）时图像/点云/骨架都用 **static** 记录：viewer 只留最新
一份，内存不随帧数涨。想回拖时间轴就 `--rerun-keep-history`（内存会一直涨）。

## 输出

`[output].jsonl` 非空时，按 `[output].hz` 追加写每帧的人：

```json
{"t": 1234.567, "frame": 8123, "device_timestamp_ms": 45678.9, "count": 2,
 "people": [{"id": 3, "conf": 0.87, "distance": 2.31, "cam": [0.12, -0.05, 2.31],
             "world": [0.31, -0.12, 1.10],
             "landmarks": {"hip": [0.12, -0.05, 2.31], "shoulder": [0.10, -0.38, 2.31],
                           "ankle": [0.14, 0.42, 2.32], "nose": [0.09, -0.55, 2.30]},
             "box": [301.2, 120.4, 360.8, 470.1],
             "valid_keypoints": 14, "age": 96}]}
```

`world` 只在 `[world].enabled` 时非 null；`landmarks` 里缺哪个关节（被遮挡、深度没测到）
就少哪个键，`cam`/世界坐标都是米制的相机坐标系（X 右 / Y 下 / Z 前）。
`--save-dir` 另存标注图（按需 `annotate()`，不是每帧都画）。

## 开发

```bash
uv run ruff check .                      # lint
uv run ruff format --check .             # 格式
uv run pytest                            # 单测（不需要相机、不加载权重）
uv run python tools/check_camera.py      # 相机自检（设备/内参/深度质量/帧率）
uv run python tools/bench.py             # 分阶段计时：取流/YOLO/跟踪/点云
```

单测全部是纯函数或无相机集成（`test_viz.py` 写 .rrd 到临时目录），所以换机器/没插
相机也能 CI。`test_camera.py` 里有一条用子进程验证 **pyrealsense2 是延迟导入**的用例，
这是"没相机也能跑测试与 lint"的前提。

## 结构

```
d435i-person-py/
├── config.toml                 全部参数（唯一的"旋钮面板"）
├── models/                     yolo26n/m/x-pose.pt（默认用 x，见"模型选择与 60 fps"）
├── src/d435i_person/
│   ├── __main__.py             CLI + 主循环（限速、点云/JSONL 的频率、存图）
│   ├── config.py               参数读取/校验/命令行覆盖
│   ├── camera.py               pyrealsense2 取流：对齐深度、内参、depth scale、AE 收敛
│   ├── geometry.py             针孔模型：内参、深度换算、反投影、点云（纯函数）
│   ├── pose.py                 YOLO pose 包装（骨架常量 0-based、bones/visible）
│   ├── pose3d.py               2D 关键点 + 真深度 → 3D 骨架 + 每人位置
│   ├── track.py                卡尔曼 + IoU 关联 + 持久 id + 轨迹
│   ├── world.py                相机系 -> 世界系（地面系）静态外参
│   ├── viz.py                  rerun 记录与 blueprint
│   └── colors.py               BGR/RGB 转换、格式名
├── tests/                      7 个测试文件，全部无需相机
└── tools/                      check_camera.py（相机自检）、bench.py（分阶段计时）
```

## 性能

实测（RTX 4060 Laptop、640×480、`imgsz 640`、`tools/bench.py --no-camera --frames 220`，
20 帧热身）：

| 阶段 | n | m | x |
| --- | --- | --- | --- |
| YOLO pose | 4.4 ms | 9.7 ms | **64.5 ms** |
| 可视化记录（JPEG 编码 + 深度/点云/骨架/坐标） | 3.8 ms | 3.8 ms | 3.8 ms |
| 跟踪 + 3D（1 人；17 点 5×5 中值取深度 + 51 个卡尔曼） | 0.95 ms | 0.95 ms | 1.4 ms |
| 点云（stride 4，1.92 万点） | 0.43 ms | 0.43 ms | 0.6 ms |
| 全链路（1 人） | ~9.6 ms | ~14.9 ms | ~70 ms |
| 能达到的端到端上限 | ~104 fps | ~67 fps | **~14 Hz** |

两点值得记住：

* **首帧很贵**：CUDA 初始化与 cuDNN 选算法都落在第一帧（x 上实测约 1 s），所以
  "只跑 10 帧"算出来的均值会翻倍；`tools/bench.py` 默认先热身 20 帧就是这个原因。
* **相机不是瓶颈**：640×480 双流 @60 fps 约 92 MB/s，USB3.2 富裕；取流+对齐几乎不占
  CPU（拷 0.92 MB）。换更大模型时先看的是模型耗时，不是相机。

自己复现/换参数：

```bash
uv run python tools/bench.py --frames 200                    # 分阶段（含相机）
uv run python tools/bench.py --no-camera --weights models/yolo26n-pose.pt   # 不占相机比模型
uv run python tools/check_camera.py --frames 60              # 只看相机侧帧率与深度质量
```

自己复现/换参数：

```bash
uv run python tools/bench.py --frames 200                    # 分阶段（含相机）
uv run python tools/bench.py --no-camera --weights models/yolo26n-pose.pt   # 不占相机比模型
uv run python tools/check_camera.py --frames 60              # 只看相机侧帧率与深度质量
```

## 故障排查

| 现象 | 原因 / 处理 |
| --- | --- |
| `没找到 RealSense 设备` | 插 USB3 口；udev 规则；`rs-enumerate-devices -s`（ROS 装的那份工具）验证 |
| 普通用户 `permission denied` / 打不开深度 | 缺 udev 规则：`sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/ && sudo udevadm trigger`，然后拔插相机 |
| 帧率只有个位数 / 深度分辨率被砍 | 插在 USB2 口或线太差；`--list` 看 `usb=` 字段必须是 3.2 |
| `RuntimeError: Couldn't resolve requests` | 该设备/固件不支持这组 分辨率+格式+帧率，用 `rs-enumerate-devices -c` 查（D435i 深度 640x480@90 支持，1280x720 只有 @30/15/6） |
| 有效深度很低（<50%）、骨架缺关节 | 无纹理/反光/太近太远；先开发射器（`--emitter 1`）、把 `laser_power` 调高，或试 `--spatial --temporal`；D435i 最准区间约 0.3–3 m |
| 相机设了 60 fps，但日志只有 15 fps 端到端 | 正常：`detect_hz` 是按模型能力设的，x 模型 64.5 ms/帧。想要 60 Hz 推理换 n（`--weights models/yolo26n-pose.pt --detect-hz 60`）；日志里的"X 帧跳推理"就是为它丢的帧 |
| 启动报 `Couldn't resolve requests`（60 fps 时） | 某些固件/机型下彩色 60 fps 只支持特定格式：本机 640x480 的 RGB8/BGR8/YUYV 都支持 @60，1280x720 只有 @30/15/6 |
| 右上角看不到坐标表 | 需要 `[rerun].show_coords = true`（默认已开）；表格在窗口偏窄时需要拖分隔条或拉宽窗口 |
| 3D 骨架"呼吸"/抖 | 提高 `[pose3d].kalman_measurement`（更平滑），或开 `--temporal` 深度滤镜 |
| 两个人换 id | 调大 `[track].iou_threshold`，或设 `match_max_distance = 1.0`（米） |
| id 越用越多、同一个人被数两遍 | `[track].max_missed` 调大（容忍更久的遮挡），或 `detect_hz` 调高 |
| 点云发红/颜色错乱 | 颜色是 BGR→RGB 换过的；如果画面红蓝互换说明有人在链路里又换了一次 |
| 3D 视图"自动缩放" | `[pose3d].eye/look_at/eye_up` 固定视角（默认已固定）；点云远近变化大时也会觉得在缩放 |
| 世界坐标全是画面的同一侧 / 高度不对 | `[world]` 的 `height`/`pitch_deg` 没量准；先 `--no-world` 用相机系确认 3D 骨架本身是对的 |
| viewer 内存一直涨 | 关 `--rerun-keep-history`，或 `--no-points`、调大 `[points].stride`、调低 `--rerun-jpeg-quality` |
| `IMU Calibration is not available` / `MIRI error` 之类日志 | 本包不用 IMU，这些是驱动噪音，无害 |

## 许可证

AGPL-3.0-or-later（与 `hik-person-py` 一致）。`pyrealsense2` / librealsense 遵循其
自带 Apache-2.0。
