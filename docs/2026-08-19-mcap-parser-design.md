# MCAP 解析器设计文档 (Spec)

- **项目**: `mcap-parser`
- **位置**: `~/work/hvyogo/ai_work/mcap-parser/`（与 `openembodied-data` 同级的新仓库）
- **目标**: 解析 PICO/OpenElephant 产出的 foxglove MCAP 文件（典型为 `head_pose_mcap.mcap`），把其中各数据流回环导出为独立文件，文件格式与 `openembodied-data` 原始产物对齐。
- **日期**: 2026-08-19
- **状态**: 设计待评审

---

## 1. 背景与动机

`openembodied-data` 管线在 `head_pose_mcap.mcap` 里把视频、手部/身体位姿、手部网格、头部轨迹、点云、相机标定等 14 个 foxglove（protobuf）通道打包成单文件。下游有时需要这些数据的**独立文件**（喂训练、做分析、跨工具链使用），而不是只能在 Foxglove Studio 里播放。本解析器是该管线的逆操作：从单个 MCAP 还原出原始文件集。

解析器是 **channel/topic 驱动** 的：不写死文件名，而是按 topic 前缀匹配通道 → 分配 writer。因此同一套解析器也适用于同类 foxglove MCAP（`camera_trajectory_mcap.mcap`、`head_pose_point_cloud.mcap`、`head_mcap_conversion.mcap`、`hand_algpose_vis.mcap` 等），只是哪些 writer 实际产出文件取决于该 MCAP 里有哪些通道。

## 2. 目标 MCAP 的通道结构（实测）

以 `data/SN_PA9410MGK8110131G_1527709884320059392_2026_07_17-16_13_39_963/head_pose_mcap.mcap` 为基准（537 MB，约 102 s，3066 帧 @ ~30 fps，时间戳为纳秒级 epoch）：

| ch_id | topic | schema | 帧数 | 内容 |
|---|---|---|---|---|
| 1 | `/pose_algo_body/tf_tree` | foxglove.FrameTransforms | 3066 | world→相机/腕/各关节的 4×4 变换（身体+相机位姿逐帧） |
| 2 | `/pose_algo_body/hand_keypoints` | foxglove.SceneUpdate | 3066 | 3D 手部骨架（spheres，world 系，21 关节，left+right） |
| 3 | `/pose_algo_body/hand_keypoints_2d` | foxglove.SceneUpdate | 3066 | 2D 手部关键点（left+right） |
| 4 | `/pose_algo_body/head_pose_trajectory` | foxglove.SceneUpdate | 3066 | 头部轨迹点（每帧一个球，累积） |
| 5 | `/pose_algo_body/hand_mesh` | foxglove.SceneUpdate | 3066 | MANO 三角网格（778 顶点、1538 面，left+right） |
| 6 | `/pcd_dir/pointcloud` | foxglove.PointCloud | 511 | RGB 点云（xyz+rgba，point_stride=16） |
| 7 | `/pcd_dir/pointcloud/static_scene` | foxglove.PointCloud | 1 | 静态场景点云 |
| 8 | `/pcd_dir/pointcloud/2d_projection` | foxglove.PointCloud | 3066 | 空占位（data_len=0，无字段）→ **跳过** |
| 9 | `/head_left_camera_undistorted/.../image` | foxglove.CompressedVideo | 3066 | H.264（x264，Annex-B NAL，keyint=36，bframes=0） |
| 10 | `/head_right_camera_undistorted/.../image` | foxglove.CameraCalibration | 3066 | H.264 右目 |
| 11 | `/head_left_camera_params/head_left_camera_undistorted/camera_info` | foxglove.CameraCalibration | 3066 | plumb_bob，左目 undistorted |
| 12 | `/head_left_camera_params/head_left_camera_raw/camera_info` | foxglove.CameraCalibration | 3066 | 左目 raw |
| 13 | `/head_right_camera_params/head_right_camera_undistorted/camera_info` | foxglove.CameraCalibration | 3066 | 右目 undistorted |
| 14 | `/head_right_camera_params/head_right_camera_raw/camera_info` | foxglove.CameraCalibration | 3066 | 右目 raw |
| 15 | `/head_depth_camera/depth/image` | foxglove.RawImage | 509 | 深度图，320×240，encoding `mono16`（2 B/px） |
| 16 | `/head_depth_camera_params/head_depth_camera/camera_info` | foxglove.CameraCalibration | 509 | 深度相机内参（每帧相同） |

解码依赖 `foxglove_schemas_protobuf`（已安装），类映射：

- `foxglove.FrameTransforms` → `foxglove_schemas_protobuf.FrameTransforms_pb2.FrameTransforms`
- `foxglove.SceneUpdate` → `foxglove_schemas_protobuf.SceneUpdate_pb2.SceneUpdate`
- `foxglove.PointCloud` → `foxglove_schemas_protobuf.PointCloud_pb2.PointCloud`
- `foxglove.CompressedVideo` → `foxglove_schemas_protobuf.CompressedVideo_pb2.CompressedVideo`
- `foxglove.CameraCalibration` → `foxglove_schemas_protobuf.CameraCalibration_pb2.CameraCalibration`
- `foxglove.RawImage` → `foxglove_schemas_protobuf.RawImage_pb2.RawImage`

## 3. 回环输出契约（MCAP 通道 → 文件）

输出目录结构与 `openembodied-data` 原始产物对齐。所有数组按**帧序**（按 `publish_time` 升序）排列。

| 源通道 | 输出文件 | 格式 / 形状 |
|---|---|---|
| ch9 | `head_left_camera_undistorted.mp4` | H.264 重封装为 MP4（`-c copy`），fps 由时间戳算 |
| ch10 | `head_right_camera_undistorted.mp4` | 同上 |
| ch11–14,16 | `camera_calibration.json` | `{left,right}_{raw,undistorted} + head_depth_camera: {width,height,distortion_model,K[9],D[5],R[9],P[12],frame_id}` |
| ch1 | `tracking_data_tf.jsonl` | 每帧一行：`{ts_ns, transforms:[{parent,child,translation[3],rotation[4]}]}`（全部 44 个 frame：2 相机 + 2 腕 + 40 指节，忠实全量） |
| ch1 | `wrist_poses.npz` | 键 `left`(N,7)、`right`(N,7)、`timestamps`(N,) = 腕根位姿 `[x,y,z,qx,qy,qz,qw]`（手部位姿的根） |
| ch2 | `left_keypoints_3d.npy` / `right_keypoints_3d.npy` | `(N,21,3)` float32 |
| ch3 | `left_keypoints_2d.npy` / `right_keypoints_2d.npy` | `(N,21,2)` float32 |
| ch4 + ch1（头部变换） | `head_pose_trajectory.npy` | `(N,7)` = `[x,y,z,qx,qy,qz,qw]` |
| ch5 | `left_hand_vertices.npz` / `right_hand_vertices.npz` | 键：`vertices`(N,778,3)、`faces`(1538,3)、`is_right`(N,)、`detected`(N,) |
| ch6 | `aligned_pointclouds.npz` | 见 §3.1 |
| ch6 | `pointclouds/frame_{i:06d}.pcd` | 每帧一个 PCD（ASCII 或 binary，含 rgb 字段） |
| ch7 | `static_scene.ply` | PLY（xyz+rgb） |
| ch7 | `aligned_pointclouds.npz`（补键 `pc_static_*`） | 见 §3.1 |
| ch15 | `depth/frame_{i:06d}.bin` | mono16 裸字节流（320×240，153600 B/帧），原样写出 `RawImage.data`，不转码、不改端序 |
| ch8 | — | 空通道，跳过 |
| 全部 | `manifest.json` | 见 §4 |

> **帧数说明**：原 `openembodied-data` 的 `left_keypoints_3d.npy` 是 6057 帧（更高频率的 WiLoR 检测），但 MCAP 只存了 3066 帧对齐子集。本解析器产出的是 **MCAP 实际帧数**（3066），无法恢复被丢弃的帧。这是预期行为，manifest 会显式记录每流的帧数。

> **不可恢复说明（重要）**：原 `body_keypoints_labels.npz` 的 24 关节 SMPL 身体骨架（Pelvis/Spine/Shoulder/Elbow/…）**不在 `head_pose_mcap.mcap` 的 tf_tree 里**。实测 tf_tree 每帧只有 44 个 frame：`head_left_camera`、`head_right_camera`、`left_wrist`、`right_wrist` + 40 个指节（thumb cmc/mcp/ip/tip、index/middle/ring/pinky 各 mcp/pip/dip/tip）。24-SMPL 身体关键点来自 `openembodied-data` 的 `body_pose_prelabel` 上游任务，本 MCAP 无法还原。故 body_tf 只产出 `tracking_data_tf.jsonl`（忠实全量 44 frame）+ `wrist_poses.npz`（腕根 7 元组），**不伪造 24 关节数组，也不做 body 2D 投影**（2D 投影改由手部通道 ch3 直接给出，见 §5.3）。

### 3.1 `aligned_pointclouds.npz` 键

对齐原 `openembodied-data` 的 `aligned_pointclouds.npz` 键，能从 MCAP 恢复的子集：

- `frame_timestamps`(N,)、`frame_timestamps_ns`(N,) —— 来自 ch1/ch4 每帧 `publish_time`
- `head_poses`(N,7) —— 来自 ch1 head_left_camera 变换
- `left_hand_positions`(N,21,3)、`right_hand_positions`(N,21,3) —— 来自 ch2（21 关节 3D）
- `left_hand_poses`(N,7)、`right_hand_poses`(N,7) —— 来自 ch1 的 left_wrist/right_wrist 变换（手根 7 元组）
- `hand_flags`(N,2) —— 每只手是否检出（ch2 有对应实体则 1，否则 0）
- `pc_static_points`(M,3)、`pc_static_colors`(M,3) —— 来自 ch7
- `pc_frame_indices`(511,)、`pc_frame_timestamps`(511,)、`pc_frame_point_counts`(511,)、`pc_frame_data`(511,) —— 来自 ch6

> **不可恢复**：`body_joint_positions`(N,24,3)、`body_joint_quaternions`(N,24,4)、`body_joint_names`(24,)、`left/right_hand_quaternions`(21 关节四元数) **不在本 MCAP**（tf_tree 无 24-SMPL 身体骨架，ch2 只给位置不给四元数）。aligned_pointclouds.npz 这些键省略，manifest 标 `missing_keys`。

## 4. `manifest.json` 结构

```json
{
  "source_mcap": "…/head_pose_mcap.mcap",
  "source_size_bytes": 537301445,
  "message_count": 37304,
  "start_time_ns": 1784276020183000064,
  "end_time_ns": 1784276122354642152,
  "fps": 30.0,
  "frame_count": 3066,
  "outputs": {
    "head_left_camera_undistorted.mp4": {"channel": 9, "frames": 3066, "timestamps_ns": [...] },
    "head_right_camera_undistorted.mp4": {"channel": 10, "frames": 3066, "timestamps_ns": [...] },
    "camera_calibration.json": {"channels": [11,12,13,14]},
    "left_keypoints_3d.npy": {"channel": 2, "frames": 3066, "timestamps_ns": [...] },
    "…": {}
  },
  "topic_to_file": { "/pose_algo_body/hand_keypoints": ["left_keypoints_3d.npy","right_keypoints_3d.npy"], "…": [] },
  "missing_keys": {
    "aligned_pointclouds.npz": ["body_joint_positions","body_joint_quaternions","body_joint_names","left_hand_quaternions","right_hand_quaternions"]
  },
  "not_recoverable": ["24-SMPL body skeleton (from body_pose_prelabel upstream)", "body_keypoints_2d (24 joints)"]
}
```

`timestamps_ns` 数组供下游按时间戳对齐做随机访问。

## 5. 架构与模块

### 5.1 仓库布局

```
mcap-parser/
  pyproject.toml
  README.md
  src/mcap_parser/
    __init__.py
    reader.py            # McapFile：单遍扫描 + 分发
    decoders.py          # foxglove protobuf 解码助手 + WRIST_FRAME_IDS 常量
    manifest.py          # 写 manifest.json
    cli.py               # python -m mcap_parser <mcap> -o <dir> [选项]
    writers/
      __init__.py
      base.py            # Writer 基类：owns_topic(t)->bool, feed(msg), finalize(out_dir)
      video.py
      hand_keypoints.py
      hand_mesh.py
      head_trajectory.py
      pointcloud.py
      body_tf.py
      camera_calib.py
  tests/
    conftest.py          # 合成 foxglove MCAP fixture（含全部 schema）
    test_writers.py
    test_integration.py
```

### 5.2 核心流程（单遍流式）

```python
class McapFile:
    def __init__(self, path): ...
    def parse(self, out_dir, *, side="both", streams=None):
        summary = self._reader.get_summary()
        writers = build_writers(summary, side=side, streams=streams)  # 按 topic 前缀匹配
        for schema, channel, msg in self._reader.iter_messages():
            for w in writers:
                if w.owns_topic(channel.topic):
                    w.feed(channel, schema, msg)
        for w in writers:
            w.finalize(out_dir)
        ManifestBuilder(self).write(out_dir, writers)
```

- **分发策略**：每条消息按 channel topic 前缀匹配给归属 writer。一个 writer 可拥有多个 topic（如 `hand_keypoints` 同时拥有 ch2/ch3，`body_tf` 拥有 ch1）。
- **跨流依赖**：`head_pose_trajectory.npy` 的 7 列需组合 ch4 轨迹点 + ch1 head_left_camera 四元数；`aligned_pointclouds.npz` 需组合 ch6/ch7 + ch1/ch2。为保持单遍，writer 在 `feed` 阶段缓存所需的中间数据，在 `finalize` 阶段拼装。reader 在所有 writer `finalize` 前会先做一次**二次扫描**把跨 writer 共享的 ch1 帧变换缓存喂给需要的 writer（见 §5.2.1）。
- **内存**：手部关键点（3066×21×3×2 ≈ 1.5 MB）、网格顶点（3066×778×3×2 ≈ 28 MB）常驻内存可接受；点云 ch6 逐帧约 300 MB 总量，`feed` 阶段逐帧 append 到对象数组，finalize 写 NPZ + 逐帧 PCD；视频 NAL 流式 append 到 `.h264` 临时文件，不常驻。

### 5.2.1 跨流共享缓存

`HeadTrajectoryWriter`、`PointcloudWriter`、`BodyTfWriter` 都需要 ch1 的逐帧 head/wrist 变换。为避免各 writer 独立缓冲全部 ch1 数据，`McapFile` 持有一个 `_tf_cache: list[dict[child_frame_id, (t, q, ts)]]`（按帧），在 `feed` 阶段由 `BodyTfWriter` 写入（body_tf 拥有 ch1），其他 writer 通过 `reader.tf_cache` 只读访问。head_left_camera 变换同样进入该缓存供 head_trajectory / pointcloud 用。

### 5.3 各 writer 细节

**VideoWriter** (`video.py`)
- 拥有 topic 前缀 `/head_{left,right}_camera_undistorted/`。
- `feed`：`CompressedVideo.data`（原始 Annex-B NAL）追加到 `out_dir/.tmp/<side>.h264`。
- `finalize`：`fps = (last_ts - first_ts) / (frames-1) / 1e9`；`ffmpeg -y -i <side>.h264 -c copy -r <fps> -f mp4 <side>.mp4`；失败兜底 fps=30；删临时文件。`--side` 过滤左右目。
- 失败语义：ffmpeg 不可用时写 `<side>.h264` 并在 manifest 标 `muxed=false`，不抛异常。

**HandKeypointsWriter** (`hand_keypoints.py`)
- 拥有 `/pose_algo_body/hand_keypoints`（3D）与 `/pose_algo_body/hand_keypoints_2d`（2D）。
- `feed`：解码 SceneUpdate，按实体 `id`（`left_hand_skeleton`/`right_hand_skeleton`）取 21 个 sphere 的 position → 3D 通道用 `(x,y,z)`，2D 通道用 `(x,y)`（z 丢弃），append 到 per-side buffer，记录 timestamps_ns。2D 直接取自 ch3，**无需投影**。
- `finalize`：`np.save` 为 `left_keypoints_3d.npy` / `right_keypoints_3d.npy` / `left_keypoints_2d.npy` / `right_keypoints_2d.npy`，左右按实体 id 判定。无对应实体的帧用 `nan` 占位并记到 manifest 的 `missing_frames`。

**HandMeshWriter** (`hand_mesh.py`)
- 拥有 `/pose_algo_body/hand_mesh`。
- `feed`：实体 `left_hand_mesh`/`right_hand_mesh` 的 Triangles primitive；`points`→`(N,778,3)`（取第一帧后校验恒定 778），`indices`→`(1538,3)`（恒定，首帧取一次）。
- `finalize`：`np.savez_compressed` 为 `left_hand_vertices.npz`，键 `vertices`/`faces`/`is_right`/`detected`（`detected` 全 True，因为 MCAP 里都是检出帧）。

**HeadTrajectoryWriter** (`head_trajectory.py`)
- 拥有 `/pose_algo_body/head_pose_trajectory`，`feed` 阶段只缓存 ch4 每帧的轨迹球 position + timestamps。
- `finalize`：逐帧取 ch4 的 position 作 `[x,y,z]`，配对 `reader.tf_cache` 同帧 head_left_camera 变换的 rotation 作 `[qx,qy,qz,qw]` → `(N,7)` = `[x,y,z,qx,qy,qz,qw]`，存 `head_pose_trajectory.npy`。配对按帧序（tf_cache 与 ch4 同为逐帧、时间戳对齐）。

**PointcloudWriter** (`pointcloud.py`)
- 拥有 `/pcd_dir/pointcloud`（ch6）、`/pcd_dir/pointcloud/static_scene`（ch7）。
- `feed`：解码 PointCloud，按 `point_stride` + `fields`（x/y/z FLOAT32@0/4/8，red/green/blue/alpha UINT8@12/13/14/15）用 `numpy.frombuffer` 解析 `data` 为 `(P,3)` xyz + `(P,3)` rgb（归一化到 0–1）。
- ch6 逐帧：append 到对象数组 `pc_frame_data`、记 `pc_frame_point_counts`/`pc_frame_timestamps`/`pc_frame_indices`；同时写 `pointclouds/frame_{i:06d}.pcd`（binary，含 rgb）。
- ch7：缓存为 `pc_static_points`/`pc_static_colors`。
- `finalize`：拼 `aligned_pointclouds.npz`（§3.1 键，`head_poses`/`left/right_hand_poses`/`left/right_hand_positions`/`hand_flags`/`frame_timestamps*` 取自 `reader.tf_cache` 与 ch2 缓冲）；写 `static_scene.ply`（ASCII，含 rgb）。
- PCD 每帧可选关闭（`--no-per-frame-pcd`）以省 IO。

**BodyTfWriter** (`body_tf.py`)
- 拥有 `/pose_algo_body/tf_tree`（ch1）。
- `feed`：解码 FrameTransforms，逐帧把所有 transforms 写入 `reader.tf_cache[frame_index]`（dict: child_frame_id → (translation[3], rotation[4], ts_ns)）；同时追加写 `tracking_data_tf.jsonl` 一行 `{ts_ns, transforms:[{parent,child,translation,rotation}]}`。
- `finalize`：从 `reader.tf_cache` 抽 `left_wrist`/`right_wrist` 的 (t,q) → `wrist_poses.npz`（键 `left`(N,7)、`right`(N,7)、`timestamps`(N,)）。**不产出 24-SMPL body_keypoints**（tf_tree 无身体骨架，见 §3 不可恢复说明）。
- 常量 `WRIST_FRAME_IDS = ("left_wrist", "right_wrist")`、`HEAD_FRAME_ID = "head_left_camera"` 硬编码于 `decoders.py`。

**CameraCalibWriter** (`camera_calib.py`)
- 拥有 topic 前缀 `/head_{left,right}_camera_params/`。
- `feed`：解码 CameraCalibration，缓存到 `{side}_{variant}` dict（每帧内容相同，首帧即可）。
- `finalize`：写 `camera_calibration.json`，结构 §3。

> 注：原 spec 的 `project.py` 3D→2D 投影模块**已移除**——手部 2D 直接来自 ch3，身体 2D 不存在。手部 2D 关键点已是 MCAP 原生通道，无需投影。

## 6. CLI

```
python -m mcap_parser <mcap_path> -o <out_dir>
  [--side {left,right,both}]            # 视频与对应手部数据按侧过滤，默认 both
  [--streams video,hand_keypoints,hand_mesh,head_trajectory,pointcloud,body_tf,camera_calib]
  [--no-per-frame-pcd]                  # 跳过逐帧 PCD
  [--overwrite]                         # 默认要求 out_dir 不存在或为空
```

- 幂等：`--overwrite` 时先清空 out_dir。
- 进度：每 500 帧打一行日志（已处理消息数 / summary.message_count）。
- 退出码：成功 0；ffmpeg 缺失但其余成功 0（manifest 标注）；解析异常 1。

## 7. 错误处理

- **未知 schema**：跳过该通道，manifest 记 `unsupported_schema`。
- **帧数不一致**：某流帧数 != 主帧数（3066）时，manifest 各流独立记 `frames`；不强制对齐。
- **缺失通道**：某 writer 拥有的 topic 在 MCAP 里不存在 → 该 writer 不产出文件，manifest 不列该输出。
- **protobuf 解码失败**：记该消息 `decode_error` 到 stderr 日志，跳过，继续。
- **视频 remux 失败**：保留 `.h264`，manifest 标 `muxed=false`，不中断。

## 8. 测试策略

- **单测 `test_writers.py`**：合成一个极小 foxglove MCAP fixture（`conftest.py` 用 `mcap.writer.Writer` + foxglove protobuf 构造，每种 schema 3 帧），逐 writer 断言：
  - `hand_keypoints`：输出 `(3,21,3)`、`(3,21,2)`。
  - `hand_mesh`：`vertices (3,778,3)`、`faces (1538,3)`、键齐全。
  - `head_trajectory`：`(3,7)`，7 列。
  - `pointcloud`：NPZ 键齐全、PCD 文件数==帧数。
  - `body_tf`：`wrist_poses.npz` 含 `left (3,7)`、`right (3,7)`、`timestamps (3,)`；`tracking_data_tf.jsonl` 行数==帧数且每行 transforms 含 44 项。
  - `camera_calib`：JSON 4 组、K 长度 9。
- **集成 `test_integration.py`**：对真实 MCAP 截前 30 s（或前 900 帧）跑一次，断言 manifest 存在、视频 mp4 可被 `cv2.VideoCapture` 打开且帧数合理、几个 NPZ 形状符合 §3 契约。537 MB 全量不入 CI。

## 9. 依赖

- 运行：`mcap`、`numpy`、`opencv-python`、`foxglove-schemas-protobuf`；系统 `ffmpeg`（视频 remux，可选）。
- CLI：`fire`（或 `argparse`，倾向 `fire` 简洁）。
- 测试：`pytest`、`mcap`（写 fixture）。
- **不依赖** `openembodied-data`：`WRIST_FRAME_IDS`/`HEAD_FRAME_ID` 常量自包含。

## 10. 非目标（YAGNI）

- 不做可视化/回放（那是 Foxglove Studio 的事）。
- 不做随机访问库 API（导出文件即终态）。
- 不处理非 foxglove schema 的 MCAP（如 ROS2 CDR）。
- 不恢复 MCAP 里不存在的数据：6057 帧高频关键点、24-SMPL 身体骨架、body 2D 投影、21 关节四元数。
- 不做 3D→2D 投影（手部 2D 直接来自 ch3 通道）。
- 不上传 TOS / 不接 MQ —— 纯本地解析。

## 11. 验收标准

1. 对基准 `head_pose_mcap.mcap` 运行后，`out_dir` 产出 §3 表中全部文件，manifest 字段完整。
2. `head_left_camera_undistorted.mp4` / 右目可用 `cv2.VideoCapture` 打开，帧数 ≈ 3066。
3. `left_hand_vertices.npz` 键与形状与 `openembodied-data` 原文件一致（3066×778×3 + 1538×3）。
4. `head_pose_trajectory.npy` 形状 `(3066,7)`。
5. `wrist_poses.npz` 含 `left (3066,7)`、`right (3066,7)`、`timestamps (3066,)`；`tracking_data_tf.jsonl` 行数 3066、每行 44 个 transform。
6. `aligned_pointclouds.npz` 含 §3.1 全部可恢复键，`pc_frame_data` 长度 511；manifest `missing_keys` 含 24-SMPL body 键。
7. 逐帧 PCD 数量 == ch6 帧数（511）。
8. `camera_calibration.json` 含 4 组标定，K 长度 9。
9. 单测全绿；集成测试对截断 MCAP 通过。

## 12. 已决默认值（spec 评审时可推翻）

- 手部 2D 关键点**直接取自 ch3**（不投影）。
- 逐帧 PCD 默认 **binary**（紧凑、IO 小）；如需人读再考虑 ASCII 选项。
- CLI 默认用 **`fire`**（简洁、支持子命令扩展）；无强烈反对则定。
- `WRIST_FRAME_IDS = ("left_wrist","right_wrist")`、`HEAD_FRAME_ID = "head_left_camera"` 硬编码于 `decoders.py`，源自对基准 MCAP tf_tree 的实测。
