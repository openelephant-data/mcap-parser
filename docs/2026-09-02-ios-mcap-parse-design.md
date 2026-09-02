# iPhone MCAP 解析设计文档 (Spec)

- **项目**: `mcap-parser`
- **分支**: `iphone`
- **目标**: 解析 iPhone/oeego ARKit 产出的 foxglove MCAP(`data/head_pose_mcap.mcap`),按 `openelephant-data-app` 的 `IosMcapParseTask` 节点输出契约,导出 7 个对齐文件 + `parse_report.json`。
- **日期**: 2026-09-02
- **状态**: 设计已确认,待实现

---

## 1. 背景与动机

`openelephant-data-app` 有一个 `IosMcapParseTask` 节点,把 iPhone 采集的原始 MCAP demux 成 7 个独立产物(`video_head.mp4`、`depth_frames.npz`、`confidence_frames.npz`、`camera_info.json`、`transforms.npz`、`hand_keypoints_3d.npz`、`parse_report.json`),喂下游训练/分析。该节点面向的输入是 `oeego.sensor_proto.*` 自定义 schema 的 MCAP。

本仓库 `data/head_pose_mcap.mcap` 是同类 iPhone 采集,但 schema 改用了 **foxglove 标准协议**(混有少量 `oeego.sensor_proto.*`)。现有 `mcap-parser` 只处理 PICO 版 foxglove MCAP(双目 + 44-frame tf_tree),channel/topic 与 iPhone 版完全不重叠,无法直接复用。

本设计把 `IosMcapParseTask` 的输出契约**原样移植**到 `mcap-parser`,并适配本文件的 foxglove schema,使 `python -m mcap_parser data/head_pose_mcap.mcap -o <dir>` 一条命令产出与参考节点逐文件对齐的产物。

## 2. 目标 MCAP 的通道结构(实测)

基准文件 `data/head_pose_mcap.mcap`(758 MB,约 117 s)。各通道 schema、帧数、频率:

| ch_id | topic | schema | 帧数 | ≈Hz | 用途 |
|---|---|---|---|---|---|
| 1 | `/task_info` | foxglove.KeyValuePair | 1404 | 12 | 任务元信息 |
| 2 | `/cam_head_l/imu` | oeego.sensor_proto.Imu | 23456 | 200 | IMU(本契约不导出) |
| 3 | `/cam_head_l/ego_hand_pose` | foxglove.ImageAnnotations | 2240 | 20 | 2D 手部圆点(本契约不导出) |
| 4 | `/oeego/mission_manager/onboard_config` | oeego.system_proto.MissionManagerProto.OnboardConfig | 234 | 2 | mission_id + target_frame_rate |
| 5 | `/cam_head_l/color/arkit_session_config` | oeego.sensor_proto.ArkitCameraSessionConfig | 2 | — | (不导出) |
| 6 | `/cam_head_l/env_info` | foxglove.KeyValuePair | 70160 | 600 | (不导出) |
| 7 | `/cam_head_l/depth/pointcloud` | foxglove.PointCloud | 7014 | 60 | (不导出,深度点云) |
| 8 | `/cam_head_l/depth/image_raw` | foxglove.RawImage | 7014 | 60 | **→ depth_frames.npz** |
| 9 | `/cam_head_l/depth/confidence` | foxglove.RawImage | 7014 | 60 | **→ confidence_frames.npz** |
| 10 | `/cam_head_l/color/camera_info` | foxglove.CameraCalibration | 7016 | 60 | **→ camera_info.json** |
| 11 | `/cam_head_l/depth/camera_info` | foxglove.CameraCalibration | 7014 | 60 | **→ camera_info.json** |
| 12 | `/cam_head_l/depth/bbox_in_color` | foxglove.KeyValuePair | 35070 | 300 | (不导出) |
| 13 | `/cam_head_l/color/arkit_frame_metadata` | oeego.sensor_proto.ArkitCameraFrameMetadata | 7016 | 60 | (不导出) |
| 14 | `/tf` | foxglove.FrameTransforms | 7016 | 60 | **→ transforms.npz** |
| 15 | `/time_reference` | foxglove.KeyValuePair | 14032 | 120 | (不导出) |
| 16 | `/cam_head_l/ego_hand_pose/pointcloud_3d` | foxglove.PointCloud | 4397 | 40 | **→ hand_keypoints_3d.npz** |
| 17 | `/cam_head_l/audio/compressed_aac` | foxglove.CompressedAudio | 2338 | 18 | (不导出) |
| 18 | `/cam_head_l/arkit/light_estimate` | oeego.sensor_proto.ArkitLightEstimateFrame | 6980 | 60 | (不导出) |
| 19 | `/cam_head_l/ego_hands_presence` | foxglove.KeyValuePair | 6 | — | (不导出) |
| 20 | `/cam_head_l/color/image_compressed_h265` | foxglove.CompressedImage | 3508 | 30 | **→ video_head.mp4** |

关键实测结论:

- **ch20 颜色流** topic 名为 `image_compressed_h265`,但 `CompressedImage.format='jpeg'`,数据首两字节 `FF D8`(JPEG SOI)——**实际是 JPEG 逐帧序列**,不是 H.265 NAL 流。
- **ch14 `/tf`** 每帧恰好 1 个 transform:`world → head_left_camera_color_optical_frame`,即逐帧头部 6DoF 位姿。
- **ch8 深度图** 是 `foxglove.RawImage 8UC1 256×192`(非参考里的 `CompressedImage 32FC1`)。
- **ch16 手部 3D** 是 `foxglove.PointCloud`,xyz FLOAT32 stride=12,每帧 13-21 点(ragged,非固定 21)。
- `onboard_config.target_frame_rate = 30`,`mission_id = 1543953655491727360`。
- 时间戳域混用:视频/深度/tf 用 epoch 秒(1788156102),IMU/ARKit 元数据用设备 uptime 小秒(11185)。各产物的时间戳字段按下文各节规定。

## 3. 输出契约(MCAP 通道 → 文件)

输出目录扁平,与 `openelephant-data-app` 参考节点逐文件对齐。所有数组按 `iter_messages` 顺序(等价于按 log_time 升序)。

| 文件 | 源通道 | 内容 / 形状 |
|---|---|---|
| `video_head.mp4` | ch20 | JPEG 序列重编码为 H.264 mp4(`-framerate 30 -i frame_%06d.jpg -c:v libx264 -bf 0 -g 30 -preset fast -pix_fmt yuv420p -an`)。fps=30 取自 onboard_config.target_frame_rate |
| `depth_frames.npz` | ch8 | `timestamps`(N,)int64 = log_time ns;`frames`(N,)object = RawImage.data 原始 bytes;`formats`(N,)object = `"<encoding>;<W>x<H>"`(`"8UC1;256x192"`) |
| `confidence_frames.npz` | ch9 | 同上,`formats`=`"8UC1;256x192"` |
| `camera_info.json` | ch10 + ch11 | 按 `frame_id` 键:`{frame_id,width,height,distortion_model,D,K[9],R[9],P[12]}`(首帧即可,每帧内容相同) |
| `transforms.npz` | ch14 | `transforms`(N,8)float64 = `[tx,ty,tz,qx,qy,qz,qw,timestamp_ns]`,ts = transform.timestamp.seconds×1e9+nanos |
| `hand_keypoints_3d.npz` | ch16 | `landmarks`(N,)object of (K,3)float32(K=13-21 ragged);`timestamps`(N,)int64 = PointCloud.timestamp.seconds×1e9+nanos |
| `parse_report.json` | 全部 | 见 §4 |

**不产出**(不在参考契约内):`hand_keypoints_2d`(ch3 circles 非 EgoHandPoseFrame 2D,跳过;`hand_2d_count=0`)、逐帧 pcd/png、audio、IMU、jsonl 元信息。

### 3.1 与参考实现的关键差异(适配 foxglove schema)

| 项 | 参考实现(oeego schema) | 本实现(foxglove schema) |
|---|---|---|
| 颜色视频 | `oeego.sensor_proto.CompressedImage`,format=h265,数据为 HEVC NAL → `-f hevc` demux | `foxglove.CompressedImage`,format=jpeg,数据为 JPEG → image2 输入(`-i frame_%06d.jpg`)。两者均 libx264 重编码 |
| 深度/置信路由 | 按 format 串判:`16`/`32` → depth,`8UC1` → confidence | 本文件两路均 8UC1,改按 **topic** 路由:`/depth/image_raw` → depth,`/depth/confidence` → confidence |
| 深度帧 bytes | CompressedImage.data(压缩 bytes) | RawImage.data(已解码原始像素 bytes) |
| camera_info | `oeego.sensor_proto.CameraInfo`,K/R/P 为 Matrix3d,`header.frame_id` | `foxglove.CameraCalibration`,K/R/P 为 repeated double,顶层 `frame_id` 字段;R=identity、P=full,按数据原样输出 |
| transforms 时间戳 | `header_ts_ns(t.header)`(TransformStamped.header.stamp) | `foxglove.FrameTransform.timestamp`(seconds+nanos);逻辑等价 |
| hand 3D | `EgoHandPoseFrame.poses[].landmarks[].{x,y,z}`,固定 21 点 | `foxglove.PointCloud` 解码 xyz(复用 `decoders.decode_pointcloud`),13-21 点 ragged;沿用参考的 `dtype=object` 数组结构 |
| case_id | 任务入参 `case_id`(流水线传入) | `onboard_config.mission_id`;缺失则从 mcap 文件名 `SN_..._<digits>_...` 抽取 |

## 4. `parse_report.json` 结构

与参考逐字段对齐:

```json
{
  "case_id": "1543953655491727360",
  "outputs": [
    {"key": "video_head", "path": "<out>/video_head.mp4", "desc": "H265/H264 视频流 remux 为 mp4"},
    {"key": "depth_frames", "path": "<out>/depth_frames.npz", "desc": "原始深度帧 (16UC1/32)"},
    {"key": "confidence_frames", "path": "<out>/confidence_frames.npz", "desc": "置信度帧 (8UC1)"},
    {"key": "camera_info", "path": "<out>/camera_info.json", "desc": "相机内参/畸变 (CameraInfo)"},
    {"key": "transforms", "path": "<out>/transforms.npz", "desc": "每帧 world→child 6DoF (TF2Message)"},
    {"key": "hand_keypoints_3d", "path": "<out>/hand_keypoints_3d.npz", "desc": "3D 手部关键点 (EgoHandPoseFrame 3D)"},
    {"key": "parse_report", "path": "<out>/parse_report.json", "desc": "iPhone MCAP 各通道解析摘要"}
  ],
  "channels": [
    {"channel_id": 8, "topic": "/cam_head_l/depth/image_raw", "schema": "foxglove.RawImage",
     "count": 7014, "bytes": 344628000, "span_s": 116.9, "hz": 60.0,
     "sample": {"width": 256, "height": 192, "encoding": "8UC1", "data_len": 49152}}
  ],
  "camera_infos": ["head_left_camera_color_optical_frame", "head_left_camera_depth_optical_frame"],
  "transform_count": 7016,
  "hand_3d_count": 4397,
  "hand_2d_count": 0,
  "video_nal_count": 3508,
  "depth_frame_count": 7014,
  "confidence_frame_count": 7014
}
```

`channels[]` 对**所有**通道都列一行(含不导出的),每行含 `channel_id/topic/schema/count/bytes/span_s/hz/sample`。`sample` 按通道 schema 给一帧摘要:

- `foxglove.CompressedImage`(ch20):`{"format": "jpeg", "data_len": <n>}`
- `foxglove.RawImage`(ch8/ch9):`{"width", "height", "encoding", "data_len"}`
- `foxglove.CameraCalibration`(ch10/ch11):`{"width","height","distortion_model","fx","fy","cx","cy"}`(fx=K[0], fy=K[4], cx=K[2], cy=K[5])
- `foxglove.FrameTransforms`(ch14):`{"transforms_per_frame": 1, "first": "world->head_left_camera_color_optical_frame"}`
- `foxglove.PointCloud`(ch16):`{"points": <K>, "frame_id": "..."}`
- 其他通道:`sample = null`(不强行解码)

## 5. 架构与模块

### 5.1 仓库布局(新增)

```
mcap-parser/
  src/mcap_parser/
    __init__.py
    reader.py            # 现有 PICO McapFile(不动)
    decoders.py          # 复用 decode_pointcloud;新增 oeego proto 池构建助手
    ios_parse.py         # 新增:IosMcapParser(单遍 demux + 7 文件 + parse_report)
    cli.py               # 新增 --profile {auto,iphone,pico} 路由
    manifest.py          # 现有(PICO 用,iPhone 不用)
    writers/             # 现有 PICO writers(不动)
  tests/
    conftest.py          # 新增合成 iPhone foxglove mcap fixture
    test_ios_parse.py    # 新增
```

### 5.2 `IosMcapParser` 核心流程(单遍流式)

```python
class IosMcapParser:
    def __init__(self, path): ...
    def parse(self, out_dir, *, profile="iphone", keep_frames_on_fail=False) -> dict:
        summary = reader.get_summary()
        chan_info = {cid: (topic, schema_name) for ...}   # 预读 channels + schemas
        # 预解析 onboard_config 类(oeego 自定义 proto,FileDescriptorSet -> DescriptorPool)
        oc_cls = _build_oeego_cls(summary, ends_with="OnboardConfig")
        # 累加器
        video_jpegs, depth_frames, conf_frames, camera_infos, transforms, hand_3d = [], [], [], {}, [], []
        channel_summary = defaultdict(...)
        for schema, channel, msg in reader.iter_messages():
            cid = channel.id; gname = chan_info[cid][1]
            _accumulate(...)  # 按 schema_name 分派到对应累加器
        _write_outputs(...)   # 写 7 文件 + parse_report
```

- **schema 分派**:按 `schema.name` 的 foxglove 名 + topic 路由,不写死 channel id。
- **oeego proto 解码**(仅 onboard_config 需要):`schema.data` 是 `FileDescriptorSet`,用 `descriptor_pool.DescriptorPool().Add(fd)` 建池后 `message_factory.GetMessageClass(pool.FindMessageTypeByName(schema.name))`。该助手放 `decoders.py`,可复用于其他 oeego proto。
- **内存**:JPEG 逐帧约 200 KB × 3508 ≈ 700 MB——`feed` 阶段不常驻,逐帧写入临时 jpg 文件后再 ffmpeg。深度 bytes 7014×49 KB ≈ 340 MB(常驻 object 数组,可接受;或先落临时再打包,实现时按需)。其余均小。

### 5.3 各产物细节

**video_head.mp4**
- `feed`:ch20 每帧 `CompressedImage.data`(JPEG bytes)写入临时目录 `frame_{i:06d}.jpg`。按 `iter_messages` 顺序,**不按 log_time 排序**(JPEG 无 B-frame 重排问题,但保持写入顺序与参考一致)。
- `finalize`:fps 取自 onboard_config.target_frame_rate(30);`ffmpeg -y -framerate 30 -i <tmp>/frame_%06d.jpg -c:v libx264 -bf 0 -g 30 -preset fast -pix_fmt yuv420p -an <out>/video_head.mp4`。
- 失败语义:ffmpeg 不可用或返回非 0 → 不产 mp4;若 `keep_frames_on_fail=True`,保留 jpg 序列到 `<out>/color_frames/`,parse_report 标 `video_head` 缺失;否则删临时 jpg。默认 `keep_frames_on_fail=False`。
- 若某帧数据非 JPEG 魔数(`FF D8`):按 NAL Annex-B(`00 00 00 01`/`00 00 01`)走 `-f hevc` 分支(复用参考 `_remux_nals_to_mp4` 逻辑);既非 JPEG 也非 NAL 则跳过该帧并记 `decode_error`。

**depth_frames.npz / confidence_frames.npz**
- `feed`:ch8 → depth,ch9 → confidence(按 topic)。每帧 `(log_time_ns, bytes(RawImage.data), f"{encoding};{width}x{height}")` append。
- `finalize`:`np.savez_compressed`,键 `timestamps`(int64)、`frames`(object,bytes)、`formats`(object)。

**camera_info.json**
- `feed`:ch10/ch11 每帧 `CameraCalibration` → `camera_infos[frame_id] = {frame_id,width,height,distortion_model,D(list),K(list 9),R(list 9),P(list 12)}`(每帧相同,后帧覆盖无碍)。
- `finalize`:`json.dump(..., indent=2, ensure_ascii=False)`。

**transforms.npz**
- `feed`:ch14 每帧 `FrameTransforms.transforms`,逐个 transform append `{tx,ty,tz,qx,qy,qz,qw,timestamp_ns}`(ts = `timestamp.seconds*1e9 + nanos`)。
- `finalize`:`np.array([[tx,ty,tz,qx,qy,qz,qw,ts]...], dtype=float64)` → `np.savez_compressed(transforms=arr)`。

**hand_keypoints_3d.npz**
- `feed`:ch16 每帧 `PointCloud`,`pts = decode_pointcloud(obj)[0]`(N,3)float64→float32;`ts_ns = timestamp.seconds*1e9+nanos`;append `(pts, ts_ns)`。
- `finalize`:`landmarks = np.array([p for p,_ in ...], dtype=object)`;`timestamps = np.array([...], dtype=int64)`;`np.savez_compressed(landmarks=landmarks, timestamps=timestamps)`。

**parse_report.json**
- 每通道 `channel_summary[cid]` 累计 `count/bytes/t0/t1/schema/topic/sample`。`span_s=(t1-t0)/1e9`,`hz=count/span_s`。
- `outputs` 按 §4 表(仅实际产出的;`parse_report` 自身最后加入)。
- `counts`:`camera_infos`(list of frame_id)、`transform_count`、`hand_3d_count`、`hand_2d_count`(=0)、`video_nal_count`(=颜色帧数)、`depth_frame_count`、`confidence_frame_count`。

## 6. CLI

```
python -m mcap_parser <mcap_path> -o <out_dir>
  [--profile {auto,iphone,pico}]   # 默认 auto
  [--keep-frames-on-fail]          # ffmpeg 失败时保留 jpg 兜底
  [--overwrite]                    # 默认要求 out_dir 不存在或为空
```

- **auto 识别**:扫描 summary 的 channel topics,若存在 `/cam_head_l/` 前缀 → iphone;否则 pico。`--profile` 显式覆盖。
- 幂等:`--overwrite` 先清空 out_dir。
- 进度:每 1000 消息打一行日志。
- 退出码:成功 0;ffmpeg 缺失但其余成功 0(parse_report 标注);解析异常 1。

## 7. 错误处理

- **未知/不支持 schema**:不导出该通道产物,但 `parse_report.channels[]` 仍列该通道(`sample=null`)。
- **oeego proto 建池失败**(onboard_config 不可解码):fps 退化为按 log_time 计算(`count/span`,span>0 否则 30);case_id 退化为文件名抽取。
- **protobuf 解码失败**:记 stderr 警告,跳过该消息,继续。
- **ffmpeg 失败**:见 §5.3 video_head 失败语义,不中断其余产物。
- **空通道**:某产物无帧 → 不写该文件,`outputs` 不列,`counts` 计 0。

## 8. 测试策略

- **单测 `test_ios_parse.py`**:`conftest.py` 用 `mcap.writer.Writer` + foxglove protobuf 构造一个极小 iPhone mcap(CompressedImage/jpeg 3 帧、RawImage 3 帧×2、CameraCalibration 2 路、FrameTransforms 3 帧、PointCloud 3 帧 + 1 个 oeego OnboardConfig)。断言:
  - `depth_frames.npz`:`timestamps(3,)`、`frames(3,)` object、`formats[0]=="8UC1;256x192"`。
  - `confidence_frames.npz`:同上。
  - `camera_info.json`:2 组,`K` 长 9。
  - `transforms.npz`:`transforms(3,8)`,最后一列 = ts ns。
  - `hand_keypoints_3d.npz`:`landmarks(3,)` object,每元素 `(K,3)`。
  - `video_head.mp4`:`cv2.VideoCapture` 可打开,帧数 == 3(需 ffmpeg;无 ffmpeg 时跳过该断言)。
  - `parse_report.json`:`outputs` 含 7 项,`channels[]` 含全部合成通道,`counts` 字段齐。
- **集成测试**:对真实 `data/head_pose_mcap.mcap` 截前 ~900 帧(`scripts/truncate_mcap.py` 已有)跑一次,断言 6 个产物存在 + 形状符合 §3 + parse_report `case_id==1543953655491727360`、`transform_count>0`。758 MB 全量不入 CI。

## 9. 依赖

- 运行:现有 `mcap`、`numpy`、`opencv-python`、`foxglove-schemas-protobuf`、`google.protobuf`(已传递依赖)。系统 `ffmpeg`(JPEG→mp4 重编码,可选)。
- CLI:现有 `fire`。
- 测试:`pytest`、`mcap`(写 fixture)。
- **不新增** Python 依赖。

## 10. 非目标(YAGNI)

- 不做可视化/回放。
- 不导出参考契约外的产物(audio/IMU/逐帧 pcd/2D 手部/jsonl 元信息)。
- 不做各流帧数对齐(manifest 各自记录 count)。
- 不做 3D→2D 投影。
- 不处理 PICO 版 mcap(已有 `McapFile` 负责;`--profile auto` 自动分流)。

## 11. 验收标准

1. `python -m mcap_parser data/head_pose_mcap.mcap -o out_iphone` 产出 §3 表中 6 个数据文件 + `parse_report.json`,共 7 个。
2. `video_head.mp4` 可被 `cv2.VideoCapture` 打开,帧数 ≈ 3508,fps ≈ 30。
3. `depth_frames.npz`:`timestamps(7014,)`、`frames(7014,)` object、`formats[0]=="8UC1;256x192"`。
4. `confidence_frames.npz`:同上,帧数 7014。
5. `camera_info.json`:2 组(`head_left_camera_color_optical_frame`、`head_left_camera_depth_optical_frame`),`K` 长 9。
6. `transforms.npz`:`transforms(7016,8)`,前 7 列为位姿、第 8 列为 ts ns。
7. `hand_keypoints_3d.npz`:`landmarks(4397,)` object,每元素 `(K,3) float32`;`timestamps(4397,)` int64。
8. `parse_report.json`:`case_id=="1543953655491727360"`,`outputs` 7 项,`channels[]` 含全部 20 通道,`counts` 字段齐全。
9. 与 `openelephant-data-app` 参考节点输出**逐文件结构对齐**(键名、键顺序、dtype 一致)。
10. 单测全绿;集成测试对截断 MCAP 通过。

## 12. 已决默认值

- `--profile auto` 默认;iPhone 识别靠 `/cam_head_l/` topic 前缀。
- JPEG→mp4 走 libx264 重编码(`-bf 0 -g 30`),不尝试 `-c copy`(JPEG 无法 remux)。
- 深度/置信路由按 topic,不按 format 串(本文件两路均 8UC1)。
- `hand_keypoints_2d` 不产出(`hand_2d_count=0`)。
- case_id 取自 `onboard_config.mission_id`,缺失则文件名抽取。
- `keep_frames_on_fail` 默认 False(ffmpeg 失败不保留 jpg)。
