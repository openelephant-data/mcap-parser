# 深度相机输出设计 (Spec)

- **项目**: `mcap-parser`
- **日期**: 2026-08-24
- **状态**: 已确认
- **背景**: `data/head_pose_mcap.mcap` 新增了深度相机相关通道，解析器需把这些数据流导出为独立文件。

## 1. 新增通道（实测）

| ch_id | topic | schema | 帧数 | 内容 |
|---|---|---|---|---|
| 15 | `/head_depth_camera/depth/image` | foxglove.RawImage | 509 | 深度图，320×240，encoding `mono16`（2 B/px），step=640，data 153600 B/帧 |
| 16 | `/head_depth_camera_params/head_depth_camera/camera_info` | foxglove.CameraCalibration | 509 | 深度相机内参（每帧相同） |

## 2. 输出契约

| 源通道 | 输出文件 | 格式 |
|---|---|---|
| ch15 | `head_depth_camera.npz` | `depth`(N,240,320) uint16（mono16 原值堆叠，zlib 压缩）+ `timestamps_ns`(N,) int64 + `frame_indices`(N,) int64 + `width/height/step` 标量 |
| ch16 | `camera_calibration.json`（键 `head_depth_camera`） | 与现有 `{left,right}_{raw,undistorted}` 平级，含 width/height/distortion_model/K/D/R/P/frame_id |

manifest 在 `outputs["head_depth_camera.npz"]` 记录：`frames`、`width`、`height`、`encoding`、`step`、`dtype`、`timestamps_ns`。

## 3. 模块改动

1. `decoders.py`：`MSG_CLASS`/`FILE_DESC` 注册 `foxglove.RawImage` → `RawImage_pb2.RawImage`。
2. 新 `writers/depth.py`：`DepthWriter`，topic 驱动，缓存 `(data, publish_time)` + 首帧元信息，`finalize` 按 step 切行、堆叠为 `depth(N,H,W) uint16`，写 `head_depth_camera.npz`（`depth` + `timestamps_ns` + `frame_indices` + `width/height/step`）。
3. `writers/camera_calib.py`：`CALIB_TOPICS` 追加 `"/head_depth_camera_params/"`；`feed` 先判 `"head_depth" in topic` → key `head_depth_camera`，避免被误判为 right。
4. `reader.py`：`STREAM_TO_CLASS["depth"]=DepthWriter`；`build_writers` 加无参 `DepthWriter()`。
5. `cli.py` / `__main__.py`：streams 列表文档加 `depth`。

## 4. 取舍

- 深度图出**单文件** `head_depth_camera.npz`（堆叠 + 索引），不做逐帧 `.bin`、不调 ffmpeg。
- depth 像素按 `step` 切行取每行前 `width*2` 字节，兼容行填充（实测 step==width*2，无填充）。
- `depth` 作为独立可选 stream，默认随 `ALL_STREAMS` 一起产出。
- 深度标定并入 `camera_calibration.json` 而非另立文件，保持单一标定入口。

## 5. 测试

- `_testing.py`：加 `raw_image_msg()`、`SCHEMA_TOPICS` 注册 ch15/ch16、`build_synth_mcap` 每帧喂深度图 + 深度标定。
- 新 `tests/test_depth.py`：逐帧 `.bin` 字节数、manifest 条目、目录命名。
- `test_camera_calib.py`：断言集合加 `head_depth_camera`。
- `test_cli.py` / `test_reader.py`：合成 MCAP 现含深度通道，确认不报错且 manifest 含 `depth/`。
