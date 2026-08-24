# mcap-parser

把 `head_pose_mcap.mcap`（foxglove/protobuf）解析为独立文件，格式对齐 `openelephant-data` 原始产物。

## 安装

使用 [uv](https://docs.astral.sh/uv/) 安装：

```bash
uv sync --extra dev
```

`uv` 会自动下载所需 Python（≥3.10）并创建 `.venv/`。之后用 `uv run` 在环境内执行命令，无需手动 activate：

```bash
uv run mcap-parser --help
```

> 可选系统依赖：视频输出依赖 `ffmpeg`（remux H.264 → `.mp4`）。未安装时不报错，自动退化为 `.h264` 裸流。macOS 可 `brew install ffmpeg`。

> 注：若与共享环境中的 pandas/streamlit 同装，`numpy` 版本约束可能冲突；本仓代码兼容 numpy 1.26 与 2.x，按需固定 `numpy==1.26.4` 即可（如 `uv add "numpy==1.26.4"`）。

## 用法

```bash
uv run python -m mcap_parser <path/to/head_pose_mcap.mcap> --out_dir <out> [--overwrite true]
  [--side left|right|both]
  [--streams video,hand_keypoints,hand_mesh,head_trajectory,pointcloud,body_tf,camera_calib,depth]
  [--per_frame_pcd true|false]
```

## 输出

- 视频：`head_{left,right}_camera_undistorted.mp4`（ffmpeg remux，缺 ffmpeg 退化为 `.h264`）
- 手部关键点：`{left,right}_keypoints_{3d,2d}.npy`，形状 `(N,21,3)` / `(N,21,2)`
- 手部网格：`{left,right}_hand_vertices.npz`（`vertices(N,778,3)` + `faces(1538,3)` + `is_right` + `detected`）
- 头部轨迹：`head_pose_trajectory.npy`，形状 `(N,7)` = `[x,y,z,qx,qy,qz,qw]`
- 点云：`aligned_pointclouds.npz` + `pointclouds/frame_000000.pcd`（逐帧 binary） + `static_scene.ply`
- 身体/手根 tf：`tracking_data_tf.jsonl`（每帧 44 个 transform） + `wrist_poses.npz`（`left/right (N,7)` + `timestamps`）
- 相机标定：`camera_calibration.json`（5 组：左/右 × raw/undistorted + `head_depth_camera`）
- 深度图：`depth/frame_{i:06d}.bin`（mono16 裸流，320×240，原样写出 `RawImage.data`，不转码）
- 清单：`manifest.json`（per-stream 帧数 + timestamps_ns + missing_keys + not_recoverable）

## 不可恢复

`head_pose_mcap.mcap` 的 tf_tree 只含 2 相机 + 2 腕 + 40 指节，**无 24-SMPL 身体骨架**。
原 `body_keypoints_labels.npz`（24 关节）来自 `openelephant-data` 的 `body_pose_prelabel` 上游，
本 MCAP 无法还原；故 body_tf 只产出 `tracking_data_tf.jsonl`（忠实全量）+ `wrist_poses.npz`（腕根），
手部 2D 直接取自 ch3 通道（不做投影）。详见 `docs/2026-08-19-mcap-parser-design.md`。

## 测试

```bash
uv run pytest -v              # 全部单测（合成 foxglove MCAP 夹具）
uv run python scripts/truncate_mcap.py <real.mcap> tests/data/head_pose_mcap_trim.mcap 300
uv run pytest tests/test_integration.py -v   # 真实截断 MCAP 集成测试
```
