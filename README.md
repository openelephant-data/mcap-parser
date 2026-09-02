# mcap-parser

把 iPhone/ARKit foxglove MCAP（`head_pose_mcap.mcap`）解析为独立文件，输出契约对齐 `openelephant-data-app` 的 `IosMcapParseTask` 节点。同时保留对 PICO 版 MCAP 的解析能力（`--profile pico`）。

## 安装

使用 [uv](https://docs.astral.sh/uv/) 安装：

```bash
uv sync --extra dev
```

`uv` 自动下载所需 Python（≥3.10）并创建 `.venv/`。之后用 `uv run` 在环境内执行命令，无需手动 activate：

```bash
uv run mcap-parser --help
```

> 可选系统依赖：视频输出依赖 `ffmpeg`（将 JPEG 序列重编码为 H.264 mp4）。未安装时不报错，跳过 `video_head.mp4`，其余产物照常输出，`parse_report.json` 的 `video_head_status` 字段记录失败原因。macOS 可 `brew install ffmpeg`。

> 注：若与共享环境中的 pandas/streamlit 同装，`numpy` 版本约束可能冲突；本仓代码兼容 numpy 1.26 与 2.x，按需固定 `numpy==1.26.4` 即可（如 `uv add "numpy==1.26.4"`）。

## 用法

```bash
uv run python -m mcap_parser <path/to/head_pose_mcap.mcap> --out_dir <out> [--overwrite true]
  [--profile auto|iphone|pico]         # 默认 auto：topic 含 /cam_head_l/ → iphone，否则 pico
  [--keep_frames_on_fail true|false]   # ffmpeg 失败时保留原始 jpg/.h265 兜底
```

`--profile auto` 按通道 topic 自动识别：含 `/cam_head_l/` 前缀走 iPhone 路径，否则走 PICO 路径。

## 输出（iPhone）

- `video_head.mp4` — 颜色流（JPEG 序列）重编码为 H.264（libx264，`-bf 0 -g fps`），fps 取自 `onboard_config.target_frame_rate`
- `depth_frames.npz` — 原始深度帧：`timestamps`(N,) int64 + `frames`(N,) object(bytes) + `formats`("8UC1;256x192")
- `confidence_frames.npz` — 置信度帧，同上结构
- `camera_info.json` — 相机内参，按 `frame_id` 键：`{frame_id,width,height,distortion_model,D,K[9],R[9],P[12]}`（color + depth 两组）
- `transforms.npz` — 每帧 world→child 6DoF：`transforms`(N,8) float64 = `[tx,ty,tz,qx,qy,qz,qw,timestamp_ns]`
- `hand_keypoints_3d.npz` — 3D 手部关键点：`landmarks`(N,) object of (K,3) float32 + `timestamps`(N,) int64
- `parse_report.json` — 解析摘要：`case_id` + `outputs[]` + `channels[]`（id/topic/schema/count/bytes/span_s/hz/sample）+ 计数（`transform_count` / `hand_3d_count` / `video_nal_count` / `depth_frame_count` / `confidence_frame_count` 等）+ `video_head_status`

> PICO 版 MCAP：`--profile pico` 走 `McapFile`，产物与契约见 `docs/2026-08-19-mcap-parser-design.md`。

## 测试

```bash
uv run pytest -v      # 全部单测（合成 foxglove MCAP 夹具，含 iPhone + PICO）

# iPhone 集成测试需先截断真实 mcap（前 30s，不入 git）：
uv run python scripts/truncate_iphone_mcap.py data/head_pose_mcap.mcap tests/data/head_pose_trunc.mcap 30
uv run pytest tests/test_ios_integration.py -v

# PICO 集成测试：
uv run python scripts/truncate_mcap.py <pico.mcap> tests/data/head_pose_mcap_trim.mcap 300
uv run pytest tests/test_integration.py -v
```

## 设计文档

- `docs/2026-09-02-ios-mcap-parse-design.md` — iPhone MCAP 解析设计
- `docs/2026-09-02-ios-mcap-parse-plan.md` — iPhone 实现计划
- `docs/2026-08-19-mcap-parser-design.md` — PICO MCAP 解析设计
