# mcap-parser

[English](README_EN.md) | [中文](README.md)

Parse `head_pose_mcap.mcap` (foxglove/protobuf) into standalone files, with formats aligned to the raw outputs of `openelephant-data`.

## Installation

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

`uv` automatically downloads the required Python (≥3.10) and creates `.venv/`. Afterwards, use `uv run` to execute commands inside the environment — no manual activation needed:

```bash
uv run mcap-parser --help
```

> Optional system dependency: video output relies on `ffmpeg` (remux H.264 → `.mp4`). If `ffmpeg` is not installed, no error is raised — the output automatically falls back to raw `.h264` streams. On macOS: `brew install ffmpeg`.

> Note: when installed alongside pandas/streamlit in a shared environment, the `numpy` version constraints may conflict. This repo is compatible with both numpy 1.26 and 2.x; pin `numpy==1.26.4` if needed (e.g. `uv add "numpy==1.26.4"`).

## Usage

```bash
uv run python -m mcap_parser <path/to/head_pose_mcap.mcap> --out_dir <out> [--overwrite true]
  [--side left|right|both]
  [--streams video,hand_keypoints,hand_mesh,head_trajectory,pointcloud,body_tf,camera_calib,depth]
  [--per_frame_pcd true|false]
```

## Outputs

- Video: `head_{left,right}_camera_undistorted.mp4` (ffmpeg remux; falls back to `.h264` if ffmpeg is missing)
- Hand keypoints: `{left,right}_keypoints_{3d,2d}.npy`, shapes `(N,21,3)` / `(N,21,2)`
- Hand mesh: `{left,right}_hand_vertices.npz` (`vertices(N,778,3)` + `faces(1538,3)` + `is_right` + `detected`)
- Head trajectory: `head_pose_trajectory.npy`, shape `(N,7)` = `[x,y,z,qx,qy,qz,qw]`
- Point cloud: `aligned_pointclouds.npz` + `pointclouds/frame_000000.pcd` (per-frame binary) + `static_scene.ply`
- Body/wrist tf: `tracking_data_tf.jsonl` (44 transforms per frame) + `wrist_poses.npz` (`left/right (N,7)` + `timestamps`)
- Camera calibration: `camera_calibration.json` (5 groups: left/right × raw/undistorted + `head_depth_camera`)
- Depth images: `head_depth_camera.npz` (`depth(N,240,320)` uint16 + `timestamps_ns`/`frame_indices` indices; mono16 raw values stacked, zlib-compressed)
- Manifest: `manifest.json` (per-stream frame counts + timestamps_ns + missing_keys + not_recoverable)

## Not Recoverable

The tf_tree in `head_pose_mcap.mcap` contains only 2 cameras + 2 wrists + 40 finger joints — **there is no 24-joint SMPL body skeleton**.
The original `body_keypoints_labels.npz` (24 joints) comes from the upstream `body_pose_prelabel` stage of `openelephant-data` and cannot be recovered from this MCAP. Therefore, body_tf only produces `tracking_data_tf.jsonl` (a faithful, complete dump) + `wrist_poses.npz` (wrist roots), and the hand 2D keypoints are taken directly from the ch3 channel (no projection). See `docs/2026-08-19-mcap-parser-design.md` for details.

## Testing

```bash
uv run pytest -v              # all unit tests (synthetic foxglove MCAP fixtures)
uv run python scripts/truncate_mcap.py <real.mcap> tests/data/head_pose_mcap_trim.mcap 300
uv run pytest tests/test_integration.py -v   # integration test with a real truncated MCAP
```
