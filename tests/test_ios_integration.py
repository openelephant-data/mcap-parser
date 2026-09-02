"""真实 head_pose_mcap.mcap 截断版(前 30s)iPhone 集成测试。

依赖 tests/data/head_pose_trunc.mcap(由 scripts/truncate_iphone_mcap.py 生成,不入 git)。
文件不存在时 skip,故 CI 默认不跑。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TRUNC = Path(__file__).parent / "data" / "head_pose_trunc.mcap"
pytestmark = pytest.mark.skipif(
    not TRUNC.exists(), reason=f"截断 mcap 不存在: {TRUNC}"
)


def test_ios_parse_real_truncated(tmp_path):
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "-m", "mcap_parser", str(TRUNC),
         "--out_dir", str(out), "--overwrite", "true"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads((out / "parse_report.json").read_text())
    assert report["case_id"] == "1543953655491727360"
    assert report["transform_count"] > 0
    assert report["depth_frame_count"] > 0
    assert report["confidence_frame_count"] > 0
    assert report["hand_3d_count"] > 0
    assert set(report["camera_infos"]) == {
        "head_left_camera_color_optical_frame",
        "head_left_camera_depth_optical_frame",
    }
    # 形状
    tr = np.load(out / "transforms.npz")["transforms"]
    assert tr.shape[1] == 8
    df = np.load(out / "depth_frames.npz", allow_pickle=True)
    assert df["frames"][0]  # bytes 非空
    hk = np.load(out / "hand_keypoints_3d.npz", allow_pickle=True)
    assert hk["landmarks"][0].shape[1] == 3
    # video_head.mp4(若 ffmpeg 可用)
    if (out / "video_head.mp4").exists():
        import cv2
        cap = cv2.VideoCapture(str(out / "video_head.mp4"))
        assert cap.isOpened()
        cap.release()
