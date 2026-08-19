import json
from pathlib import Path

import numpy as np
import pytest

from mcap_parser.reader import McapFile

TRIM = Path(__file__).parent / "data" / "head_pose_mcap_trim.mcap"
pytestmark = pytest.mark.skipif(not TRIM.exists(), reason="截断 MCAP 不存在，跳过集成测试")


def test_integration_full_parse(tmp_path):
    out = tmp_path / "out"
    ctx = McapFile(TRIM).parse(out)
    m = json.loads((out / "manifest.json").read_text())
    assert m["frame_count"] > 0
    assert m["message_count"] > 0
    # 手部网格
    z = np.load(out / "left_hand_vertices.npz")
    assert z["vertices"].shape[1:] == (778, 3)
    assert z["faces"].shape == (1538, 3)
    # 头部轨迹 7 列
    traj = np.load(out / "head_pose_trajectory.npy")
    assert traj.shape[1] == 7
    # wrist
    wp = np.load(out / "wrist_poses.npz")
    assert wp["left"].shape[1] == 7
    # calib
    calib = json.loads((out / "camera_calibration.json").read_text())
    assert len(calib) == 4
    # 视频存在（可能 muxed=False 但 .mp4/.h264 至少一个）
    assert any((out / f"head_{s}_camera_undistorted.mp4").exists()
               or (out / f"{s}.h264").exists() for s in ("left", "right"))
    # 点云 PCD 数 == pc_frames
    n_pc = ctx.outputs["aligned_pointclouds.npz"]["pc_frames"]
    pcds = list((out / "pointclouds").glob("*.pcd"))
    assert len(pcds) == n_pc
