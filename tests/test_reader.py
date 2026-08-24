import pytest

from mcap_parser.reader import McapFile


def test_reader_dispatch_all_writers(synth_mcap, tmp_path):
    out = tmp_path / "out"
    ctx = McapFile(synth_mcap).parse(out)
    outs = set(ctx.outputs.keys())
    assert "camera_calibration.json" in outs
    assert "left_keypoints_3d.npy" in outs
    assert "right_keypoints_3d.npy" in outs
    assert "left_hand_vertices.npz" in outs
    assert "tracking_data_tf.jsonl" in outs
    assert "wrist_poses.npz" in outs
    assert "head_pose_trajectory.npy" in outs
    assert "aligned_pointclouds.npz" in outs
    assert "static_scene.ply" in outs
    assert "depth/" in outs
    for side in ("left", "right"):
        assert f"head_{side}_camera_undistorted.mp4" in outs


def test_reader_overwrite_guard(synth_mcap, tmp_path):
    out = tmp_path / "out"
    McapFile(synth_mcap).parse(out)
    with pytest.raises(RuntimeError):
        McapFile(synth_mcap).parse(out)  # 非空、未 overwrite
    # overwrite 通过
    McapFile(synth_mcap).parse(out, overwrite=True)
