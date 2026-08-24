import json

from mcap_parser.reader import McapFile


def test_manifest_written(synth_mcap, tmp_path):
    out = tmp_path / "out"
    McapFile(synth_mcap).parse(out)
    m = json.loads((out / "manifest.json").read_text())
    assert m["source_mcap"].endswith("synth.mcap")
    assert m["message_count"] == 3 * 17  # 3 帧 × 17 条消息（含两份 hand_mesh + 深度图 + 深度标定）
    assert m["frame_count"] == 3
    assert "camera_calibration.json" in m["outputs"]
    assert "body_joint_positions" in m["missing_keys"]["aligned_pointclouds.npz"]
    assert any("24-SMPL body skeleton" in s for s in m["not_recoverable"])
    assert m["outputs"]["tracking_data_tf.jsonl"]["transforms_per_frame"] > 0
