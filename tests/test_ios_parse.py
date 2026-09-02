"""iPhone MCAP 解析单测。"""
from __future__ import annotations

import base64
import json

import numpy as np

from mcap.reader import make_reader

from mcap_parser._testing import ONBOARD_CONFIG_SCHEMA_B64
from mcap_parser.decoders import oeego_msg_class
from mcap_parser.ios_parse import IosMcapParser


def test_oeego_msg_class_builds_and_parses():
    schema_data = base64.b64decode(ONBOARD_CONFIG_SCHEMA_B64)
    cls = oeego_msg_class(
        "oeego.system_proto.MissionManagerProto.OnboardConfig", schema_data
    )
    m = cls()
    m.mission_id = "123"
    m.target_frame_rate = 30
    roundtrip = cls()
    roundtrip.ParseFromString(m.SerializeToString())
    assert roundtrip.mission_id == "123"
    assert roundtrip.target_frame_rate == 30


def test_ios_synth_fixture_has_channels(ios_synth_mcap):
    with open(ios_synth_mcap, "rb") as f:
        s = make_reader(f).get_summary()
    topics = {ch.topic for ch in s.channels.values()}
    assert "/cam_head_l/color/image_compressed_h265" in topics
    assert "/cam_head_l/depth/image_raw" in topics
    assert "/cam_head_l/depth/confidence" in topics
    assert "/cam_head_l/color/camera_info" in topics
    assert "/cam_head_l/depth/camera_info" in topics
    assert "/tf" in topics
    assert "/cam_head_l/ego_hand_pose/pointcloud_3d" in topics
    assert "/oeego/mission_manager/onboard_config" in topics


def test_ios_parse_writes_report_channels(ios_synth_mcap, tmp_path):
    out = tmp_path / "out"
    IosMcapParser(ios_synth_mcap).parse(out, overwrite=True)
    report = json.loads((out / "parse_report.json").read_text())
    assert report["case_id"] == "1543953655491727360"
    topics = {c["topic"] for c in report["channels"]}
    assert "/cam_head_l/color/image_compressed_h265" in topics
    assert "/tf" in topics
    assert len(report["channels"]) == 8  # 8 个通道
    # 每条 channel 记录都有 count/bytes/span_s/hz/schema
    for c in report["channels"]:
        assert c["count"] == 3
        assert c["schema"]
        assert c["span_s"] >= 0
        assert c["hz"] >= 0


def test_ios_parse_camera_info(ios_synth_mcap, tmp_path):
    out = tmp_path / "out"
    IosMcapParser(ios_synth_mcap).parse(out, overwrite=True)
    ci = json.loads((out / "camera_info.json").read_text())
    assert set(ci.keys()) == {
        "head_left_camera_color_optical_frame",
        "head_left_camera_depth_optical_frame",
    }
    color = ci["head_left_camera_color_optical_frame"]
    assert color["width"] == 1920 and color["height"] == 1440
    assert color["distortion_model"] == "plumb_bob"
    assert len(color["K"]) == 9
    assert len(color["R"]) == 9
    assert len(color["P"]) == 12
    assert color["frame_id"] == "head_left_camera_color_optical_frame"


def test_ios_parse_transforms(ios_synth_mcap, tmp_path):
    out = tmp_path / "out"
    IosMcapParser(ios_synth_mcap).parse(out, overwrite=True)
    z = np.load(out / "transforms.npz")
    arr = z["transforms"]
    assert arr.shape == (3, 8)
    assert arr.dtype == np.float64
    # 前 7 列为位姿,第 8 列为 ts ns(epoch 量级)
    assert abs(arr[0, 0] - (-0.07)) < 1e-6
    assert abs(arr[0, 3] - (-0.49)) < 1e-6
    assert arr[0, 7] > 1_000_000_000  # epoch ns


def test_ios_parse_depth_and_confidence(ios_synth_mcap, tmp_path):
    out = tmp_path / "out"
    IosMcapParser(ios_synth_mcap).parse(out, overwrite=True)
    for name, fmt_prefix in [("depth_frames", "8UC1;4x4"),
                             ("confidence_frames", "8UC1;4x4")]:
        z = np.load(out / f"{name}.npz", allow_pickle=True)
        assert z["timestamps"].shape == (3,)
        assert z["timestamps"].dtype == np.int64
        assert z["frames"].shape == (3,)
        assert z["frames"].dtype == object
        assert isinstance(z["frames"][0], bytes)
        assert z["formats"][0] == fmt_prefix


def test_ios_parse_hand_keypoints_3d(ios_synth_mcap, tmp_path):
    out = tmp_path / "out"
    IosMcapParser(ios_synth_mcap).parse(out, overwrite=True)
    z = np.load(out / "hand_keypoints_3d.npz", allow_pickle=True)
    lm = z["landmarks"]
    ts = z["timestamps"]
    assert lm.shape == (3,)
    assert lm.dtype == object
    assert lm[0].shape == (21, 3)
    assert lm[0].dtype == np.float32
    assert ts.shape == (3,)
    assert ts.dtype == np.int64
