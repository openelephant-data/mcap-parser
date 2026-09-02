"""iPhone MCAP 解析单测。"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

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


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def test_ios_parse_video_head(ios_synth_mcap, tmp_path):
    if not _has_ffmpeg():
        pytest.skip("ffmpeg not installed")
    out = tmp_path / "out"
    IosMcapParser(ios_synth_mcap).parse(out, overwrite=True)
    mp4 = out / "video_head.mp4"
    assert mp4.exists() and mp4.stat().st_size > 0
    import cv2
    cap = cv2.VideoCapture(str(mp4))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    cap.release()


def test_ios_parse_report_full(ios_synth_mcap, tmp_path):
    out = tmp_path / "out"
    IosMcapParser(ios_synth_mcap).parse(out, overwrite=True)
    report = json.loads((out / "parse_report.json").read_text())
    keys = {o["key"] for o in report["outputs"]}
    assert "parse_report" in keys
    assert "camera_info" in keys
    assert "transforms" in keys
    assert "depth_frames" in keys
    assert "confidence_frames" in keys
    assert "hand_keypoints_3d" in keys
    # video_head 仅在有 ffmpeg 时存在
    for key in ("camera_info", "transforms", "depth_frames",
                "confidence_frames", "hand_keypoints_3d", "parse_report"):
        e = next(o for o in report["outputs"] if o["key"] == key)
        assert e["desc"]
    expected_order = ["video_head", "depth_frames", "confidence_frames",
                      "camera_info", "transforms", "hand_keypoints_3d", "parse_report"]
    # video_head 仅在有 ffmpeg 时存在;过滤掉实际不存在的
    actual_order = [o["key"] for o in report["outputs"]]
    for k in actual_order:
        assert k in expected_order, f"unexpected output key {k}"
    # 在 expected_order 中的、且实际存在的 key,其相对顺序必须与 expected_order 一致
    present = [k for k in expected_order if k in actual_order]
    assert actual_order == present, f"outputs order mismatch: {actual_order} vs {present}"
    assert report["case_id"] == "1543953655491727360"
    assert report["transform_count"] == 3
    assert report["hand_3d_count"] == 3
    assert report["hand_2d_count"] == 0
    assert report["depth_frame_count"] == 3
    assert report["confidence_frame_count"] == 3
    assert report["video_nal_count"] == 3
    assert set(report["camera_infos"]) == {
        "head_left_camera_color_optical_frame",
        "head_left_camera_depth_optical_frame",
    }


def test_cli_auto_routes_iphone(ios_synth_mcap, tmp_path):
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "-m", "mcap_parser", str(ios_synth_mcap),
         "--out_dir", str(out), "--overwrite", "true"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    # iPhone 产物标志:parse_report.json + transforms.npz(非 PICO 的 manifest.json)
    assert (out / "parse_report.json").exists()
    assert (out / "transforms.npz").exists()
    assert not (out / "manifest.json").exists()


def test_cli_explicit_profile_pico_still_works(synth_mcap, tmp_path):
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "-m", "mcap_parser", str(synth_mcap),
         "--out_dir", str(out), "--overwrite", "true",
         "--profile", "pico"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "manifest.json").exists()
