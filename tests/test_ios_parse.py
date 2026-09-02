"""iPhone MCAP 解析单测。"""
from __future__ import annotations

import numpy as np
import pytest

from mcap_parser._testing import ONBOARD_CONFIG_SCHEMA_B64
from mcap_parser.decoders import oeego_msg_class


def test_oeego_msg_class_builds_and_parses():
    schema_data = __import__("base64").b64decode(ONBOARD_CONFIG_SCHEMA_B64)
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


from mcap.reader import make_reader


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
