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
