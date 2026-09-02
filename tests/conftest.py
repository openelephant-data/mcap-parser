"""pytest 配置：合成 MCAP 夹具。"""
from __future__ import annotations

from pathlib import Path

import pytest

from mcap_parser._testing import build_synth_mcap, build_ios_synth_mcap


@pytest.fixture
def synth_mcap(tmp_path) -> Path:
    """3 帧合成 foxglove MCAP。"""
    return build_synth_mcap(tmp_path / "synth.mcap")


@pytest.fixture
def ios_synth_mcap(tmp_path) -> Path:
    """3 帧 iPhone 风格 foxglove MCAP。"""
    return build_ios_synth_mcap(tmp_path / "ios_synth.mcap")
