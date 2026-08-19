"""Writer 基类 + 共享 ParseContext。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ParseContext:
    """单遍扫描的共享状态。各 writer 在 feed 阶段写入，finalize 阶段只读跨流部分。"""

    # BodyTfWriter 写入（ch1，逐帧）：
    tf_cache: list = field(default_factory=list)  # list[ dict[child_id -> (parent, t(3,), q(4,), ts)] ]
    tf_ts: list = field(default_factory=list)  # list[int]

    # HandKeypointsWriter 写入（ch2/ch3，逐帧）：
    hand_3d: dict = field(default_factory=lambda: {"left": [], "right": []})   # list[(21,3) float32]
    hand_2d: dict = field(default_factory=lambda: {"left": [], "right": []})   # list[(21,2) float32]
    hand_ts: dict = field(default_factory=lambda: {"left": [], "right": []})   # list[int]（3d 通道记一次）

    # HeadTrajectoryWriter 写入（ch4，逐帧）：
    head_pos: list = field(default_factory=list)   # list[(3,)]
    head_ts: list = field(default_factory=list)     # list[int]

    # PointcloudWriter 写入（ch6/ch7）：
    pc_frames: list = field(default_factory=list)   # list[(points(M,3), colors(M,3), ts)]
    pc_static: Any = None  # (points(M,3), colors(M,3))

    # CameraCalibWriter 写入：
    calib: dict = field(default_factory=dict)  # {"left_undistorted": {...}, ...}

    # 所有 writer 写入：
    outputs: dict = field(default_factory=dict)  # {filename: manifest dict}
    frame_count: int = 0


class Writer(ABC):
    """通道/topic 驱动的输出 writer。子类只需实现 feed/finalize。"""

    name: str = ""
    topics: tuple[str, ...] = ()
    schema_name: str = ""

    def owns_topic(self, topic: str) -> bool:
        return any(topic.startswith(p) for p in self.topics)

    @abstractmethod
    def feed(self, topic: str, msg, ctx: ParseContext) -> None: ...

    @abstractmethod
    def finalize(self, out_dir: Path, ctx: ParseContext) -> None: ...

    def _decode(self, data: bytes):
        from ..decoders import decode  # 延迟导入避免循环

        return decode(self.schema_name, data)
