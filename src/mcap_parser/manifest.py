"""manifest.json 写入器。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .writers.base import ParseContext, Writer

MISSING_KEYS = {
    "aligned_pointclouds.npz": [
        "body_joint_positions",
        "body_joint_quaternions",
        "body_joint_names",
        "left_hand_quaternions",
        "right_hand_quaternions",
    ]
}
NOT_RECOVERABLE = [
    "24-SMPL body skeleton (from body_pose_prelabel upstream)",
    "body_keypoints_2d (24 joints)",
    "hand joint quaternions (ch2 只给位置)",
]


class ManifestBuilder:
    def __init__(
        self,
        mcap_path: Path,
        summary: Any,
        ctx: ParseContext,
        writers: list[Writer],
    ):
        self.mcap_path = mcap_path
        self.summary = summary
        self.ctx = ctx
        self.writers = writers

    def write(self, out_dir: Path) -> None:
        st = self.summary.statistics if self.summary else None
        frame_count = self.ctx.frame_count
        manifest = {
            "source_mcap": str(self.mcap_path),
            "source_size_bytes": self.mcap_path.stat().st_size,
            "message_count": st.message_count if st else 0,
            "start_time_ns": st.message_start_time if st else 0,
            "end_time_ns": st.message_end_time if st else 0,
            "frame_count": frame_count,
            "outputs": self.ctx.outputs,
            "missing_keys": MISSING_KEYS,
            "not_recoverable": NOT_RECOVERABLE,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )
