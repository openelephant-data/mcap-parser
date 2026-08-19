"""手部关键点 writer：ch2(3D) + ch3(2D) -> left/right_keypoints_{3d,2d}.npy。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..decoders import HAND_SKELETON_IDS, N_HAND_JOINTS
from .base import ParseContext, Writer

KP_TOPICS = ("/pose_algo_body/hand_keypoints", "/pose_algo_body/hand_keypoints_2d")


class HandKeypointsWriter(Writer):
    name = "hand_keypoints"
    topics = KP_TOPICS
    schema_name = "foxglove.SceneUpdate"

    def __init__(self, side_filter: str = "both"):
        self.side_filter = side_filter

    def feed(self, topic: str, msg, ctx: ParseContext) -> None:
        is_2d = topic.endswith("_2d")
        obj = self._decode(msg.data)
        ts = msg.publish_time
        buf = ctx.hand_2d if is_2d else ctx.hand_3d
        present = {"left": False, "right": False}
        for e in obj.entities:
            if e.id == HAND_SKELETON_IDS[0]:
                side = "left"
            elif e.id == HAND_SKELETON_IDS[1]:
                side = "right"
            else:
                continue
            if self.side_filter != "both" and side != self.side_filter:
                continue
            if len(e.spheres) != N_HAND_JOINTS:
                continue
            pts = np.array(
                [(s.pose.position.x, s.pose.position.y, s.pose.position.z) for s in e.spheres],
                dtype=np.float32,
            )
            buf[side].append(pts if not is_2d else pts[:, :2])
            present[side] = True
        # 3D 通道负责记 timestamps（两侧都记，保证长度对齐）
        if not is_2d:
            for side in ("left", "right"):
                if not present[side]:
                    # 缺失侧用全 nan 占位（3D）
                    buf[side].append(np.full((N_HAND_JOINTS, 3), np.nan, dtype=np.float32))
                ctx.hand_ts[side].append(ts)

    def finalize(self, out_dir: Path, ctx: ParseContext) -> None:
        for side in ("left", "right"):
            for buf, suffix in ((ctx.hand_3d[side], "_3d"), (ctx.hand_2d[side], "_2d")):
                if not buf:
                    continue
                arr = np.stack(buf)
                np.save(out_dir / f"{side}_keypoints{suffix}.npy", arr)
                ctx.outputs[f"{side}_keypoints{suffix}.npy"] = {
                    "frames": len(buf),
                    "side": side,
                    "timestamps_ns": list(ctx.hand_ts[side][: len(buf)]),
                }
