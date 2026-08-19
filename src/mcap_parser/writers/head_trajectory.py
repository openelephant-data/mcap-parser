"""头部轨迹 writer：ch4(pos) + ctx.tf_cache(head quat) -> head_pose_trajectory.npy (N,7)。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..decoders import HEAD_FRAME_ID, TRAJ_ENTITY_ID
from .base import ParseContext, Writer

TRAJ_TOPIC = ("/pose_algo_body/head_pose_trajectory",)


class HeadTrajectoryWriter(Writer):
    name = "head_trajectory"
    topics = TRAJ_TOPIC
    schema_name = "foxglove.SceneUpdate"

    def feed(self, topic: str, msg, ctx: ParseContext) -> None:
        obj = self._decode(msg.data)
        pos = None
        for e in obj.entities:
            if e.id != TRAJ_ENTITY_ID:
                continue
            if e.spheres:
                s = e.spheres[0]
                pos = (float(s.pose.position.x), float(s.pose.position.y), float(s.pose.position.z))
            break
        ctx.head_pos.append(
            np.array(pos if pos is not None else [np.nan] * 3, dtype=np.float64)
        )
        ctx.head_ts.append(msg.publish_time)

    def finalize(self, out_dir: Path, ctx: ParseContext) -> None:
        n = len(ctx.head_pos)
        if n == 0:
            return
        pos = np.stack(ctx.head_pos)  # (N,3)
        quat = np.full((n, 4), np.nan)
        for i in range(n):
            fi = i if i < len(ctx.tf_cache) else len(ctx.tf_cache) - 1
            if 0 <= fi < len(ctx.tf_cache):
                frame = ctx.tf_cache[fi]
                if HEAD_FRAME_ID in frame:
                    quat[i] = frame[HEAD_FRAME_ID][2]
        out = np.concatenate([pos, quat], axis=1)  # (N,7)
        np.save(out_dir / "head_pose_trajectory.npy", out)
        ctx.outputs["head_pose_trajectory.npy"] = {
            "frames": n,
            "timestamps_ns": list(ctx.head_ts),
        }
