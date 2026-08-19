"""身体/手根 tf writer：ch1 -> tracking_data_tf.jsonl + wrist_poses.npz。

feed 阶段把每帧全部 transforms 写入 ctx.tf_cache（供 head_trajectory/pointcloud 跨流用）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..decoders import HEAD_FRAME_ID, WRIST_FRAME_IDS
from .base import ParseContext, Writer

TF_TOPIC = ("/pose_algo_body/tf_tree",)


class BodyTfWriter(Writer):
    name = "body_tf"
    topics = TF_TOPIC
    schema_name = "foxglove.FrameTransforms"

    def feed(self, topic: str, msg, ctx: ParseContext) -> None:
        obj = self._decode(msg.data)
        ts = msg.publish_time
        frame: dict[str, tuple] = {}
        for t in obj.transforms:
            tr = np.array(
                [float(t.translation.x), float(t.translation.y), float(t.translation.z)],
                dtype=np.float64,
            )
            qu = np.array(
                [float(t.rotation.x), float(t.rotation.y),
                 float(t.rotation.z), float(t.rotation.w)],
                dtype=np.float64,
            )
            frame[t.child_frame_id] = (t.parent_frame_id, tr, qu, ts)
        ctx.tf_cache.append(frame)
        ctx.tf_ts.append(ts)

    def finalize(self, out_dir: Path, ctx: ParseContext) -> None:
        n = len(ctx.tf_cache)
        if n == 0:
            return

        # tracking_data_tf.jsonl
        with open(out_dir / "tracking_data_tf.jsonl", "w") as f:
            for i, frame in enumerate(ctx.tf_cache):
                ts = ctx.tf_ts[i]
                transforms = [
                    {
                        "parent": v[0],
                        "child": child,
                        "translation": v[1].tolist(),
                        "rotation": v[2].tolist(),
                    }
                    for child, v in frame.items()
                ]
                f.write(json.dumps({"ts_ns": int(ts), "transforms": transforms}) + "\n")
        ctx.outputs["tracking_data_tf.jsonl"] = {
            "frames": n,
            "transforms_per_frame": len(ctx.tf_cache[0]),
        }

        # wrist_poses.npz
        left = np.full((n, 7), np.nan)
        right = np.full((n, 7), np.nan)
        tss = np.array(ctx.tf_ts, dtype=np.int64)
        for i, frame in enumerate(ctx.tf_cache):
            for side, wid in (("left", WRIST_FRAME_IDS[0]), ("right", WRIST_FRAME_IDS[1])):
                arr = left if side == "left" else right
                if wid in frame:
                    arr[i] = np.concatenate([frame[wid][1], frame[wid][2]])
        np.savez(
            out_dir / "wrist_poses.npz", left=left, right=right, timestamps=tss,
        )
        ctx.outputs["wrist_poses.npz"] = {
            "frames": n,
            "timestamps_ns": tss.tolist(),
        }
