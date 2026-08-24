"""深度相机 writer：ch15 -> head_depth_camera.npz（mono16 堆叠 + 索引）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import ParseContext, Writer

DEPTH_TOPICS = ("/head_depth_camera/depth/image",)


class DepthWriter(Writer):
    name = "depth"
    topics = DEPTH_TOPICS
    schema_name = "foxglove.RawImage"

    def __init__(self) -> None:
        self._frames: list[tuple[bytes, int]] = []
        self._meta: dict = {}

    def feed(self, topic: str, msg, ctx: ParseContext) -> None:
        obj = self._decode(msg.data)
        self._frames.append((bytes(obj.data), msg.publish_time))
        if not self._meta:  # 首帧元信息（width/height/encoding/step 全帧一致）
            self._meta = {
                "width": int(obj.width),
                "height": int(obj.height),
                "encoding": obj.encoding,
                "step": int(obj.step),
            }

    def finalize(self, out_dir: Path, ctx: ParseContext) -> None:
        if not self._frames:
            return
        w = self._meta["width"]
        h = self._meta["height"]
        bpr = self._meta["step"]  # bytes per row（含可能的行填充）
        n = len(self._frames)
        depth = np.empty((n, h, w), dtype=np.uint16)
        ts = np.empty(n, dtype=np.int64)
        for i, (data, t) in enumerate(self._frames):
            # 按 step 切行，取每行前 w*2 字节为 w 个 uint16 像素（兼容行填充）
            raw = np.frombuffer(data, dtype=np.uint8).reshape(h, bpr)
            depth[i] = raw[:, : w * 2].copy().view(np.uint16).reshape(h, w)
            ts[i] = t
        idx = np.arange(n, dtype=np.int64)
        path = out_dir / "head_depth_camera.npz"
        np.savez_compressed(
            path,
            depth=depth,
            timestamps_ns=ts,
            frame_indices=idx,
            width=np.array(w),
            height=np.array(h),
            step=np.array(bpr),
        )
        ctx.outputs["head_depth_camera.npz"] = {
            "frames": n,
            "width": w,
            "height": h,
            "encoding": self._meta["encoding"],
            "step": bpr,
            "dtype": "uint16",
            "timestamps_ns": [t for _, t in self._frames],
        }
