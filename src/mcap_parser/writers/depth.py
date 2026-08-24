"""深度相机 writer：ch15 -> depth/frame_{i:06d}.bin（mono16 裸流，逐帧原样写出）。"""
from __future__ import annotations

from pathlib import Path

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
        depth_dir = out_dir / "depth"
        depth_dir.mkdir(exist_ok=True)
        for i, (data, _ts) in enumerate(self._frames):
            (depth_dir / f"frame_{i:06d}.bin").write_bytes(data)
        ctx.outputs["depth/"] = {
            "frames": len(self._frames),
            **self._meta,
            "timestamps_ns": [t for _, t in self._frames],
        }
