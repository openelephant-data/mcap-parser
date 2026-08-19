"""视频 writer：ch9/10 -> head_{left,right}_camera_undistorted.mp4（ffmpeg remux，失败退 .h264）。"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .base import ParseContext, Writer

VIDEO_TOPICS = ("/head_left_camera_undistorted/", "/head_right_camera_undistorted/")


class VideoWriter(Writer):
    name = "video"
    topics = VIDEO_TOPICS
    schema_name = "foxglove.CompressedVideo"

    def __init__(self, side_filter: str = "both"):
        self.side_filter = side_filter
        self._frames: dict[str, list[tuple[bytes, int]]] = {}

    def feed(self, topic: str, msg, ctx: ParseContext) -> None:
        side = "left" if "head_left" in topic else "right"
        if self.side_filter != "both" and side != self.side_filter:
            return
        obj = self._decode(msg.data)
        self._frames.setdefault(side, []).append((bytes(obj.data), msg.publish_time))

    def finalize(self, out_dir: Path, ctx: ParseContext) -> None:
        tmp = out_dir / ".tmp"
        tmp.mkdir(exist_ok=True)
        for side, frames in self._frames.items():
            if not frames:
                continue
            h264 = tmp / f"{side}.h264"
            with open(h264, "wb") as f:
                for data, _ts in frames:
                    f.write(data)
            mp4 = out_dir / f"head_{side}_camera_undistorted.mp4"
            ts0, ts1 = frames[0][1], frames[-1][1]
            fps = (len(frames) - 1) / ((ts1 - ts0) / 1e9) if (ts1 > ts0 and len(frames) > 1) else 30.0
            muxed = False
            if shutil.which("ffmpeg"):
                r = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(h264), "-c", "copy",
                     "-r", f"{fps:.3f}", "-f", "mp4", str(mp4)],
                    capture_output=True,
                )
                muxed = r.returncode == 0
            if muxed:
                h264.unlink()
            else:
                # 退化为 .h264
                fallback = out_dir / f"{side}.h264"
                h264.replace(fallback)
                mp4 = fallback  # 输出名同步
            ctx.outputs[f"head_{side}_camera_undistorted.mp4"] = {
                "frames": len(frames),
                "fps": round(fps, 3),
                "muxed": muxed,
                "timestamps_ns": [t for _, t in frames],
            }
        # 清理空 .tmp
        if tmp.exists() and not any(tmp.iterdir()):
            tmp.rmdir()
