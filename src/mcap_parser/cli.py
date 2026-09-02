"""CLI: python -m mcap_parser <mcap> --out_dir <dir> [--profile ...]。

--profile {auto,iphone,pico} 默认 auto:iPhone 文件(topic 含 /cam_head_l/)走
IosMcapParser,否则走 PICO McapFile。
"""
from __future__ import annotations

from pathlib import Path

from mcap.reader import make_reader

from .ios_parse import IosMcapParser
from .reader import McapFile


def _detect_profile(mcap_path: str) -> str:
    with open(mcap_path, "rb") as f:
        s = make_reader(f).get_summary()
    topics = {ch.topic for ch in s.channels.values()}
    if any(t.startswith("/cam_head_l/") for t in topics):
        return "iphone"
    return "pico"


def main(
    mcap_path: str,
    out_dir: str,
    side: str = "both",
    streams: str | None = None,
    per_frame_pcd: bool = True,
    profile: str = "auto",
    keep_frames_on_fail: bool = False,
    overwrite: bool = False,
) -> None:
    resolved = profile if profile != "auto" else _detect_profile(mcap_path)
    if resolved == "iphone":
        outs = IosMcapParser(mcap_path).parse(
            out_dir, overwrite=overwrite, keep_frames_on_fail=keep_frames_on_fail)
        print(f"[mcap_parser] iphone done: {len(outs)} outputs in {out_dir}")
        return
    stream_list = streams.split(",") if isinstance(streams, str) else streams
    ctx = McapFile(mcap_path).parse(
        out_dir, side=side, streams=stream_list,
        per_frame_pcd=per_frame_pcd, overwrite=overwrite,
    )
    print(f"[mcap_parser] pico done: {len(ctx.outputs)} outputs in {out_dir}")
