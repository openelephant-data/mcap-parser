"""CLI: python -m mcap_parser <mcap> --out_dir <dir> [--side ...] [--streams ...]"""
from __future__ import annotations

from .reader import McapFile


def main(
    mcap_path: str,
    out_dir: str,
    side: str = "both",
    streams: str | None = None,
    per_frame_pcd: bool = True,
    overwrite: bool = False,
) -> None:
    stream_list = streams.split(",") if isinstance(streams, str) else streams
    ctx = McapFile(mcap_path).parse(
        out_dir,
        side=side,
        streams=stream_list,
        per_frame_pcd=per_frame_pcd,
        overwrite=overwrite,
    )
    print(f"[mcap_parser] done: {len(ctx.outputs)} outputs in {out_dir}")
