"""截断基准 head_pose_mcap.mcap 前 max_frames 帧，给集成测试用。

用法: python scripts/truncate_mcap.py <in.mcap> <out.mcap> [max_frames=300]
"""
from __future__ import annotations

import sys
from pathlib import Path

from mcap.reader import make_reader
from mcap.writer import Writer

from mcap_parser.decoders import schema_data


def truncate(in_path: str, out_path: str, max_frames: int = 300) -> None:
    fin = open(in_path, "rb")
    r = make_reader(fin)
    summary = r.get_summary()
    schema_cache: dict[str, int] = {}
    channel_cache: dict[int, int] = {}
    fout = open(out_path, "wb")
    w = Writer(fout)
    w.start()
    video_chs = {
        cid for cid, ch in summary.channels.items()
        if "head_left_camera_undistorted" in ch.topic and ch.topic.endswith("/image")
    }
    frame = 0
    for schema, channel, msg in r.iter_messages():
        if channel.id in video_chs:
            if frame >= max_frames:
                break
            frame += 1
        elif frame >= max_frames:
            continue
        sname = summary.schemas[channel.schema_id].name
        if sname not in schema_cache:
            schema_cache[sname] = w.register_schema(
                name=sname, encoding="protobuf", data=schema_data(sname)
            )
        old_cid = channel.id
        if old_cid not in channel_cache:
            channel_cache[old_cid] = w.register_channel(
                topic=channel.topic, message_encoding="protobuf",
                schema_id=schema_cache[sname],
            )
        w.add_message(
            channel_id=channel_cache[old_cid],
            log_time=msg.log_time, data=msg.data, publish_time=msg.publish_time,
        )
    w.finish()
    fout.close()
    fin.close()


if __name__ == "__main__":
    truncate(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 300)
