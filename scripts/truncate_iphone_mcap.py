"""通用按时间截断 mcap(支持 foxglove + oeego 自定义 schema)。

用法: python scripts/truncate_iphone_mcap.py <in.mcap> <out.mcap> [seconds=30]

与 scripts/truncate_mcap.py 的区别:不依赖特定 topic 计帧,按 log_time 时间窗截断;
用原 schema data(summary.schemas[].data)而非 schema_data(name),故支持 oeego 自定义 schema。
给 iPhone head_pose_mcap.mcap 的集成测试用。
"""
from __future__ import annotations

import sys

from mcap.reader import make_reader
from mcap.writer import Writer


def truncate(in_path: str, out_path: str, seconds: int = 30) -> None:
    fin = open(in_path, "rb")
    r = make_reader(fin)
    summary = r.get_summary()
    schema_cache: dict[str, int] = {}
    channel_cache: dict[int, int] = {}
    fout = open(out_path, "wb")
    w = Writer(fout)
    w.start()
    t0 = None
    cutoff = None
    for schema, channel, msg in r.iter_messages():
        if t0 is None:
            t0 = msg.log_time
            cutoff = t0 + int(seconds) * 1_000_000_000
        if msg.log_time > cutoff:
            break
        sch = summary.schemas[channel.schema_id]
        sname = sch.name
        sdata = sch.data  # 原样 FileDescriptorSet,支持 oeego
        if sname not in schema_cache:
            schema_cache[sname] = w.register_schema(
                name=sname, encoding="protobuf", data=sdata)
        if channel.id not in channel_cache:
            channel_cache[channel.id] = w.register_channel(
                topic=channel.topic, message_encoding="protobuf",
                schema_id=schema_cache[sname])
        w.add_message(
            channel_id=channel_cache[channel.id], log_time=msg.log_time,
            data=msg.data, publish_time=msg.publish_time)
    w.finish()
    fout.close()
    fin.close()


if __name__ == "__main__":
    truncate(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 30)
