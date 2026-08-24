import numpy as np

from mcap_parser._testing import raw_image_msg
from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.depth import DepthWriter


def test_depth_writes_per_frame_bin(tmp_path):
    w = DepthWriter()
    ctx = ParseContext()
    msgs = [raw_image_msg(ts=1_000_000_000 + i * 33_000_000, n=4) for i in range(3)]
    for m in msgs:
        w.feed("/head_depth_camera/depth/image", m, ctx)
    w.finalize(tmp_path, ctx)

    depth_dir = tmp_path / "depth"
    files = sorted(p.name for p in depth_dir.iterdir())
    assert files == ["frame_000000.bin", "frame_000001.bin", "frame_000002.bin"]

    # 每帧 mono16 裸字节：4*4 像素 * 2 字节 = 32 字节
    for p in depth_dir.iterdir():
        assert p.stat().st_size == 32

    # 字节内容与原 RawImage.data 一致（逐帧原样写出）
    for i, m in enumerate(msgs):
        from foxglove_schemas_protobuf import RawImage_pb2
        obj = RawImage_pb2.RawImage()
        obj.ParseFromString(m.data)
        assert (depth_dir / f"frame_{i:06d}.bin").read_bytes() == obj.data

    # manifest 条目
    entry = ctx.outputs["depth/"]
    assert entry["frames"] == 3
    assert entry["width"] == 4 and entry["height"] == 4
    assert entry["encoding"] == "mono16"
    assert entry["step"] == 8
    assert len(entry["timestamps_ns"]) == 3


def test_depth_no_frames_no_output(tmp_path):
    w = DepthWriter()
    ctx = ParseContext()
    w.finalize(tmp_path, ctx)
    assert "depth/" not in ctx.outputs
    assert not (tmp_path / "depth").exists()


def test_depth_pixels_differ_per_frame(tmp_path):
    """帧间内容应不同，确认逐帧顺序未被压缩成同一帧。"""
    w = DepthWriter()
    ctx = ParseContext()
    for i in range(3):
        w.feed("/head_depth_camera/depth/image",
               raw_image_msg(ts=1_000_000_000 + i * 33_000_000, n=4), ctx)
    w.finalize(tmp_path, ctx)
    blobs = [(tmp_path / "depth" / f"frame_{i:06d}.bin").read_bytes()
             for i in range(3)]
    assert len(set(blobs)) == 3
