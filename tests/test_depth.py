import numpy as np

from mcap_parser._testing import raw_image_msg
from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.depth import DepthWriter


def _decode_raw(msg):
    from foxglove_schemas_protobuf import RawImage_pb2
    obj = RawImage_pb2.RawImage()
    obj.ParseFromString(msg.data)
    return obj


def test_depth_writes_npz(tmp_path):
    w = DepthWriter()
    ctx = ParseContext()
    msgs = [raw_image_msg(ts=1_000_000_000 + i * 33_000_000, n=4) for i in range(3)]
    for m in msgs:
        w.feed("/head_depth_camera/depth/image", m, ctx)
    w.finalize(tmp_path, ctx)

    z = np.load(tmp_path / "head_depth_camera.npz")
    assert z["depth"].shape == (3, 4, 4)
    assert z["depth"].dtype == np.uint16
    assert z["timestamps_ns"].shape == (3,)
    assert z["frame_indices"].tolist() == [0, 1, 2]
    assert int(z["width"]) == 4 and int(z["height"]) == 4
    assert int(z["step"]) == 8

    # depth 内容与原 RawImage.data 一致（原值堆叠，无转码）
    for i, m in enumerate(msgs):
        obj = _decode_raw(m)
        ref = np.frombuffer(obj.data, dtype=np.uint16).reshape(4, 4)
        np.testing.assert_array_equal(z["depth"][i], ref)

    # manifest 条目
    entry = ctx.outputs["head_depth_camera.npz"]
    assert entry["frames"] == 3
    assert entry["encoding"] == "mono16"
    assert entry["dtype"] == "uint16"
    assert len(entry["timestamps_ns"]) == 3


def test_depth_no_frames_no_output(tmp_path):
    w = DepthWriter()
    ctx = ParseContext()
    w.finalize(tmp_path, ctx)
    assert "head_depth_camera.npz" not in ctx.outputs
    assert not (tmp_path / "head_depth_camera.npz").exists()


def test_depth_frames_distinguishable(tmp_path):
    """帧间内容应不同，确认堆叠保留逐帧差异。"""
    w = DepthWriter()
    ctx = ParseContext()
    for i in range(3):
        w.feed("/head_depth_camera/depth/image",
               raw_image_msg(ts=1_000_000_000 + i * 33_000_000, n=4), ctx)
    w.finalize(tmp_path, ctx)
    z = np.load(tmp_path / "head_depth_camera.npz")
    frames = [z["depth"][i].tobytes() for i in range(3)]
    assert len(set(frames)) == 3


def test_depth_timestamps_match_feed(tmp_path):
    w = DepthWriter()
    ctx = ParseContext()
    tss = [1_000_000_000 + i * 33_000_000 for i in range(3)]
    for t in tss:
        w.feed("/head_depth_camera/depth/image", raw_image_msg(ts=t, n=4), ctx)
    w.finalize(tmp_path, ctx)
    z = np.load(tmp_path / "head_depth_camera.npz")
    assert z["timestamps_ns"].tolist() == tss
