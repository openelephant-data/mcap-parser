from mcap_parser._testing import video_msg
from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.video import VideoWriter


def test_video_writer_concat_and_manifest(tmp_path):
    w = VideoWriter()
    ctx = ParseContext()
    for i in range(3):
        ts = 1_000_000_000 + i * 33_333_333
        w.feed("/head_left_camera_undistorted/head_left_camera_undistorted/image",
               video_msg(ts, "left"), ctx)
        w.feed("/head_right_camera_undistorted/head_right_camera_undistorted/image",
               video_msg(ts, "right"), ctx)
    w.finalize(tmp_path, ctx)

    # 伪 NAL 非 H.264，ffmpeg 必失败 -> 退化为 .h264，manifest 标 muxed=False
    for side in ("left", "right"):
        entry = ctx.outputs[f"head_{side}_camera_undistorted.mp4"]
        assert entry["frames"] == 3
        assert entry["muxed"] is False
        assert (tmp_path / f"{side}.h264").exists()


def test_video_side_filter(tmp_path):
    w = VideoWriter(side_filter="left")
    ctx = ParseContext()
    ts = 1_000_000_000
    w.feed("/head_left_camera_undistorted/head_left_camera_undistorted/image",
           video_msg(ts, "left"), ctx)
    w.feed("/head_right_camera_undistorted/head_right_camera_undistorted/image",
           video_msg(ts, "right"), ctx)
    w.finalize(tmp_path, ctx)
    assert "head_left_camera_undistorted.mp4" in ctx.outputs
    assert "head_right_camera_undistorted.mp4" not in ctx.outputs
