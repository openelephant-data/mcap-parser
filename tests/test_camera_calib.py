import json

from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.camera_calib import CameraCalibWriter
from mcap_parser._testing import calib_msg, depth_calib_msg


def test_camera_calib_writes_5_groups(tmp_path):
    w = CameraCalibWriter()
    ctx = ParseContext()
    topics = [
        "/head_left_camera_params/head_left_camera_undistorted/camera_info",
        "/head_left_camera_params/head_left_camera_raw/camera_info",
        "/head_right_camera_params/head_right_camera_undistorted/camera_info",
        "/head_right_camera_params/head_right_camera_raw/camera_info",
        "/head_depth_camera_params/head_depth_camera/camera_info",
    ]
    for t in topics[:4]:
        side = "left" if "head_left" in t else "right"
        variant = "undistorted" if "undistorted" in t else "raw"
        w.feed(t, calib_msg(side, variant), ctx)
    # 深度标定走专用分支（head_depth_camera）
    w.feed(topics[4], depth_calib_msg(), ctx)
    # 重复喂一帧，确认只存首帧
    w.feed(topics[0], calib_msg("left", "undistorted", ts=2), ctx)
    w.finalize(tmp_path, ctx)

    data = json.loads((tmp_path / "camera_calibration.json").read_text())
    assert set(data.keys()) == {
        "left_undistorted", "left_raw", "right_undistorted", "right_raw",
        "head_depth_camera",
    }
    assert len(data["left_undistorted"]["K"]) == 9
    assert len(data["left_undistorted"]["P"]) == 12
    assert data["left_undistorted"]["frame_id"] == "head_left_camera"
    assert data["head_depth_camera"]["frame_id"] == "head_depth_camera"
    assert data["head_depth_camera"]["width"] == 320
    assert ctx.outputs["camera_calibration.json"]["groups"] == sorted(data.keys())
