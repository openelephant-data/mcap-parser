"""相机标定 writer：ch11-14 -> camera_calibration.json。"""
from __future__ import annotations

import json
from pathlib import Path

from .base import ParseContext, Writer

CALIB_TOPICS = (
    "/head_left_camera_params/",
    "/head_right_camera_params/",
    "/head_depth_camera_params/",
)


class CameraCalibWriter(Writer):
    name = "camera_calib"
    topics = CALIB_TOPICS
    schema_name = "foxglove.CameraCalibration"

    def feed(self, topic: str, msg, ctx: ParseContext) -> None:
        obj = self._decode(msg.data)
        if "head_depth" in topic:
            key = "head_depth_camera"
        else:
            side = "left" if "head_left" in topic else "right"
            variant = "undistorted" if "undistorted" in topic else "raw"
            key = f"{side}_{variant}"
        if key in ctx.calib:
            return  # 每帧相同，只存首帧
        ctx.calib[key] = {
            "width": int(obj.width),
            "height": int(obj.height),
            "distortion_model": obj.distortion_model,
            "K": list(obj.K),
            "D": list(obj.D),
            "R": list(obj.R),
            "P": list(obj.P),
            "frame_id": obj.frame_id,
        }

    def finalize(self, out_dir: Path, ctx: ParseContext) -> None:
        if not ctx.calib:
            return
        path = out_dir / "camera_calibration.json"
        path.write_text(
            json.dumps(ctx.calib, indent=2, ensure_ascii=False)
        )
        ctx.outputs["camera_calibration.json"] = {
            "frames": 1,
            "groups": sorted(ctx.calib.keys()),
        }
