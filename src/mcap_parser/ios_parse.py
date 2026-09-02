"""iPhone/oeego ARKit foxglove MCAP -> 7 文件契约(对齐 openelephant-data-app IosMcapParseTask)。

单遍 iter_messages,按 schema 名 + topic 路由到累加器,最后写:
video_head.mp4 / depth_frames.npz / confidence_frames.npz / camera_info.json /
transforms.npz / hand_keypoints_3d.npz / parse_report.json
"""
from __future__ import annotations

import base64
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from foxglove_schemas_protobuf import (
    CameraCalibration_pb2,
    CompressedImage_pb2,
    FrameTransforms_pb2,
    PointCloud_pb2,
    RawImage_pb2,
)
from mcap.reader import make_reader

from .decoders import decode_pointcloud, oeego_msg_class

OUTPUT_DESCRIPTIONS = {
    "parse_report": "iPhone MCAP 各通道解析摘要",
    "video_head": "H265/H264 视频流 remux 为 mp4",
    "camera_info": "相机内参/畸变 (CameraInfo)",
    "transforms": "每帧 world→child 6DoF (TF2Message)",
    "hand_keypoints_3d": "3D 手部关键点 (EgoHandPoseFrame 3D)",
    "depth_frames": "原始深度帧 (16UC1/32)",
    "confidence_frames": "置信度帧 (8UC1)",
}


def _flat(xs) -> list[float]:
    return [float(x) for x in xs]


class IosMcapParser:
    """单遍 demux iPhone foxglove MCAP。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    # -- 累加器 ---------------------------------------------------------
    @staticmethod
    def _new_accum():
        return {
            "video_jpegs": [],        # list[(log_time_ns, bytes)]
            "video_nals": [],         # list[(log_time_ns, bytes)]  (hevc 分支)
            "depth": [],              # list[(log_time_ns, bytes, fmt)]
            "conf": [],               # list[(log_time_ns, bytes, fmt)]
            "camera_infos": {},        # frame_id -> dict
            "transforms": [],         # list[(tx,ty,tz,qx,qy,qz,qw,ts_ns)]
            "hand_3d": [],            # list[(pts(K,3) float32, ts_ns)]
            "oc_fps": 0,              # onboard_config.target_frame_rate
            "case_id": "",            # onboard_config.mission_id
            "summary": defaultdict(lambda: {
                "count": 0, "bytes": 0, "t0": None, "t1": None,
                "schema": "", "topic": "", "sample": None,
            }),
        }

    # -- onboard_config 类解析 -----------------------------------------
    @staticmethod
    def _resolve_onboard_cls(chan_info):
        for cid, (topic, gname, sdata) in chan_info.items():
            if gname.endswith("OnboardConfig"):
                try:
                    return oeego_msg_class(gname, sdata)
                except Exception:
                    return None
        return None

    # -- 单条消息分派 ---------------------------------------------------
    @staticmethod
    def _feed(channel, msg, chan_info, oc_cls, accum):
        cid = channel.id
        topic, gname, _ = chan_info[cid]
        s = accum["summary"][cid]
        s["topic"] = topic
        s["schema"] = gname
        s["count"] += 1
        s["bytes"] += len(msg.data)
        if s["t0"] is None or msg.log_time < s["t0"]:
            s["t0"] = msg.log_time
        if s["t1"] is None or msg.log_time > s["t1"]:
            s["t1"] = msg.log_time

        data = msg.data
        if gname == "foxglove.CompressedImage" and "/color/" in topic:
            m = CompressedImage_pb2.CompressedImage()
            m.ParseFromString(data)
            if s["sample"] is None:
                s["sample"] = {"format": str(m.format), "data_len": len(m.data)}
            head = bytes(m.data[:4])
            if head[:2] == b"\xff\xd8":
                accum["video_jpegs"].append((msg.log_time, bytes(m.data)))
            elif head[:3] in (b"\x00\x00\x01",) or head[:4] == b"\x00\x00\x00\x01":
                accum["video_nals"].append((msg.log_time, bytes(m.data)))
            # 既非 jpeg 也非 NAL:跳过
        elif gname == "foxglove.RawImage" and topic.endswith("/image_raw"):
            m = RawImage_pb2.RawImage()
            m.ParseFromString(data)
            fmt = f"{m.encoding};{m.width}x{m.height}"
            if s["sample"] is None:
                s["sample"] = {"width": int(m.width), "height": int(m.height),
                               "encoding": str(m.encoding), "data_len": len(m.data)}
            accum["depth"].append((msg.log_time, bytes(m.data), fmt))
        elif gname == "foxglove.RawImage" and topic.endswith("/confidence"):
            m = RawImage_pb2.RawImage()
            m.ParseFromString(data)
            fmt = f"{m.encoding};{m.width}x{m.height}"
            if s["sample"] is None:
                s["sample"] = {"width": int(m.width), "height": int(m.height),
                               "encoding": str(m.encoding), "data_len": len(m.data)}
            accum["conf"].append((msg.log_time, bytes(m.data), fmt))
        elif gname == "foxglove.CameraCalibration" and topic.endswith("/camera_info"):
            m = CameraCalibration_pb2.CameraCalibration()
            m.ParseFromString(data)
            k = _flat(m.K)
            accum["camera_infos"][str(m.frame_id)] = {
                "frame_id": str(m.frame_id),
                "width": int(m.width),
                "height": int(m.height),
                "distortion_model": str(m.distortion_model),
                "D": _flat(m.D),
                "K": k,
                "R": _flat(m.R),
                "P": _flat(m.P),
            }
            if s["sample"] is None:
                s["sample"] = {
                    "width": int(m.width), "height": int(m.height),
                    "distortion_model": str(m.distortion_model),
                    "fx": k[0] if len(k) > 0 else None,
                    "fy": k[4] if len(k) > 4 else None,
                    "cx": k[2] if len(k) > 2 else None,
                    "cy": k[5] if len(k) > 5 else None,
                }
        elif gname == "foxglove.FrameTransforms" and topic == "/tf":
            m = FrameTransforms_pb2.FrameTransforms()
            m.ParseFromString(data)
            if s["sample"] is None:
                tr = list(m.transforms)
                first = f"{tr[0].parent_frame_id}->{tr[0].child_frame_id}" if tr else None
                s["sample"] = {"transforms_per_frame": len(tr), "first": first}
            for t in m.transforms:
                ts_ns = int(t.timestamp.seconds) * 1_000_000_000 + int(t.timestamp.nanos)
                accum["transforms"].append((
                    float(t.translation.x), float(t.translation.y), float(t.translation.z),
                    float(t.rotation.x), float(t.rotation.y), float(t.rotation.z),
                    float(t.rotation.w), ts_ns,
                ))
        elif gname == "foxglove.PointCloud" and topic.endswith("/pointcloud_3d"):
            m = PointCloud_pb2.PointCloud()
            m.ParseFromString(data)
            pts, _ = decode_pointcloud(m)
            pts = pts.astype(np.float32)
            if s["sample"] is None:
                s["sample"] = {"points": int(len(pts)), "frame_id": str(m.frame_id)}
            ts_ns = int(m.timestamp.seconds) * 1_000_000_000 + int(m.timestamp.nanos)
            accum["hand_3d"].append((pts, ts_ns))
        elif gname.endswith("OnboardConfig") and oc_cls is not None:
            m = oc_cls()
            m.ParseFromString(data)
            if accum["oc_fps"] == 0:
                tfr = int(getattr(m, "target_frame_rate", 0) or 0)
                if tfr > 0:
                    accum["oc_fps"] = tfr
            if not accum["case_id"]:
                mid = getattr(m, "mission_id", "")
                if mid:
                    accum["case_id"] = str(mid)

    # -- parse_report ---------------------------------------------------
    @staticmethod
    def _channel_entries(summary):
        entries = []
        for cid in sorted(summary):
            s = summary[cid]
            t0, t1 = s["t0"], s["t1"]
            span_s = (t1 - t0) / 1e9 if (t0 is not None and t1 is not None) else 0.0
            hz = s["count"] / span_s if span_s > 0 else 0.0
            entries.append({
                "channel_id": cid, "topic": s["topic"], "schema": s["schema"],
                "count": s["count"], "bytes": s["bytes"],
                "span_s": round(span_s, 3), "hz": round(hz, 3), "sample": s["sample"],
            })
        return entries

    def _build_report(self, accum, outputs):
        counts = {
            "camera_infos": list(accum["camera_infos"].keys()),
            "transform_count": len(accum["transforms"]),
            "hand_3d_count": len(accum["hand_3d"]),
            "hand_2d_count": 0,
            "video_nal_count": len(accum["video_jpegs"]) + len(accum["video_nals"]),
            "depth_frame_count": len(accum["depth"]),
            "confidence_frame_count": len(accum["conf"]),
        }
        out_entries = []
        for key, path in outputs.items():
            e = {"key": key, "path": str(path)}
            desc = OUTPUT_DESCRIPTIONS.get(key)
            if desc:
                e["desc"] = desc
            out_entries.append(e)
        case_id = accum["case_id"] or self._case_id_from_name()
        return {"case_id": case_id, "outputs": out_entries,
                "channels": self._channel_entries(accum["summary"]), **counts}

    def _case_id_from_name(self) -> str:
        """从文件名 SN_..._<digits>_... 抽 case_id;失败返回空串。"""
        name = self.path.stem
        import re
        m = re.search(r"_(\d{15,})_", name) or re.search(r"(\d{15,})", name)
        return m.group(1) if m else ""

    # -- 入口 -----------------------------------------------------------
    def parse(self, out_dir: str | Path, *, overwrite: bool = False,
              keep_frames_on_fail: bool = False) -> dict:
        out_dir = Path(out_dir)
        if out_dir.exists() and any(out_dir.iterdir()):
            if not overwrite:
                raise RuntimeError(f"out_dir not empty: {out_dir}(用 overwrite=True 覆盖)")
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        f = open(self.path, "rb")
        try:
            r = make_reader(f)
            summary = r.get_summary()
            chan_info = {
                cid: (ch.topic, summary.schemas[ch.schema_id].name,
                      summary.schemas[ch.schema_id].data)
                for cid, ch in summary.channels.items()
            }
            oc_cls = self._resolve_onboard_cls(chan_info)
            accum = self._new_accum()
            for schema, channel, msg in r.iter_messages():
                self._feed(channel, msg, chan_info, oc_cls, accum)
            outputs: dict[str, Path] = {}
            # 各产物写入在后续 Task 逐步填充;本 Task 先只写 parse_report
            report = self._build_report(accum, outputs)
            (out_dir / "parse_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2))
            outputs["parse_report"] = out_dir / "parse_report.json"
        finally:
            f.close()
        return outputs
