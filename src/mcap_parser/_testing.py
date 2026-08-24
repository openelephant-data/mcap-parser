"""测试用消息构造器与合成 MCAP 构建器（属于包内，避免与 site-packages 的 tests 包冲突）。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from foxglove_schemas_protobuf import (
    CameraCalibration_pb2,
    CompressedVideo_pb2,
    FrameTransforms_pb2,
    PointCloud_pb2,
    RawImage_pb2,
    SceneUpdate_pb2,
)
from mcap.writer import Writer

from .decoders import schema_data

BASE_TS = 1_000_000_000  # ns
STEP = 33_333_333  # ~30fps


@dataclass
class FakeMsg:
    data: bytes
    publish_time: int


def _pts():
    return np.random.rand(21, 3).astype(np.float32)


def calib_msg(side: str, variant: str, ts: int = BASE_TS) -> FakeMsg:
    m = CameraCalibration_pb2.CameraCalibration()
    m.timestamp.seconds = ts // 1_000_000_000
    m.timestamp.nanos = ts % 1_000_000_000
    m.width, m.height = 1920, 1456
    m.distortion_model = "plumb_bob"
    m.D.extend([0.0] * 5)
    m.K.extend([1000.0, 0, 960, 0, 1000, 720, 0, 0, 1])
    m.R.extend([1, 0, 0, 0, 1, 0, 0, 0, 1])
    m.P.extend([1000, 0, 960, 0, 0, 1000, 720, 0, 0, 0, 1, 0])
    m.frame_id = f"head_{side}_camera"
    return FakeMsg(m.SerializeToString(), ts)


def depth_calib_msg(ts: int = BASE_TS) -> FakeMsg:
    m = CameraCalibration_pb2.CameraCalibration()
    m.timestamp.seconds = ts // 1_000_000_000
    m.timestamp.nanos = ts % 1_000_000_000
    m.width, m.height = 320, 240
    m.distortion_model = "plumb_bob"
    m.D.extend([0.0] * 5)
    m.K.extend([320, 0, 160, 0, 320, 120, 0, 0, 1])
    m.R.extend([1, 0, 0, 0, 1, 0, 0, 0, 1])
    m.P.extend([320, 0, 160, 0, 0, 320, 120, 0, 0, 0, 1, 0])
    m.frame_id = "head_depth_camera"
    return FakeMsg(m.SerializeToString(), ts)


def raw_image_msg(ts: int = BASE_TS, n: int = 4) -> FakeMsg:
    """构造一个 mono16 深度图（n×n 像素，每像素 2 字节，step = n*2）。"""
    m = RawImage_pb2.RawImage()
    m.timestamp.seconds = ts // 1_000_000_000
    m.timestamp.nanos = ts % 1_000_000_000
    m.frame_id = "head_depth_camera"
    m.width, m.height = n, n
    m.encoding = "mono16"
    m.step = n * 2
    # 像素随 ts 偏移的 ramp，任意 ts 步长下帧间字节均不同
    vals = (np.arange(n * n, dtype=np.uint16) + (ts % 65536)).astype(np.uint16)
    m.data = vals.tobytes()
    return FakeMsg(m.SerializeToString(), ts)


def hand_kp_3d_msg(ts: int = BASE_TS, left: bool = True, right: bool = True) -> FakeMsg:
    m = SceneUpdate_pb2.SceneUpdate()
    for side_name, is_right in (("left", False), ("right", True)):
        if (side_name == "left" and not left) or (side_name == "right" and not right):
            continue
        e = m.entities.add()
        e.frame_id = "world"
        e.id = f"{side_name}_hand_skeleton"
        e.frame_locked = True
        pts = _pts()
        for r in range(21):
            s = e.spheres.add()
            s.pose.position.x = float(pts[r, 0])
            s.pose.position.y = float(pts[r, 1])
            s.pose.position.z = float(pts[r, 2])
    return FakeMsg(m.SerializeToString(), ts)


def hand_kp_2d_msg(ts: int = BASE_TS) -> FakeMsg:
    m = SceneUpdate_pb2.SceneUpdate()
    for side_name, is_right in (("left", False), ("right", True)):
        e = m.entities.add()
        e.frame_id = "world"
        e.id = f"{side_name}_hand_skeleton"
        e.frame_locked = True
        pts = np.random.rand(21, 2).astype(np.float32)
        for r in range(21):
            s = e.spheres.add()
            s.pose.position.x = float(pts[r, 0] * 1920)
            s.pose.position.y = float(pts[r, 1] * 1456)
            s.pose.position.z = 0.0
    return FakeMsg(m.SerializeToString(), ts)


def hand_mesh_msg(ts: int = BASE_TS, side: str = "left") -> FakeMsg:
    m = SceneUpdate_pb2.SceneUpdate()
    e = m.entities.add()
    e.frame_id = "world"
    e.id = f"{side}_hand_mesh"
    e.frame_locked = True
    tri = e.triangles.add()
    verts = np.random.rand(778, 3).astype(np.float32)
    for v in verts:
        p = tri.points.add()
        p.x, p.y, p.z = float(v[0]), float(v[1]), float(v[2])
    tri.indices.extend(list(range(0, 4614)))  # 4614 ints -> reshape (1538,3)
    return FakeMsg(m.SerializeToString(), ts)


def tf_msg(ts: int = BASE_TS, n_finger_joints: int = 4) -> FakeMsg:
    """构造一个 tf_tree 消息：head 相机 + 两腕 + 少量指节（测试够用）。

    frame[child] 在 ctx 里存为 (parent, t(3,), q(4,), ts)。
    """
    m = FrameTransforms_pb2.FrameTransforms()
    frames = {
        "head_left_camera": ([0.0, 0.27, 0.74], [0.0, 0.0, 0.0, 1.0]),
        "head_right_camera": ([0.0, 0.29, 0.74], [0.0, 0.0, 0.0, 1.0]),
        "left_wrist": ([0.2, 0.62, 0.48], [0.1, 0.0, 0.0, 0.99]),
        "right_wrist": ([0.3, 0.32, 0.49], [0.0, 0.1, 0.0, 0.99]),
    }
    for i in range(n_finger_joints):
        frames[f"left_thumb_{i}"] = ([0.21 + i * 0.01, 0.6, 0.47], [1.0, 0, 0, 0])
        frames[f"right_thumb_{i}"] = ([0.31 + i * 0.01, 0.3, 0.47], [1.0, 0, 0, 0])
    for child, (t, q) in frames.items():
        ft = m.transforms.add()
        ft.timestamp.seconds = ts // 1_000_000_000
        ft.timestamp.nanos = ts % 1_000_000_000
        ft.parent_frame_id = "world"
        ft.child_frame_id = child
        ft.translation.x, ft.translation.y, ft.translation.z = t
        ft.rotation.x, ft.rotation.y, ft.rotation.z, ft.rotation.w = q
    return FakeMsg(m.SerializeToString(), ts)


def head_traj_msg(ts: int = BASE_TS, pos=(0.0, 0.27, 0.74)) -> FakeMsg:
    m = SceneUpdate_pb2.SceneUpdate()
    e = m.entities.add()
    e.frame_id = "world"
    e.id = "head_pose_trajectory"
    e.frame_locked = True
    s = e.spheres.add()
    s.pose.position.x, s.pose.position.y, s.pose.position.z = pos
    return FakeMsg(m.SerializeToString(), ts)


def pointcloud_msg(ts: int = BASE_TS, n: int = 5, static: bool = False) -> FakeMsg:
    m = PointCloud_pb2.PointCloud()
    m.timestamp.seconds = ts // 1_000_000_000
    m.timestamp.nanos = ts % 1_000_000_000
    m.frame_id = "world"
    m.point_stride = 16
    for name, off, typ in [
        ("x", 0, 9), ("y", 4, 9), ("z", 8, 9),
        ("red", 12, 2), ("green", 13, 2), ("blue", 14, 2), ("alpha", 15, 2),
    ]:
        f = m.fields.add()
        f.name, f.offset, f.type = name, off, typ
    pts = np.random.rand(n, 3).astype(np.float32)
    rgb = np.random.randint(0, 256, (n, 3), dtype=np.uint8)
    buf = np.zeros((n, 16), dtype=np.uint8)
    buf[:, 0:12] = pts.view(np.uint8).reshape(n, 12)
    buf[:, 12:15] = rgb
    m.data = buf.tobytes()
    return FakeMsg(m.SerializeToString(), ts)


def video_msg(ts: int = BASE_TS, side: str = "left") -> FakeMsg:
    """构造一个伪 NAL（不是真 H.264，只测字节流拼接 + manifest）。"""
    m = CompressedVideo_pb2.CompressedVideo()
    m.timestamp.seconds = ts // 1_000_000_000
    m.timestamp.nanos = ts % 1_000_000_000
    m.frame_id = f"head_{side}_camera"
    m.format = "h264"
    m.data = bytes([0, 0, 0, 1, 0x67]) + bytes([side == "left"]) * 8
    return FakeMsg(m.SerializeToString(), ts)


SCHEMA_TOPICS = [
    ("foxglove.FrameTransforms", "/pose_algo_body/tf_tree"),
    ("foxglove.SceneUpdate", "/pose_algo_body/hand_keypoints"),
    ("foxglove.SceneUpdate", "/pose_algo_body/hand_keypoints_2d"),
    ("foxglove.SceneUpdate", "/pose_algo_body/head_pose_trajectory"),
    ("foxglove.SceneUpdate", "/pose_algo_body/hand_mesh"),
    ("foxglove.PointCloud", "/pcd_dir/pointcloud"),
    ("foxglove.PointCloud", "/pcd_dir/pointcloud/static_scene"),
    ("foxglove.PointCloud", "/pcd_dir/pointcloud/2d_projection"),
    ("foxglove.CompressedVideo", "/head_left_camera_undistorted/head_left_camera_undistorted/image"),
    ("foxglove.CompressedVideo", "/head_right_camera_undistorted/head_right_camera_undistorted/image"),
    ("foxglove.CameraCalibration", "/head_left_camera_params/head_left_camera_undistorted/camera_info"),
    ("foxglove.CameraCalibration", "/head_left_camera_params/head_left_camera_raw/camera_info"),
    ("foxglove.CameraCalibration", "/head_right_camera_params/head_right_camera_undistorted/camera_info"),
    ("foxglove.CameraCalibration", "/head_right_camera_params/head_right_camera_raw/camera_info"),
    ("foxglove.RawImage", "/head_depth_camera/depth/image"),
    ("foxglove.CameraCalibration", "/head_depth_camera_params/head_depth_camera/camera_info"),
]


def build_synth_mcap(p: Path) -> Path:
    """3 帧合成 foxglove MCAP。"""
    f = open(p, "wb")
    w = Writer(f)
    w.start()
    channels = {}
    for sch_name, topic in SCHEMA_TOPICS:
        sid = w.register_schema(name=sch_name, encoding="protobuf", data=schema_data(sch_name))
        cid = w.register_channel(topic=topic, message_encoding="protobuf", schema_id=sid)
        channels[topic] = cid
    for i in range(3):
        ts = BASE_TS + i * STEP
        msgs = [
            (SCHEMA_TOPICS[0][1], tf_msg(ts).data),
            (SCHEMA_TOPICS[1][1], hand_kp_3d_msg(ts).data),
            (SCHEMA_TOPICS[2][1], hand_kp_2d_msg(ts).data),
            (SCHEMA_TOPICS[3][1], head_traj_msg(ts, pos=(0.0 + i, 0.27, 0.74)).data),
            (SCHEMA_TOPICS[4][1], hand_mesh_msg(ts, "left").data),
            (SCHEMA_TOPICS[4][1], hand_mesh_msg(ts, "right").data),
            (SCHEMA_TOPICS[5][1], pointcloud_msg(ts, n=4).data),
            (SCHEMA_TOPICS[6][1], pointcloud_msg(ts, n=6, static=True).data),
            (SCHEMA_TOPICS[7][1], b""),  # 空 2d_projection
            (SCHEMA_TOPICS[8][1], video_msg(ts, "left").data),
            (SCHEMA_TOPICS[9][1], video_msg(ts, "right").data),
            (SCHEMA_TOPICS[10][1], calib_msg("left", "undistorted", ts).data),
            (SCHEMA_TOPICS[11][1], calib_msg("left", "raw", ts).data),
            (SCHEMA_TOPICS[12][1], calib_msg("right", "undistorted", ts).data),
            (SCHEMA_TOPICS[13][1], calib_msg("right", "raw", ts).data),
            (SCHEMA_TOPICS[14][1], raw_image_msg(ts).data),
            (SCHEMA_TOPICS[15][1], depth_calib_msg(ts).data),
        ]
        for topic, data in msgs:
            w.add_message(channel_id=channels[topic], log_time=ts, data=data, publish_time=ts)
    w.finish()
    f.close()
    return p
