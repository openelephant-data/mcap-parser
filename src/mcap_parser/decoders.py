"""Foxglove protobuf 解码助手 + frame-id 常量。

frame-id 常量源自对基准 head_pose_mcap.mcap tf_tree 的实测：
tf_tree 每帧含 44 个 frame（2 相机 + 2 腕 + 40 指节），无 24-SMPL 身体骨架。
"""
from __future__ import annotations

import numpy as np
from google.protobuf import descriptor_pb2
from foxglove_schemas_protobuf import (
    CameraCalibration_pb2,
    CompressedVideo_pb2,
    FrameTransforms_pb2,
    PointCloud_pb2,
    SceneUpdate_pb2,
)

# --- frame-id / 实体常量（实测） ---
HEAD_FRAME_ID = "head_left_camera"
WRIST_FRAME_IDS = ("left_wrist", "right_wrist")
HAND_SKELETON_IDS = ("left_hand_skeleton", "right_hand_skeleton")
HAND_MESH_IDS = ("left_hand_mesh", "right_hand_mesh")
TRAJ_ENTITY_ID = "head_pose_trajectory"
N_HAND_JOINTS = 21
N_MESH_VERTS = 778
N_MESH_FACES = 1538  # 4614 indices / 3

# --- schema -> 类 / FileDescriptor ---
MSG_CLASS = {
    "foxglove.CameraCalibration": CameraCalibration_pb2.CameraCalibration,
    "foxglove.CompressedVideo": CompressedVideo_pb2.CompressedVideo,
    "foxglove.FrameTransforms": FrameTransforms_pb2.FrameTransforms,
    "foxglove.PointCloud": PointCloud_pb2.PointCloud,
    "foxglove.SceneUpdate": SceneUpdate_pb2.SceneUpdate,
}
FILE_DESC = {
    "foxglove.CameraCalibration": CameraCalibration_pb2.DESCRIPTOR,
    "foxglove.CompressedVideo": CompressedVideo_pb2.DESCRIPTOR,
    "foxglove.FrameTransforms": FrameTransforms_pb2.DESCRIPTOR,
    "foxglove.PointCloud": PointCloud_pb2.DESCRIPTOR,
    "foxglove.SceneUpdate": SceneUpdate_pb2.DESCRIPTOR,
}

# foxglove NumericType 枚举值 -> numpy dtype 字符串
_NUMPY_DTYPE = {
    1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4",
    7: "i8", 8: "u8", 9: "f4", 10: "f8",
}


def schema_data(schema_name: str) -> bytes:
    """构造 mcap protobuf schema 所需的 FileDescriptorSet（含依赖）。"""
    fd = FILE_DESC[schema_name]
    fdset = descriptor_pb2.FileDescriptorSet()
    seen: set[str] = set()

    def _add(file_desc) -> None:
        if file_desc.name in seen:
            return
        for dep in file_desc.dependencies:
            _add(dep)
        file_desc.CopyToProto(fdset.file.add())
        seen.add(file_desc.name)

    _add(fd)
    return fdset.SerializeToString()


def decode(schema_name: str, data: bytes):
    """按 schema 名反序列化一条 foxglove 消息。"""
    obj = MSG_CLASS[schema_name]()
    obj.ParseFromString(data)
    return obj


def decode_pointcloud(obj) -> tuple[np.ndarray, np.ndarray]:
    """把 foxglove.PointCloud 的 data 解为 (points(N,3), colors(N,3)∈[0,1])。

    按字段的 offset+type 通用解析（基准 MCAP：x/y/z FLOAT32@0/4/8，rgba UINT8@12..15，stride 16）。
    """
    stride = obj.point_stride
    n = len(obj.data) // stride if stride else 0
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    raw = np.frombuffer(obj.data, dtype=np.uint8).reshape(n, stride)
    fld = {f.name: (f.offset, f.type) for f in obj.fields}

    def _col(name: str):
        off, typ = fld[name]
        dt = _NUMPY_DTYPE.get(typ)
        if dt is None:
            return None
        size = np.dtype(dt).itemsize
        return raw[:, off:off + size].copy().view(dt).reshape(n)

    pts = np.stack([_col(ax) for ax in ("x", "y", "z")], axis=1).astype(np.float64)
    colors = np.zeros((n, 3))
    for i, cn in enumerate(("red", "green", "blue")):
        if cn in fld:
            colors[:, i] = _col(cn).astype(np.float64) / 255.0
    return pts, colors
