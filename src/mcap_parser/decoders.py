"""Foxglove protobuf 解码助手 + frame-id 常量。

frame-id 常量源自对基准 head_pose_mcap.mcap tf_tree 的实测：
tf_tree 每帧含 44 个 frame（2 相机 + 2 腕 + 40 指节），无 24-SMPL 身体骨架。
"""
from __future__ import annotations

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from foxglove_schemas_protobuf import (
    CameraCalibration_pb2,
    CompressedImage_pb2,
    CompressedVideo_pb2,
    FrameTransforms_pb2,
    PointCloud_pb2,
    RawImage_pb2,
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
    "foxglove.CompressedImage": CompressedImage_pb2.CompressedImage,
    "foxglove.CompressedVideo": CompressedVideo_pb2.CompressedVideo,
    "foxglove.FrameTransforms": FrameTransforms_pb2.FrameTransforms,
    "foxglove.PointCloud": PointCloud_pb2.PointCloud,
    "foxglove.RawImage": RawImage_pb2.RawImage,
    "foxglove.SceneUpdate": SceneUpdate_pb2.SceneUpdate,
}
FILE_DESC = {
    "foxglove.CameraCalibration": CameraCalibration_pb2.DESCRIPTOR,
    "foxglove.CompressedImage": CompressedImage_pb2.DESCRIPTOR,
    "foxglove.CompressedVideo": CompressedVideo_pb2.DESCRIPTOR,
    "foxglove.FrameTransforms": FrameTransforms_pb2.DESCRIPTOR,
    "foxglove.PointCloud": PointCloud_pb2.DESCRIPTOR,
    "foxglove.RawImage": RawImage_pb2.DESCRIPTOR,
    "foxglove.SceneUpdate": SceneUpdate_pb2.DESCRIPTOR,
}

# foxglove PackedElementField.NumericType 枚举值 -> numpy dtype 字符串
# 实测 foxglove 枚举: UINT8=1 INT8=2 UINT16=3 INT16=4 UINT32=5 INT32=6 FLOAT32=7 FLOAT64=8
_NUMPY_DTYPE = {
    1: "u1", 2: "i1", 3: "u2", 4: "i2", 5: "u4", 6: "i4",
    7: "f4", 8: "f8",
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


def oeego_msg_class(schema_name: str, schema_data: bytes):
    """从 oeego/自定义 schema 的 FileDescriptorSet(schema.data)动态构建 protobuf 消息类。

    schema_name 是全限定消息名(如 'oeego.system_proto.MissionManagerProto.OnboardConfig')。
    """
    fds = descriptor_pb2.FileDescriptorSet()
    fds.ParseFromString(schema_data)
    pool = descriptor_pool.DescriptorPool()
    for fd in fds.file:
        pool.Add(fd)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(schema_name))
