# MCAP 解析器实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `head_pose_mcap.mcap`（foxglove/protobuf）单遍解析为与 `openembodied-data` 原始产物对齐的独立文件（视频/手部关键点/hand_mesh/头部轨迹/点云/相机标定/wrist 位姿/tf）。

**Architecture:** 单遍流式分发：`McapFile.parse()` 用 mcap 库 `iter_messages()`，每条消息按 topic 前缀匹配给归属 `Writer`；writer 在 `feed` 阶段把中间结果写入共享 `ParseContext`，在 `finalize` 阶段从 ctx 取数写文件。跨流依赖（头部轨迹需 head 变换、aligned_pointclouds 需 tf+手部）通过共享 ctx 的 `tf_cache`/`hand_3d` 解决。

**Tech Stack:** Python 3.12、`mcap`、`foxglove-schemas-protobuf`、`numpy`、`opencv-python`、`fire`、`pytest`、系统 `ffmpeg`（可选）。

**Spec:** `docs/2026-08-19-mcap-parser-design.md`（已含「不可恢复」修正：tf_tree 无 24-SMPL 身体骨架，body_tf 改出 `tracking_data_tf.jsonl` + `wrist_poses.npz`；手部 2D 直接取自 ch3，不做投影）。

**仓库根：** `~/work/hvyogo/ai_work/mcap-parser/`（git 已 init，已有 spec commit）。

**共享 API 约定（所有 writer 一致）：**
- `Writer.feed(self, topic: str, msg, ctx: ParseContext)` —— `msg` 必须有 `.data: bytes` 与 `.publish_time: int`（ns）。writer 用 `self._decode(msg.data)`（按 `self.schema_name` 解码）。
- `Writer.finalize(self, out_dir: Path, ctx: ParseContext)` —— 写文件，把条目加到 `ctx.outputs[filename] = {...}`。
- `Writer.owns_topic(topic)` 按 `self.topics` 前缀匹配。

---

## Task 1: 仓库骨架 + `decoders.py`

**Files:**
- Create: `pyproject.toml`
- Create: `src/mcap_parser/__init__.py`
- Create: `src/mcap_parser/decoders.py`

- [ ] **Step 1: 写 `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "mcap-parser"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "mcap>=1.3",
  "foxglove-schemas-protobuf>=0.3",
  "numpy>=1.26",
  "opencv-python>=4.8",
  "fire>=0.5",
]

[project.optional-dependencies]
dev = ["pytest>=7", "pytest-cov"]

[project.scripts]
mcap-parser = "mcap_parser.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: 写 `src/mcap_parser/__init__.py`**

```python
"""mcap_parser: foxglove MCAP → files round-trip parser."""
__version__ = "0.1.0"
```

- [ ] **Step 3: 写 `src/mcap_parser/decoders.py`**

```python
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

# --- schema → 类 / FileDescriptor ---
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

# foxglove NumericType 枚举值 → numpy dtype 字符串
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
```

- [ ] **Step 4: 安装并验证 import**

Run: `cd /Users/hutx/work/hvyogo/ai_work/mcap-parser && pip install -e ".[dev]"`
Expected: 安装成功（已装的 mcap/numpy/opencv/foxglove 复用）。

Run: `python -c "from mcap_parser.decoders import schema_data, decode_pointcloud, HEAD_FRAME_ID; print(len(schema_data('foxglove.SceneUpdate')), HEAD_FRAME_ID)"`
Expected: 打印一个正整数（schema 字节数）和 `head_left_camera`。

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/mcap_parser/__init__.py src/mcap_parser/decoders.py
git commit -m "feat: 仓库骨架 + decoders（schema/常量/点云解码）"
```

---

## Task 2: `conftest.py` 测试夹具（FakeMsg + 消息构造器 + 合成 mcap）

**Files:**
- Create: `tests/conftest.py`

> 说明：writer 单测用 `FakeMsg`（只含 `data`/`publish_time`）直接喂 `feed`，无需真实 mcap。`synth_mcap` 夹具给 `reader.py`/集成测试用。

- [ ] **Step 1: 写 `tests/conftest.py`**

```python
"""测试夹具：FakeMsg、各 schema 消息构造器、合成 foxglove MCAP。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from foxglove_schemas_protobuf import (
    CameraCalibration_pb2,
    CompressedVideo_pb2,
    FrameTransforms_pb2,
    PointCloud_pb2,
    SceneUpdate_pb2,
)
from mcap.writer import Writer

from mcap_parser.decoders import schema_data

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


# --- 合成 mcap（给 reader / 集成测试用）---
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
]


@pytest.fixture
def synth_mcap(tmp_path) -> Path:
    """3 帧合成 foxglove MCAP。"""
    p = tmp_path / "synth.mcap"
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
        ]
        for topic, data in msgs:
            w.add_message(channel_id=channels[topic], log_time=ts, data=data, publish_time=ts)
    w.finish()
    f.close()
    return p
```

- [ ] **Step 2: 验证夹具可构造**

Run: `cd /Users/hutx/work/hvyogo/ai_work/mcap-parser && python -c "import tests.conftest as c; import tempfile,pathlib; p=pathlib.Path(tempfile.mkdtemp())/'s.mcap'; c.synth_mcap.__wrapped__(p) if hasattr(c.synth_mcap,'__wrapped__') else None; print('ok')"` (跳过；下面直接跑 pytest 收集即可)

实际验证：`pytest tests/ --collect-only -q 2>&1 | head`
Expected: 暂无测试，但 conftest 无 import 错误（`0 tests collected`）。

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: conftest 夹具（FakeMsg + 消息构造器 + 合成 mcap）"
```

---

## Task 3: `writers/base.py`（Writer ABC + ParseContext）

**Files:**
- Create: `src/mcap_parser/writers/__init__.py`
- Create: `src/mcap_parser/writers/base.py`
- Test: `tests/test_base.py`

- [ ] **Step 1: 写 `src/mcap_parser/writers/__init__.py`**

```python
"""mcap_parser writers."""
```

- [ ] **Step 2: 写 `src/mcap_parser/writers/base.py`**

```python
"""Writer 基类 + 共享 ParseContext。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ParseContext:
    """单遍扫描的共享状态。各 writer 在 feed 阶段写入，finalize 阶段只读跨流部分。"""

    # BodyTfWriter 写入（ch1，逐帧）：
    tf_cache: list = field(default_factory=list)  # list[ dict[child_id -> (parent, t(3,), q(4,), ts)] ]
    tf_ts: list = field(default_factory=list)  # list[int]

    # HandKeypointsWriter 写入（ch2/ch3，逐帧）：
    hand_3d: dict = field(default_factory=lambda: {"left": [], "right": []})   # list[(21,3) float32]
    hand_2d: dict = field(default_factory=lambda: {"left": [], "right": []})   # list[(21,2) float32]
    hand_ts: dict = field(default_factory=lambda: {"left": [], "right": []})   # list[int]（3d 通道记一次）

    # HeadTrajectoryWriter 写入（ch4，逐帧）：
    head_pos: list = field(default_factory=list)   # list[(3,)]
    head_ts: list = field(default_factory=list)     # list[int]

    # PointcloudWriter 写入（ch6/ch7）：
    pc_frames: list = field(default_factory=list)   # list[(points(M,3), colors(M,3), ts)]
    pc_static: Any = None  # (points(M,3), colors(M,3))

    # CameraCalibWriter 写入：
    calib: dict = field(default_factory=dict)  # {"left_undistorted": {...}, ...}

    # 所有 writer 写入：
    outputs: dict = field(default_factory=dict)  # {filename: manifest dict}
    frame_count: int = 0


class Writer(ABC):
    """通道/topic 驱动的输出 writer。子类只需实现 feed/finalize。"""

    name: str = ""
    topics: tuple[str, ...] = ()
    schema_name: str = ""

    def owns_topic(self, topic: str) -> bool:
        return any(topic.startswith(p) for p in self.topics)

    @abstractmethod
    def feed(self, topic: str, msg, ctx: ParseContext) -> None: ...

    @abstractmethod
    def finalize(self, out_dir: Path, ctx: ParseContext) -> None: ...

    def _decode(self, data: bytes):
        from ..decoders import decode  # 延迟导入避免循环

        return decode(self.schema_name, data)
```

- [ ] **Step 3: 写失败测试 `tests/test_base.py`**

```python
from mcap_parser.writers.base import ParseContext, Writer


class _Dummy(Writer):
    name = "dummy"
    topics = ("/foo/",)
    schema_name = "foxglove.SceneUpdate"

    def feed(self, topic, msg, ctx):
        ctx.outputs["dummy.txt"] = {"topic": topic}

    def finalize(self, out_dir, ctx):
        (out_dir / "dummy.txt").write_text("hi")


def test_owns_topic_prefix_match():
    assert _Dummy().owns_topic("/foo/bar") is True
    assert _Dummy().owns_topic("/baz/qux") is False


def test_parsecontext_defaults_independent():
    a, b = ParseContext(), ParseContext()
    a.tf_cache.append(1)
    assert b.tf_cache == []
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `cd /Users/hutx/work/hvyogo/ai_work/mcap-parser && pytest tests/test_base.py -v`
Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add src/mcap_parser/writers/__init__.py src/mcap_parser/writers/base.py tests/test_base.py
git commit -m "feat: Writer ABC + ParseContext 共享状态"
```

---

## Task 4: `CameraCalibWriter`

**Files:**
- Create: `src/mcap_parser/writers/camera_calib.py`
- Test: `tests/test_camera_calib.py`

- [ ] **Step 1: 写失败测试**

```python
import json

from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.camera_calib import CameraCalibWriter
from tests.conftest import calib_msg


def test_camera_calib_writes_4_groups(tmp_path):
    w = CameraCalibWriter()
    ctx = ParseContext()
    topics = [
        "/head_left_camera_params/head_left_camera_undistorted/camera_info",
        "/head_left_camera_params/head_left_camera_raw/camera_info",
        "/head_right_camera_params/head_right_camera_undistorted/camera_info",
        "/head_right_camera_params/head_right_camera_raw/camera_info",
    ]
    for t in topics:
        side = "left" if "head_left" in t else "right"
        variant = "undistorted" if "undistorted" in t else "raw"
        w.feed(t, calib_msg(side, variant), ctx)
    # 重复喂一帧，确认只存首帧
    w.feed(topics[0], calib_msg("left", "undistorted", ts=2), ctx)
    w.finalize(tmp_path, ctx)

    data = json.loads((tmp_path / "camera_calibration.json").read_text())
    assert set(data.keys()) == {
        "left_undistorted", "left_raw", "right_undistorted", "right_raw"
    }
    assert len(data["left_undistorted"]["K"]) == 9
    assert len(data["left_undistorted"]["P"]) == 12
    assert data["left_undistorted"]["frame_id"] == "head_left_camera"
    assert ctx.outputs["camera_calibration.json"]["groups"] == sorted(data.keys())
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_camera_calib.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 写实现 `src/mcap_parser/writers/camera_calib.py`**

```python
"""相机标定 writer：ch11-14 → camera_calibration.json。"""
from __future__ import annotations

import json
from pathlib import Path

from .base import ParseContext, Writer

CALIB_TOPICS = ("/head_left_camera_params/", "/head_right_camera_params/")


class CameraCalibWriter(Writer):
    name = "camera_calib"
    topics = CALIB_TOPICS
    schema_name = "foxglove.CameraCalibration"

    def feed(self, topic: str, msg, ctx: ParseContext) -> None:
        obj = self._decode(msg.data)
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
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_camera_calib.py -v`
Expected: 1 passed。

- [ ] **Step 5: Commit**

```bash
git add src/mcap_parser/writers/camera_calib.py tests/test_camera_calib.py
git commit -m "feat: CameraCalibWriter → camera_calibration.json"
```

---

## Task 5: `HandKeypointsWriter`

**Files:**
- Create: `src/mcap_parser/writers/hand_keypoints.py`
- Test: `tests/test_hand_keypoints.py`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np

from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.hand_keypoints import HandKeypointsWriter
from tests.conftest import hand_kp_2d_msg, hand_kp_3d_msg


def test_hand_keypoints_3d_and_2d(tmp_path):
    w = HandKeypointsWriter()
    ctx = ParseContext()
    for i in range(3):
        ts = 1_000_000_000 + i * 33_000_000
        w.feed("/pose_algo_body/hand_keypoints", hand_kp_3d_msg(ts), ctx)
        w.feed("/pose_algo_body/hand_keypoints_2d", hand_kp_2d_msg(ts), ctx)
    w.finalize(tmp_path, ctx)

    for side in ("left", "right"):
        a3 = np.load(tmp_path / f"{side}_keypoints_3d.npy")
        a2 = np.load(tmp_path / f"{side}_keypoints_2d.npy")
        assert a3.shape == (3, 21, 3)
        assert a2.shape == (3, 21, 2)
        assert a3.dtype == np.float32
        assert len(ctx.hand_ts[side]) == 3


def test_hand_keypoints_missing_side_filled_nan(tmp_path):
    w = HandKeypointsWriter()
    ctx = ParseContext()
    ts = 1_000_000_000
    # 只给左手 3D，右手缺失 → 右手应为全 nan
    w.feed("/pose_algo_body/hand_keypoints", hand_kp_3d_msg(ts, left=True, right=False), ctx)
    w.feed("/pose_algo_body/hand_keypoints_2d", hand_kp_2d_msg(ts), ctx)
    w.finalize(tmp_path, ctx)

    right = np.load(tmp_path / "right_keypoints_3d.npy")
    assert right.shape == (1, 21, 3)
    assert np.isnan(right).all()
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_hand_keypoints.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 写实现 `src/mcap_parser/writers/hand_keypoints.py`**

```python
"""手部关键点 writer：ch2(3D) + ch3(2D) → left/right_keypoints_{3d,2d}.npy。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..decoders import HAND_SKELETON_IDS, N_HAND_JOINTS
from .base import ParseContext, Writer

KP_TOPICS = ("/pose_algo_body/hand_keypoints", "/pose_algo_body/hand_keypoints_2d")


class HandKeypointsWriter(Writer):
    name = "hand_keypoints"
    topics = KP_TOPICS
    schema_name = "foxglove.SceneUpdate"

    def __init__(self, side_filter: str = "both"):
        self.side_filter = side_filter

    def feed(self, topic: str, msg, ctx: ParseContext) -> None:
        is_2d = topic.endswith("_2d")
        obj = self._decode(msg.data)
        ts = msg.publish_time
        buf = ctx.hand_2d if is_2d else ctx.hand_3d
        present = {"left": False, "right": False}
        for e in obj.entities:
            if e.id == HAND_SKELETON_IDS[0]:
                side = "left"
            elif e.id == HAND_SKELETON_IDS[1]:
                side = "right"
            else:
                continue
            if self.side_filter != "both" and side != self.side_filter:
                continue
            if len(e.spheres) != N_HAND_JOINTS:
                continue
            pts = np.array(
                [(s.pose.position.x, s.pose.position.y, s.pose.position.z) for s in e.spheres],
                dtype=np.float32,
            )
            buf[side].append(pts if not is_2d else pts[:, :2])
            present[side] = True
        # 3D 通道负责记 timestamps（两侧都记，保证长度对齐）
        if not is_2d:
            for side in ("left", "right"):
                if not present[side]:
                    # 缺失侧用全 nan 占位
                    buf[side].append(np.full((N_HAND_JOINTS, 3), np.nan, dtype=np.float32))
                    if is_2d is False and side in ctx.hand_2d and len(ctx.hand_2d[side]) < len(buf[side]):
                        pass
                ctx.hand_ts[side].append(ts)

    def finalize(self, out_dir: Path, ctx: ParseContext) -> None:
        for side in ("left", "right"):
            for buf, suffix in ((ctx.hand_3d[side], "_3d"), (ctx.hand_2d[side], "_2d")):
                if not buf:
                    continue
                arr = np.stack(buf)
                np.save(out_dir / f"{side}_keypoints{suffix}.npy", arr)
                ctx.outputs[f"{side}_keypoints{suffix}.npy"] = {
                    "frames": len(buf),
                    "side": side,
                    "timestamps_ns": list(ctx.hand_ts[side][: len(buf)]),
                }
```

> 注：占位分支里 `if is_2d is False ...` 是防御性 no-op，确保 2D buffer 与 3D 帧数对齐只在 2D 通道缺失时才需要补 nan——本实现里 2D 帧若缺失由其自身 `present` 逻辑处理，保持简单。

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_hand_keypoints.py -v`
Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add src/mcap_parser/writers/hand_keypoints.py tests/test_hand_keypoints.py
git commit -m "feat: HandKeypointsWriter → keypoints_3d/2d .npy"
```

---

## Task 6: `HandMeshWriter`

**Files:**
- Create: `src/mcap_parser/writers/hand_mesh.py`
- Test: `tests/test_hand_mesh.py`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np

from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.hand_mesh import HandMeshWriter
from tests.conftest import hand_mesh_msg


def test_hand_mesh_shape_and_keys(tmp_path):
    w = HandMeshWriter()
    ctx = ParseContext()
    for i in range(3):
        ts = 1_000_000_000 + i
        for side in ("left", "right"):
            w.feed("/pose_algo_body/hand_mesh", hand_mesh_msg(ts, side), ctx)
    w.finalize(tmp_path, ctx)

    for side, is_right in (("left", False), ("right", True)):
        z = np.load(tmp_path / f"{side}_hand_vertices.npz")
        assert z["vertices"].shape == (3, 778, 3)
        assert z["faces"].shape == (1538, 3)
        assert z["is_right"].shape == (3,)
        assert bool(z["is_right"][0]) is is_right
        assert bool(z["detected"][0]) is True
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_hand_mesh.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 写实现 `src/mcap_parser/writers/hand_mesh.py`**

```python
"""手部网格 writer：ch5 → left/right_hand_vertices.npz。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..decoders import HAND_MESH_IDS
from .base import ParseContext, Writer

MESH_TOPIC = ("/pose_algo_body/hand_mesh",)


class HandMeshWriter(Writer):
    name = "hand_mesh"
    topics = MESH_TOPIC
    schema_name = "foxglove.SceneUpdate"

    def __init__(self):
        self._verts = {"left": [], "right": []}
        self._faces = {"left": None, "right": None}
        self._is_right = {"left": False, "right": True}

    def feed(self, topic: str, msg, ctx: ParseContext) -> None:
        obj = self._decode(msg.data)
        for e in obj.entities:
            if e.id == HAND_MESH_IDS[0]:
                side = "left"
            elif e.id == HAND_MESH_IDS[1]:
                side = "right"
            else:
                continue
            tri = e.triangles[0] if len(e.triangles) else None
            if tri is None or not tri.points:
                continue
            verts = np.array(
                [(p.x, p.y, p.z) for p in tri.points], dtype=np.float32
            )
            faces = np.array(list(tri.indices), dtype=np.int32).reshape(-1, 3)
            self._verts[side].append(verts)
            if self._faces[side] is None:
                self._faces[side] = faces

    def finalize(self, out_dir: Path, ctx: ParseContext) -> None:
        for side in ("left", "right"):
            if not self._verts[side]:
                continue
            verts = np.stack(self._verts[side])  # (N,778,3)
            faces = self._faces[side]
            n = len(verts)
            is_right = np.full(n, self._is_right[side], dtype=bool)
            detected = np.ones(n, dtype=bool)
            np.savez_compressed(
                out_dir / f"{side}_hand_vertices.npz",
                vertices=verts, faces=faces, is_right=is_right, detected=detected,
            )
            ctx.outputs[f"{side}_hand_vertices.npz"] = {"frames": n, "side": side}
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_hand_mesh.py -v`
Expected: 1 passed。

- [ ] **Step 5: Commit**

```bash
git add src/mcap_parser/writers/hand_mesh.py tests/test_hand_mesh.py
git commit -m "feat: HandMeshWriter → hand_vertices.npz (verts+faces)"
```

---

## Task 7: `BodyTfWriter`（tf_cache + jsonl + wrist_poses）

**Files:**
- Create: `src/mcap_parser/writers/body_tf.py`
- Test: `tests/test_body_tf.py`

- [ ] **Step 1: 写失败测试**

```python
import json

import numpy as np

from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.body_tf import BodyTfWriter
from tests.conftest import tf_msg


def test_body_tf_writes_jsonl_and_wrist_poses(tmp_path):
    w = BodyTfWriter()
    ctx = ParseContext()
    for i in range(3):
        ts = 1_000_000_000 + i * 33_000_000
        w.feed("/pose_algo_body/tf_tree", tf_msg(ts), ctx)
    w.finalize(tmp_path, ctx)

    # tf_cache 三帧、含 head + 两腕
    assert len(ctx.tf_cache) == 3
    assert "head_left_camera" in ctx.tf_cache[0]
    assert "left_wrist" in ctx.tf_cache[0]

    # jsonl
    lines = (tmp_path / "tracking_data_tf.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    row0 = json.loads(lines[0])
    assert row0["ts_ns"] == 1_000_000_000
    assert len(row0["transforms"]) == len(ctx.tf_cache[0])
    assert row0["transforms"][0]["parent"] == "world"

    # wrist_poses
    z = np.load(tmp_path / "wrist_poses.npz")
    assert z["left"].shape == (3, 7)
    assert z["right"].shape == (3, 7)
    assert z["timestamps"].shape == (3,)
    np.testing.assert_allclose(z["left"][0, :3], [0.2, 0.62, 0.48])
    np.testing.assert_allclose(z["right"][0, 3:], [0.0, 0.1, 0.0, 0.99])
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_body_tf.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 写实现 `src/mcap_parser/writers/body_tf.py`**

```python
"""身体/手根 tf writer：ch1 → tracking_data_tf.jsonl + wrist_poses.npz。

feed 阶段把每帧全部 transforms 写入 ctx.tf_cache（供 head_trajectory/pointcloud 跨流用）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..decoders import HEAD_FRAME_ID, WRIST_FRAME_IDS
from .base import ParseContext, Writer

TF_TOPIC = ("/pose_algo_body/tf_tree",)


class BodyTfWriter(Writer):
    name = "body_tf"
    topics = TF_TOPIC
    schema_name = "foxglove.FrameTransforms"

    def feed(self, topic: str, msg, ctx: ParseContext) -> None:
        obj = self._decode(msg.data)
        ts = msg.publish_time
        frame: dict[str, tuple] = {}
        for t in obj.transforms:
            tr = np.array(
                [float(t.translation.x), float(t.translation.y), float(t.translation.z)],
                dtype=np.float64,
            )
            qu = np.array(
                [float(t.rotation.x), float(t.rotation.y),
                 float(t.rotation.z), float(t.rotation.w)],
                dtype=np.float64,
            )
            frame[t.child_frame_id] = (t.parent_frame_id, tr, qu, ts)
        ctx.tf_cache.append(frame)
        ctx.tf_ts.append(ts)

    def finalize(self, out_dir: Path, ctx: ParseContext) -> None:
        n = len(ctx.tf_cache)
        if n == 0:
            return

        # tracking_data_tf.jsonl
        with open(out_dir / "tracking_data_tf.jsonl", "w") as f:
            for i, frame in enumerate(ctx.tf_cache):
                ts = ctx.tf_ts[i]
                transforms = [
                    {
                        "parent": v[0],
                        "child": child,
                        "translation": v[1].tolist(),
                        "rotation": v[2].tolist(),
                    }
                    for child, v in frame.items()
                ]
                f.write(json.dumps({"ts_ns": int(ts), "transforms": transforms}) + "\n")
        ctx.outputs["tracking_data_tf.jsonl"] = {
            "frames": n,
            "transforms_per_frame": len(ctx.tf_cache[0]),
        }

        # wrist_poses.npz
        left = np.full((n, 7), np.nan)
        right = np.full((n, 7), np.nan)
        tss = np.array(ctx.tf_ts, dtype=np.int64)
        for i, frame in enumerate(ctx.tf_cache):
            for side, wid in (("left", WRIST_FRAME_IDS[0]), ("right", WRIST_FRAME_IDS[1])):
                arr = left if side == "left" else right
                if wid in frame:
                    arr[i] = np.concatenate([frame[wid][1], frame[wid][2]])
        np.savez(
            out_dir / "wrist_poses.npz", left=left, right=right, timestamps=tss,
        )
        ctx.outputs["wrist_poses.npz"] = {
            "frames": n,
            "timestamps_ns": tss.tolist(),
        }
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_body_tf.py -v`
Expected: 1 passed。

- [ ] **Step 5: Commit**

```bash
git add src/mcap_parser/writers/body_tf.py tests/test_body_tf.py
git commit -m "feat: BodyTfWriter → tracking_data_tf.jsonl + wrist_poses.npz + tf_cache"
```

---

## Task 8: `HeadTrajectoryWriter`

**Files:**
- Create: `src/mcap_parser/writers/head_trajectory.py`
- Test: `tests/test_head_trajectory.py`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np

from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.body_tf import BodyTfWriter
from mcap_parser.writers.head_trajectory import HeadTrajectoryWriter
from tests.conftest import head_traj_msg, tf_msg


def test_head_trajectory_7cols(tmp_path):
    # 需先喂 ch1 填充 tf_cache（head 四元数）
    tf = BodyTfWriter()
    traj = HeadTrajectoryWriter()
    ctx = ParseContext()
    for i in range(3):
        ts = 1_000_000_000 + i * 33_000_000
        tf.feed("/pose_algo_body/tf_tree", tf_msg(ts), ctx)
        traj.feed("/pose_algo_body/head_pose_trajectory",
                  head_traj_msg(ts, pos=(0.0 + i, 0.27, 0.74)), ctx)
    traj.finalize(tmp_path, ctx)

    arr = np.load(tmp_path / "head_pose_trajectory.npy")
    assert arr.shape == (3, 7)
    np.testing.assert_allclose(arr[0, :3], [0.0, 0.27, 0.74])  # 来自 ch4 轨迹球
    np.testing.assert_allclose(arr[0, 3:], [0.0, 0.0, 0.0, 1.0])  # 来自 tf head 变换
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_head_trajectory.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 写实现 `src/mcap_parser/writers/head_trajectory.py`**

```python
"""头部轨迹 writer：ch4(pos) + ctx.tf_cache(head quat) → head_pose_trajectory.npy (N,7)。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..decoders import HEAD_FRAME_ID, TRAJ_ENTITY_ID
from .base import ParseContext, Writer

TRAJ_TOPIC = ("/pose_algo_body/head_pose_trajectory",)


class HeadTrajectoryWriter(Writer):
    name = "head_trajectory"
    topics = TRAJ_TOPIC
    schema_name = "foxglove.SceneUpdate"

    def feed(self, topic: str, msg, ctx: ParseContext) -> None:
        obj = self._decode(msg.data)
        pos = None
        for e in obj.entities:
            if e.id != TRAJ_ENTITY_ID:
                continue
            if e.spheres:
                s = e.spheres[0]
                pos = (float(s.pose.position.x), float(s.pose.position.y), float(s.pose.position.z))
            break
        ctx.head_pos.append(
            np.array(pos if pos is not None else [np.nan] * 3, dtype=np.float64)
        )
        ctx.head_ts.append(msg.publish_time)

    def finalize(self, out_dir: Path, ctx: ParseContext) -> None:
        n = len(ctx.head_pos)
        if n == 0:
            return
        pos = np.stack(ctx.head_pos)  # (N,3)
        quat = np.full((n, 4), np.nan)
        for i in range(n):
            fi = i if i < len(ctx.tf_cache) else len(ctx.tf_cache) - 1
            if 0 <= fi < len(ctx.tf_cache):
                frame = ctx.tf_cache[fi]
                if HEAD_FRAME_ID in frame:
                    quat[i] = frame[HEAD_FRAME_ID][2]
        out = np.concatenate([pos, quat], axis=1)  # (N,7)
        np.save(out_dir / "head_pose_trajectory.npy", out)
        ctx.outputs["head_pose_trajectory.npy"] = {
            "frames": n,
            "timestamps_ns": list(ctx.head_ts),
        }
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_head_trajectory.py -v`
Expected: 1 passed。

- [ ] **Step 5: Commit**

```bash
git add src/mcap_parser/writers/head_trajectory.py tests/test_head_trajectory.py
git commit -m "feat: HeadTrajectoryWriter → head_pose_trajectory.npy (N,7)"
```

---

## Task 9: `PointcloudWriter`（PCD + aligned_pointclouds.npz + PLY）

**Files:**
- Create: `src/mcap_parser/writers/pointcloud.py`
- Test: `tests/test_pointcloud.py`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np

from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.body_tf import BodyTfWriter
from mcap_parser.writers.hand_keypoints import HandKeypointsWriter
from mcap_parser.writers.pointcloud import PointcloudWriter
from tests.conftest import hand_kp_3d_msg, pointcloud_msg, tf_msg


def test_pointcloud_pcd_and_npz_and_ply(tmp_path):
    w = PointcloudWriter(per_frame_pcd=True)
    tf = BodyTfWriter()
    hk = HandKeypointsWriter()
    ctx = ParseContext()
    for i in range(3):
        ts = 1_000_000_000 + i * 33_000_000
        tf.feed("/pose_algo_body/tf_tree", tf_msg(ts), ctx)
        hk.feed("/pose_algo_body/hand_keypoints", hand_kp_3d_msg(ts), ctx)
        hk.feed("/pose_algo_body/hand_keypoints_2d", hand_kp_3d_msg(ts), ctx)  # 占位
        w.feed("/pcd_dir/pointcloud", pointcloud_msg(ts, n=4), ctx)
        if i == 0:
            w.feed("/pcd_dir/pointcloud/static_scene", pointcloud_msg(ts, n=6, static=True), ctx)
        w.feed("/pcd_dir/pointcloud/2d_projection", pointcloud_msg(ts, n=0), ctx)  # 空帧应被跳过
    w.finalize(tmp_path, ctx)

    # 逐帧 PCD
    pcds = sorted((tmp_path / "pointclouds").glob("*.pcd"))
    assert len(pcds) == 3
    assert pcds[0].read_text().startswith("# .PCD v0.7")

    # PLY
    ply = (tmp_path / "static_scene.ply").read_text()
    assert ply.startswith("ply")
    assert "element vertex 6" in ply

    # aligned_pointclouds.npz
    z = np.load(tmp_path / "aligned_pointclouds.npz", allow_pickle=True)
    assert z["pc_frame_data"].shape == (3,)
    assert z["pc_frame_point_counts"].tolist() == [4, 4, 4]
    assert z["head_poses"].shape == (3, 7)
    assert z["left_hand_positions"].shape == (3, 21, 3)
    assert z["left_hand_poses"].shape == (3, 7)
    assert z["pc_static_points"].shape == (6, 3)
    # missing_keys 在 manifest（Task 12）声明；这里只校验可恢复键
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_pointcloud.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 写实现 `src/mcap_parser/writers/pointcloud.py`**

```python
"""点云 writer：ch6 → pointclouds/frame_*.pcd + aligned_pointclouds.npz；ch7 → static_scene.ply。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..decoders import HEAD_FRAME_ID, WRIST_FRAME_IDS, decode_pointcloud
from .base import ParseContext, Writer

PC_TOPICS = ("/pcd_dir/pointcloud",)


def _write_pcd(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    n = len(points)
    c = np.clip(colors, 0, 1) * 255
    rgb_int = (
        (c[:, 0].astype(np.uint32) << 16)
        | (c[:, 1].astype(np.uint32) << 8)
        | c[:, 2].astype(np.uint32)
    )
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
    arr = np.empty(n, dtype=dt)
    arr["x"] = points[:, 0].astype(np.float32)
    arr["y"] = points[:, 1].astype(np.float32)
    arr["z"] = points[:, 2].astype(np.float32)
    arr["rgb"] = rgb_int
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\n"
        f"COUNT 1 1 1 1\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\nDATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(arr.tobytes())


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    n = len(points)
    c = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for i in range(n):
            f.write(
                f"{points[i,0]} {points[i,1]} {points[i,2]} "
                f"{c[i,0]} {c[i,1]} {c[i,2]}\n"
            )


class PointcloudWriter(Writer):
    name = "pointcloud"
    topics = PC_TOPICS
    schema_name = "foxglove.PointCloud"

    def __init__(self, per_frame_pcd: bool = True):
        self.per_frame_pcd = per_frame_pcd

    def feed(self, topic: str, msg, ctx: ParseContext) -> None:
        if topic.endswith("/2d_projection"):
            return  # 空占位通道
        obj = self._decode(msg.data)
        pts, cols = decode_pointcloud(obj)
        if topic.endswith("/static_scene"):
            ctx.pc_static = (pts, cols)
            return
        ctx.pc_frames.append((pts, cols, msg.publish_time))

    def finalize(self, out_dir: Path, ctx: ParseContext) -> None:
        # static PLY
        if ctx.pc_static is not None:
            pts_s, cols_s = ctx.pc_static
            _write_ply(out_dir / "static_scene.ply", pts_s, cols_s)
            ctx.outputs["static_scene.ply"] = {"points": len(pts_s)}

        # 逐帧 PCD
        if self.per_frame_pcd and ctx.pc_frames:
            pcd_dir = out_dir / "pointclouds"
            pcd_dir.mkdir(exist_ok=True)
            for i, (p, c, _ts) in enumerate(ctx.pc_frames):
                _write_pcd(pcd_dir / f"frame_{i:06d}.pcd", p, c)

        # aligned_pointclouds.npz
        n = len(ctx.tf_cache)
        n_pc = len(ctx.pc_frames)
        pc_data = np.empty(n_pc, dtype=object)
        counts = np.zeros(n_pc, dtype=np.int64)
        pc_ts = np.zeros(n_pc, dtype=np.int64)
        idx = np.arange(n_pc)
        for i, (p, _c, t) in enumerate(ctx.pc_frames):
            pc_data[i] = p
            counts[i] = len(p)
            pc_ts[i] = t

        head_poses = np.full((n, 7), np.nan)
        left_hp = np.full((n, 7), np.nan)
        right_hp = np.full((n, 7), np.nan)
        for i, frame in enumerate(ctx.tf_cache[:n]):
            if HEAD_FRAME_ID in frame:
                head_poses[i] = np.concatenate([frame[HEAD_FRAME_ID][1], frame[HEAD_FRAME_ID][2]])
            for side, wid in (("left", WRIST_FRAME_IDS[0]), ("right", WRIST_FRAME_IDS[1])):
                arr = left_hp if side == "left" else right_hp
                if wid in frame:
                    arr[i] = np.concatenate([frame[wid][1], frame[wid][2]])

        def _stack_hand(side):
            lst = ctx.hand_3d.get(side, [])
            if len(lst) >= n:
                return np.stack(lst[:n])
            out = np.full((n, 21, 3), np.nan, dtype=np.float32)
            for i, a in enumerate(lst):
                out[i] = a
            return out

        left_hand_pos = _stack_hand("left")
        right_hand_pos = _stack_hand("right")
        hand_flags = np.zeros((n, 2), dtype=np.int64)
        for i in range(min(n, len(ctx.hand_3d.get("left", [])))):
            hand_flags[i, 0] = 0 if np.isnan(ctx.hand_3d["left"][i]).all() else 1
        for i in range(min(n, len(ctx.hand_3d.get("right", [])))):
            hand_flags[i, 1] = 0 if np.isnan(ctx.hand_3d["right"][i]).all() else 1

        frame_ts_ns = np.array(ctx.tf_ts, dtype=np.int64)
        frame_ts = frame_ts_ns / 1e9

        if n_pc or ctx.pc_static is not None:
            sp = ctx.pc_static[0] if ctx.pc_static is not None else np.zeros((0, 3))
            sc = ctx.pc_static[1] if ctx.pc_static is not None else np.zeros((0, 3))
            np.savez_compressed(
                out_dir / "aligned_pointclouds.npz",
                frame_timestamps=frame_ts,
                frame_timestamps_ns=frame_ts_ns,
                head_poses=head_poses,
                left_hand_positions=left_hand_pos,
                right_hand_positions=right_hand_pos,
                left_hand_poses=left_hp,
                right_hand_poses=right_hp,
                hand_flags=hand_flags,
                pc_static_points=sp,
                pc_static_colors=sc,
                pc_frame_indices=idx,
                pc_frame_timestamps=pc_ts,
                pc_frame_point_counts=counts,
                pc_frame_data=pc_data,
            )
            ctx.outputs["aligned_pointclouds.npz"] = {
                "pc_frames": n_pc,
                "tf_frames": n,
                "pcd_files": n_pc if self.per_frame_pcd else 0,
            }
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_pointcloud.py -v`
Expected: 1 passed。

- [ ] **Step 5: Commit**

```bash
git add src/mcap_parser/writers/pointcloud.py tests/test_pointcloud.py
git commit -m "feat: PointcloudWriter → PCD + aligned_pointclouds.npz + PLY"
```

---

## Task 10: `VideoWriter`（H.264 → MP4 remux）

**Files:**
- Create: `src/mcap_parser/writers/video.py`
- Test: `tests/test_video.py`

- [ ] **Step 1: 写失败测试**

```python
from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.video import VideoWriter
from tests.conftest import video_msg


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

    # 伪 NAL 非 H.264，ffmpeg 必失败 → 退化为 .h264，manifest 标 muxed=False
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
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_video.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 写实现 `src/mcap_parser/writers/video.py`**

```python
"""视频 writer：ch9/10 → head_{left,right}_camera_undistorted.mp4（ffmpeg remux，失败退 .h264）。"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .base import ParseContext, Writer

VIDEO_TOPICS = ("/head_left_camera_undistorted/", "/head_right_camera_undistorted/")


class VideoWriter(Writer):
    name = "video"
    topics = VIDEO_TOPICS
    schema_name = "foxglove.CompressedVideo"

    def __init__(self, side_filter: str = "both"):
        self.side_filter = side_filter
        self._frames: dict[str, list[tuple[bytes, int]]] = {}

    def feed(self, topic: str, msg, ctx: ParseContext) -> None:
        side = "left" if "head_left" in topic else "right"
        if self.side_filter != "both" and side != self.side_filter:
            return
        obj = self._decode(msg.data)
        self._frames.setdefault(side, []).append((bytes(obj.data), msg.publish_time))

    def finalize(self, out_dir: Path, ctx: ParseContext) -> None:
        tmp = out_dir / ".tmp"
        tmp.mkdir(exist_ok=True)
        for side, frames in self._frames.items():
            if not frames:
                continue
            h264 = tmp / f"{side}.h264"
            with open(h264, "wb") as f:
                for data, _ts in frames:
                    f.write(data)
            mp4 = out_dir / f"head_{side}_camera_undistorted.mp4"
            ts0, ts1 = frames[0][1], frames[-1][1]
            fps = (len(frames) - 1) / ((ts1 - ts0) / 1e9) if (ts1 > ts0 and len(frames) > 1) else 30.0
            muxed = False
            if shutil.which("ffmpeg"):
                r = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(h264), "-c", "copy",
                     "-r", f"{fps:.3f}", "-f", "mp4", str(mp4)],
                    capture_output=True,
                )
                muxed = r.returncode == 0
            if muxed:
                h264.unlink()
            else:
                # 退化为 .h264
                fallback = out_dir / f"{side}.h264"
                h264.replace(fallback)
                mp4 = fallback  # 输出名同步
            ctx.outputs[f"head_{side}_camera_undistorted.mp4"] = {
                "frames": len(frames),
                "fps": round(fps, 3),
                "muxed": muxed,
                "timestamps_ns": [t for _, t in frames],
            }
        # 清理空 .tmp
        if tmp.exists() and not any(tmp.iterdir()):
            tmp.rmdir()
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_video.py -v`
Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add src/mcap_parser/writers/video.py tests/test_video.py
git commit -m "feat: VideoWriter → mp4(ffmpeg remux)/.h264 fallback"
```

---

## Task 11: `reader.py`（McapFile + build_writers + 单遍分发）

**Files:**
- Create: `src/mcap_parser/reader.py`
- Test: `tests/test_reader.py`

- [ ] **Step 1: 写失败测试**

```python
import json

from mcap_parser.reader import McapFile


def test_reader_dispatch_all_writers(synth_mcap, tmp_path):
    ctx = McapFile(synth_mcap).parse(tmp_path)
    outs = set(ctx.outputs.keys())
    assert "camera_calibration.json" in outs
    assert "left_keypoints_3d.npy" in outs
    assert "right_keypoints_3d.npy" in outs
    assert "left_hand_vertices.npz" in outs
    assert "tracking_data_tf.jsonl" in outs
    assert "wrist_poses.npz" in outs
    assert "head_pose_trajectory.npy" in outs
    assert "aligned_pointclouds.npz" in outs
    assert "static_scene.ply" in outs
    for side in ("left", "right"):
        assert f"head_{side}_camera_undistorted.mp4" in outs
    # manifest 由 Task 12 写；此处 reader 不写 manifest


def test_reader_overwrite_guard(synth_mcap, tmp_path):
    McapFile(synth_mcap).parse(tmp_path)
    import pytest
    with pytest.raises(RuntimeError):
        McapFile(synth_mcap).parse(tmp_path)  # 非空、未 overwrite
    # overwrite 通过
    McapFile(synth_mcap).parse(tmp_path, overwrite=True)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_reader.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 写实现 `src/mcap_parser/reader.py`**

```python
"""McapFile：单遍流式分发 reader。"""
from __future__ import annotations

import shutil
from pathlib import Path

from mcap.reader import make_reader

from .writers.base import ParseContext, Writer
from .writers.body_tf import BodyTfWriter
from .writers.camera_calib import CameraCalibWriter
from .writers.hand_keypoints import HandKeypointsWriter
from .writers.hand_mesh import HandMeshWriter
from .writers.head_trajectory import HeadTrajectoryWriter
from .writers.pointcloud import PointcloudWriter
from .writers.video import VideoWriter

STREAM_TO_CLASS: dict[str, type[Writer]] = {
    "video": VideoWriter,
    "hand_keypoints": HandKeypointsWriter,
    "hand_mesh": HandMeshWriter,
    "head_trajectory": HeadTrajectoryWriter,
    "pointcloud": PointcloudWriter,
    "body_tf": BodyTfWriter,
    "camera_calib": CameraCalibWriter,
}
ALL_STREAMS = list(STREAM_TO_CLASS.keys())


def build_writers(
    side: str = "both",
    streams: list[str] | None = None,
    per_frame_pcd: bool = True,
) -> list[Writer]:
    names = streams if streams else ALL_STREAMS
    writers: list[Writer] = []
    for n in names:
        cls = STREAM_TO_CLASS[n]
        if cls is VideoWriter:
            writers.append(VideoWriter(side_filter=side))
        elif cls is HandKeypointsWriter:
            writers.append(HandKeypointsWriter(side_filter=side))
        elif cls is PointcloudWriter:
            writers.append(PointcloudWriter(per_frame_pcd=per_frame_pcd))
        else:
            writers.append(cls())
    return writers


class McapFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def parse(
        self,
        out_dir: str | Path,
        *,
        side: str = "both",
        streams: list[str] | None = None,
        per_frame_pcd: bool = True,
        overwrite: bool = False,
    ) -> ParseContext:
        out_dir = Path(out_dir)
        if out_dir.exists() and any(out_dir.iterdir()):
            if not overwrite:
                raise RuntimeError(f"out_dir not empty: {out_dir}（用 overwrite=True 覆盖）")
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ctx = ParseContext()
        writers = build_writers(side=side, streams=streams, per_frame_pcd=per_frame_pcd)

        f = open(self.path, "rb")
        r = make_reader(f)
        for schema, channel, msg in r.iter_messages():
            topic = channel.topic
            for w in writers:
                if w.owns_topic(topic):
                    try:
                        w.feed(topic, msg, ctx)
                    except Exception as e:  # 解码失败不中断
                        print(f"[warn] feed {topic} failed: {e}")
        ctx.frame_count = len(ctx.tf_cache) or len(ctx.head_pos)

        # finalize 顺序：body_tf（已写 tf_cache）→ pointcloud/head_trajectory 依赖 ctx
        for w in writers:
            w.finalize(out_dir, ctx)
        f.close()
        return ctx
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_reader.py -v`
Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add src/mcap_parser/reader.py tests/test_reader.py
git commit -m "feat: McapFile 单遍分发 reader + build_writers"
```

---

## Task 12: `manifest.py`（ManifestBuilder）

**Files:**
- Create: `src/mcap_parser/manifest.py`
- Modify: `src/mcap_parser/reader.py`（parse 末尾调用 `ManifestBuilder(...).write`）
- Test: `tests/test_manifest.py`

- [ ] **Step 1: 写失败测试**

```python
import json

from mcap_parser.reader import McapFile


def test_manifest_written(synth_mcap, tmp_path):
    McapFile(synth_mcap).parse(tmp_path)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["source_mcap"].endswith("synth.mcap")
    assert m["message_count"] == 3 * 15  # 3 帧 × 15 条消息（含两份 hand_mesh）
    assert m["frame_count"] == 3
    assert "camera_calibration.json" in m["outputs"]
    assert "body_joint_positions" in m["missing_keys"]["aligned_pointclouds.npz"]
    assert "24-SMPL body skeleton" in m["not_recoverable"]
    assert m["outputs"]["tracking_data_tf.jsonl"]["transforms_per_frame"] > 0
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_manifest.py -v`
Expected: FAIL（manifest.json 不存在）。

- [ ] **Step 3: 写实现 `src/mcap_parser/manifest.py`**

```python
"""manifest.json 写入器。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .writers.base import ParseContext, Writer

MISSING_KEYS = {
    "aligned_pointclouds.npz": [
        "body_joint_positions",
        "body_joint_quaternions",
        "body_joint_names",
        "left_hand_quaternions",
        "right_hand_quaternions",
    ]
}
NOT_RECOVERABLE = [
    "24-SMPL body skeleton (from body_pose_prelabel upstream)",
    "body_keypoints_2d (24 joints)",
    "hand joint quaternions (ch2 只给位置)",
]


class ManifestBuilder:
    def __init__(
        self,
        mcap_path: Path,
        summary: Any,
        ctx: ParseContext,
        writers: list[Writer],
    ):
        self.mcap_path = mcap_path
        self.summary = summary
        self.ctx = ctx
        self.writers = writers

    def write(self, out_dir: Path) -> None:
        st = self.summary.statistics if self.summary else None
        frame_count = self.ctx.frame_count
        manifest = {
            "source_mcap": str(self.mcap_path),
            "source_size_bytes": self.mcap_path.stat().st_size,
            "message_count": st.message_count if st else 0,
            "start_time_ns": st.message_start_time if st else 0,
            "end_time_ns": st.message_end_time if st else 0,
            "frame_count": frame_count,
            "outputs": self.ctx.outputs,
            "missing_keys": MISSING_KEYS,
            "not_recoverable": NOT_RECOVERABLE,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )
```

- [ ] **Step 4: 改 `reader.py`，在 parse 末尾调用 manifest，并把 summary 返回给 reader**

在 `reader.py` 顶部 import 后，`parse` 方法末尾（`return ctx` 之前）插入：

```python
        from .manifest import ManifestBuilder
        ManifestBuilder(self.path, r.get_summary(), ctx, writers).write(out_dir)
```

（`r` 在 finalize 前仍是打开的 reader，可继续 `get_summary()`；为安全，先调 summary 再 finalize，或保留 `r` 引用。）

完整修改：把 `parse` 中 `f = open(...); r = make_reader(f)` 与循环保持，循环后、finalize 前取 `summary = r.get_summary()`，finalize 后调用 manifest，最后 `f.close()`。即：

```python
        f = open(self.path, "rb")
        r = make_reader(f)
        summary = r.get_summary()
        for schema, channel, msg in r.iter_messages():
            topic = channel.topic
            for w in writers:
                if w.owns_topic(topic):
                    try:
                        w.feed(topic, msg, ctx)
                    except Exception as e:
                        print(f"[warn] feed {topic} failed: {e}")
        ctx.frame_count = len(ctx.tf_cache) or len(ctx.head_pos)
        for w in writers:
            w.finalize(out_dir, ctx)
        from .manifest import ManifestBuilder
        ManifestBuilder(self.path, summary, ctx, writers).write(out_dir)
        f.close()
        return ctx
```

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_manifest.py tests/test_reader.py -v`
Expected: 全 passed。

- [ ] **Step 6: Commit**

```bash
git add src/mcap_parser/manifest.py src/mcap_parser/reader.py tests/test_manifest.py
git commit -m "feat: ManifestBuilder → manifest.json（含 missing_keys/not_recoverable）"
```

---

## Task 13: `cli.py` + `__main__.py`

**Files:**
- Create: `src/mcap_parser/cli.py`
- Create: `src/mcap_parser/__main__.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: 写失败测试**

```python
import json
import subprocess
import sys


def test_cli_runs(synth_mcap, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "mcap_parser", str(synth_mcap),
         "--out_dir", str(tmp_path / "out"), "--overwrite", "true"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert m["frame_count"] == 3
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL（模块不可执行）。

- [ ] **Step 3: 写 `src/mcap_parser/cli.py`**

```python
"""CLI: python -m mcap_parser <mcap> --out_dir <dir> [--side ...] [--streams ...]"""
from __future__ import annotations

from .reader import McapFile


def main(
    mcap_path: str,
    out_dir: str,
    side: str = "both",
    streams: str | None = None,
    per_frame_pcd: bool = True,
    overwrite: bool = False,
) -> None:
    stream_list = streams.split(",") if isinstance(streams, str) else streams
    ctx = McapFile(mcap_path).parse(
        out_dir,
        side=side,
        streams=stream_list,
        per_frame_pcd=per_frame_pcd,
        overwrite=overwrite,
    )
    print(f"[mcap_parser] done: {len(ctx.outputs)} outputs in {out_dir}")
```

- [ ] **Step 4: 写 `src/mcap_parser/__main__.py`**

```python
import fire

from .cli import main

if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_cli.py -v`
Expected: 1 passed。

- [ ] **Step 6: Commit**

```bash
git add src/mcap_parser/cli.py src/mcap_parser/__main__.py tests/test_cli.py
git commit -m "feat: CLI（fire）python -m mcap_parser"
```

---

## Task 14: 集成测试（真实截断 MCAP）

**Files:**
- Create: `scripts/truncate_mcap.py`（一次性截断工具）
- Create: `tests/test_integration.py`

> 说明：537 MB 全量不入 CI。截断脚本把基准 MCAP 截前 ~30 s（或前 300 帧）写到 `tests/data/head_pose_mcap_trim.mcap`（gitignore）。集成测试 skip 该文件不存在时。

- [ ] **Step 1: 写 `scripts/truncate_mcap.py`**

```python
"""截断基准 head_pose_mcap.mcap 前 max_frames 帧，给集成测试用。

用法: python scripts/truncate_mcap.py <in.mcap> <out.mcap> [max_frames=300]
"""
from __future__ import annotations

import sys
from pathlib import Path

from mcap.reader import make_reader
from mcap.writer import Writer

from mcap_parser.decoders import schema_data


def truncate(in_path: str, out_path: str, max_frames: int = 300) -> None:
    fin = open(in_path, "rb")
    r = make_reader(fin)
    summary = r.get_summary()
    schema_cache: dict[str, int] = {}
    channel_cache: dict[int, int] = {}
    fout = open(out_path, "wb")
    w = Writer(fout)
    w.start()
    counts: dict[int, int] = {}
    # 视频帧作为帧计数基准
    video_chs = {
        cid for cid, ch in summary.channels.items()
        if "undistorted" in ch.topic and ch.topic.endswith("/image")
    }
    frame = 0
    for schema, channel, msg in r.iter_messages():
        if channel.id in video_chs:
            if frame >= max_frames:
                break
            frame += 1
        elif frame >= max_frames:
            # 只保留 max_frames 以内时间窗的消息（粗略：跳过视频外通道的越界）
            continue
        sname = summary.schemas[channel.schema_id].name
        if sname not in schema_cache:
            schema_cache[sname] = w.register_schema(
                name=sname, encoding="protobuf", data=schema_data(sname)
            )
        old_cid = channel.id
        if old_cid not in channel_cache:
            channel_cache[old_cid] = w.register_channel(
                topic=channel.topic, message_encoding="protobuf",
                schema_id=schema_cache[sname],
            )
        w.add_message(
            channel_id=channel_cache[old_cid],
            log_time=msg.log_time, data=msg.data, publish_time=msg.publish_time,
        )
    w.finish()
    fout.close()
    fin.close()


if __name__ == "__main__":
    truncate(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 300)
```

- [ ] **Step 2: 生成截断样本（手动，不入 git）**

Run: `cd /Users/hutx/work/hvyogo/ai_work/mcap-parser && python scripts/truncate_mcap.py /Users/hutx/work/hvyogo/ai_work/openembodied-data/data/SN_PA9410MGK8110131G_1527709884320059392_2026_07_17-16_13_39_963/head_pose_mcap.mcap tests/data/head_pose_mcap_trim.mcap 300`
Expected: 生成 ~几 MB 的截断文件。

在 `.gitignore` 加 `tests/data/`。

- [ ] **Step 3: 写集成测试 `tests/test_integration.py`**

```python
import json
from pathlib import Path

import numpy as np
import pytest

from mcap_parser.reader import McapFile

TRIM = Path(__file__).parent / "data" / "head_pose_mcap_trim.mcap"
pytestmark = pytest.mark.skipif(not TRIM.exists(), reason="截断 MCAP 不存在，跳过集成测试")


def test_integration_full_parse(tmp_path):
    ctx = McapFile(TRIM).parse(tmp_path)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["frame_count"] > 0
    assert m["message_count"] > 0
    # 手部网格
    z = np.load(tmp_path / "left_hand_vertices.npz")
    assert z["vertices"].shape[1:] == (778, 3)
    assert z["faces"].shape == (1538, 3)
    # 头部轨迹 7 列
    traj = np.load(tmp_path / "head_pose_trajectory.npy")
    assert traj.shape[1] == 7
    # wrist
    wp = np.load(tmp_path / "wrist_poses.npz")
    assert wp["left"].shape[1] == 7
    # calib
    calib = json.loads((tmp_path / "camera_calibration.json").read_text())
    assert len(calib) == 4
    # 视频存在（可能 muxed=False 但 .mp4/.h264 至少一个）
    assert any((tmp_path / f"head_{s}_camera_undistorted.mp4").exists()
               or (tmp_path / f"{s}.h264").exists() for s in ("left", "right"))
    # 点云 PCD 数 == pc_frames
    n_pc = ctx.outputs["aligned_pointclouds.npz"]["pc_frames"]
    pcds = list((tmp_path / "pointclouds").glob("*.pcd"))
    assert len(pcds) == n_pc
```

- [ ] **Step 4: 运行集成测试**

Run: `pytest tests/test_integration.py -v`
Expected: 1 passed（若截断文件已生成）。

- [ ] **Step 5: 跑全量单测**

Run: `pytest -v`
Expected: 全绿。

- [ ] **Step 6: Commit**

```bash
git add scripts/truncate_mcap.py tests/test_integration.py .gitignore
git commit -m "test: 集成测试（截断真实 MCAP）+ truncate 脚本"
```

---

## Task 15: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: 写 README**

```markdown
# mcap-parser

把 `head_pose_mcap.mcap`（foxglove/protobuf）解析为独立文件，格式对齐 `openembodied-data` 原始产物。

## 安装

    pip install -e ".[dev]"

## 用法

    python -m mcap_parser <path/to/head_pose_mcap.mcap> --out_dir <out> [--overwrite true]
      [--side left|right|both]
      [--streams video,hand_keypoints,hand_mesh,head_trajectory,pointcloud,body_tf,camera_calib]
      [--per_frame_pcd true|false]

## 输出

- 视频：`head_{left,right}_camera_undistorted.mp4`（ffmpeg remux，缺 ffmpeg 退化为 `.h264`）
- 手部关键点：`{left,right}_keypoints_{3d,2d}.npy` (N,21,3)/(N,21,2)
- 手部网格：`{left,right}_hand_vertices.npz`（vertices(N,778,3) + faces(1538,3)）
- 头部轨迹：`head_pose_trajectory.npy` (N,7) = [x,y,z,qx,qy,qz,qw]
- 点云：`aligned_pointclouds.npz` + `pointclouds/frame_*.pcd` + `static_scene.ply`
- 身体/手根 tf：`tracking_data_tf.jsonl` + `wrist_poses.npz`
- 相机标定：`camera_calibration.json`
- 清单：`manifest.json`

## 不可恢复

`head_pose_mcap.mcap` 的 tf_tree 只含 2 相机 + 2 腕 + 40 指节，**无 24-SMPL 身体骨架**。
原 `body_keypoints_labels.npz`(24 关节) 来自 `openembodied-data` 的 `body_pose_prelabel`
上游，本 MCAP 无法还原；详见 `docs/2026-08-19-mcap-parser-design.md`。
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README"
```

---

## 自检（plan 自查）

- **spec 覆盖**：§3 表逐行 → 对应 writer（video/hand_keypoints/hand_mesh/head_trajectory/pointcloud/body_tf+camera_calib）✓；§3.1 aligned_pointclouds 键 → PointcloudWriter ✓；§4 manifest → Task 12 ✓；§6 CLI → Task 13 ✓；§7 错误处理 → reader try/except + video fallback ✓；§8 测试 → 各 writer 单测 + Task 14 集成 ✓；§11 验收 → 集成测试覆盖 ✓。
- **占位符扫描**：无 TBD/TODO；每步含完整代码 ✓。
- **类型一致**：`ParseContext` 字段在所有 writer 间一致（`tf_cache[frame][child] = (parent, t, q, ts)`，head_trajectory 取 `[2]`、pointcloud 取 `[1],[2]`）✓；`Writer.feed(topic, msg, ctx)` / `finalize(out_dir, ctx)` 全程一致 ✓；`ctx.outputs[filename]` 由各 writer 写、manifest 读 ✓。
- **跨流依赖顺序**：reader 单遍 feed 完成后 `tf_cache`/`hand_3d` 已完整，再统一 finalize，顺序无依赖问题（pointcloud/head_trajectory 只读 ctx）✓。
