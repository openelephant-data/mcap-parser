# iPhone MCAP 解析 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `mcap-parser` 能解析 iPhone/ARKit foxglove MCAP,产出与 `openelephant-data-app` 的 `IosMcapParseTask` 逐文件对齐的 7 个产物。

**Architecture:** 新增自包含 `IosMcapParser`(单遍 `iter_messages`,按 schema 名 + topic 路由,累积后写文件),与现有 PICO `McapFile` 并列;CLI `--profile {auto,iphone,pico}` 自动分流。foxglove schema 用现有 `foxglove_schemas_protobuf`;唯一需要的 oeego 自定义 proto(OnboardConfig)用 `FileDescriptorSet`→`DescriptorPool` 动态建类。复用 `decoders.decode_pointcloud`。

**Tech Stack:** Python 3.10+, mcap, foxglove-schemas-protobuf, numpy, opencv-python(cv2,测试里造 JPEG),fire,系统 ffmpeg(JPEG→mp4 重编码,可选)。

**Spec:** `docs/2026-09-02-ios-mcap-parse-design.md`

---

## 文件结构

| 文件 | 责任 | 动作 |
|---|---|---|
| `src/mcap_parser/decoders.py` | 新增 `oeego_msg_class()` 动态构建 oeego proto 类 | 修改 |
| `src/mcap_parser/ios_parse.py` | `IosMcapParser`:单遍 demux + 7 文件 + parse_report | 新建 |
| `src/mcap_parser/cli.py` | `--profile` 路由 | 修改 |
| `src/mcap_parser/_testing.py` | iPhone 消息构造器 + `build_ios_synth_mcap` | 修改 |
| `tests/conftest.py` | `ios_synth_mcap` fixture | 修改 |
| `tests/test_ios_parse.py` | 单测 | 新建 |
| `tests/test_ios_integration.py` | 真实截断 mcap 集成测试 | 新建 |

现有 PICO 路径(`reader.py`、`writers/`、`manifest.py`)**不动**。

---

## Task 1: oeego proto 动态解码助手

**Files:**
- Modify: `src/mcap_parser/decoders.py`(末尾追加)
- Test: `tests/test_ios_parse.py`(新建)

- [ ] **Step 1: 写失败测试**

创建 `tests/test_ios_parse.py`:

```python
"""iPhone MCAP 解析单测。"""
from __future__ import annotations

import base64

import numpy as np
import pytest

from mcap_parser._testing import build_ios_synth_mcap
from mcap_parser.decoders import oeego_msg_class

# 从真实 data/head_pose_mcap.mcap ch4 提取的 OnboardConfig schema(FileDescriptorSet)
ONBOARD_SCHEMA_B64 = (
    "Cq4HChBjb3JlX3Byb3RvLnByb3RvEhBvZWVnby5jb3JlX3Byb3RvImgKBkhlYWRlchIQCgNzZXEY"
    "ASABKA1SA3NlcRIxCgVzdGFtcBgCIAEoCzIbLm9lZWdvLmNvcmVfcHJvdG8uVGltZXN0YW1wUgVz"
    "dGFtcBIZCghmcmFtZV9pZBgDIAEoCVIHZnJhbWVJZCI1CglUaW1lc3RhbXASEgoEc2VjcxgBIAEo"
    "A1IEc2VjcxIUCgVuc2VjcxgCIAEoA1IFbnNlY3M"
)
# 注:上面是占位最小片段;实际用 _testing 里的完整常量(见 Task 2)
```

> 注:本测试在 Task 1 只验证 `oeego_msg_class` 能从一段 FileDescriptorSet 建出类并反序列化。完整 OnboardConfig schema 字节在 Task 2 落到 `_testing.py` 的 `ONBOARD_CONFIG_SCHEMA_B64` 常量;这里先 import 它。所以 Step 1 实际写成:

```python
"""iPhone MCAP 解析单测。"""
from __future__ import annotations

import numpy as np
import pytest

from mcap_parser._testing import (
    ONBOARD_CONFIG_SCHEMA_B64,
    build_ios_synth_mcap,
)
from mcap_parser.decoders import oeego_msg_class


def test_oeego_msg_class_builds_and_parses():
    schema_data = __import__("base64").b64decode(ONBOARD_CONFIG_SCHEMA_B64)
    cls = oeego_msg_class(
        "oeego.system_proto.MissionManagerProto.OnboardConfig", schema_data
    )
    m = cls()
    m.mission_id = "123"
    m.target_frame_rate = 30
    roundtrip = cls()
    roundtrip.ParseFromString(m.SerializeToString())
    assert roundtrip.mission_id == "123"
    assert roundtrip.target_frame_rate == 30
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_oeego_msg_class_builds_and_parses -v`
Expected: FAIL(`ImportError: cannot import name 'oeego_msg_class'` 或 `ONBOARD_CONFIG_SCHEMA_B64`)

- [ ] **Step 3: 实现 `oeego_msg_class`**

在 `src/mcap_parser/decoders.py` 顶部 import 区追加:

```python
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
```

在文件末尾追加:

```python
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
```

- [ ] **Step 4: 落地 `ONBOARD_CONFIG_SCHEMA_B64` 常量(测试前置依赖)**

在 `src/mcap_parser/_testing.py` 顶部(import 之后)追加从真实 mcap 提取的完整 OnboardConfig schema bytes。用一条命令把 base64 写入文件,避免手抄:

Run(提取并写入常量,生成一段 Python 赋值):
```bash
.venv/bin/python - <<'PY'
import base64
from mcap.reader import make_reader
with open("data/head_pose_mcap.mcap","rb") as f:
    s=make_reader(f).get_summary()
    sch=s.schemas[s.channels[4].schema_id]
    b=base64.b64encode(sch.data).decode()
print(b)
PY
```

把输出作为 `ONBOARD_CONFIG_SCHEMA_B64 = "<输出>"` 一行加到 `_testing.py` 顶部。这是一行长字符串常量。

- [ ] **Step 5: 运行测试验证通过**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_oeego_msg_class_builds_and_parses -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/mcap_parser/decoders.py src/mcap_parser/_testing.py tests/test_ios_parse.py
git commit -m "feat(ios): oeego proto 动态解码助手 + OnboardConfig schema 常量"
```

---

## Task 2: iPhone 合成 mcap fixture

**Files:**
- Modify: `src/mcap_parser/_testing.py`(追加消息构造器 + `build_ios_synth_mcap`)
- Modify: `tests/conftest.py`(追加 `ios_synth_mcap` fixture)
- Test: `tests/test_ios_parse.py`

- [ ] **Step 1: 写失败测试(fixture 可打开且通道齐全)**

追加到 `tests/test_ios_parse.py`:

```python
from mcap.reader import make_reader


def test_ios_synth_fixture_has_channels(ios_synth_mcap):
    with open(ios_synth_mcap, "rb") as f:
        s = make_reader(f).get_summary()
    topics = {ch.topic for ch in s.channels.values()}
    assert "/cam_head_l/color/image_compressed_h265" in topics
    assert "/cam_head_l/depth/image_raw" in topics
    assert "/cam_head_l/depth/confidence" in topics
    assert "/cam_head_l/color/camera_info" in topics
    assert "/cam_head_l/depth/camera_info" in topics
    assert "/tf" in topics
    assert "/cam_head_l/ego_hand_pose/pointcloud_3d" in topics
    assert "/oeego/mission_manager/onboard_config" in topics
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_ios_synth_fixture_has_channels -v`
Expected: FAIL(`fixture 'ios_synth_mcap' not found`)

- [ ] **Step 3: 实现 iPhone 消息构造器 + `build_ios_synth_mcap`**

在 `src/mcap_parser/_testing.py` 追加(import 区补 `CompressedImage_pb2`、`cv2` 按需):

```python
import base64

import cv2
from foxglove_schemas_protobuf import (
    CameraCalibration_pb2,
    CompressedImage_pb2,
    FrameTransforms_pb2,
    PointCloud_pb2,
    RawImage_pb2,
)
from mcap.writer import Writer

from .decoders import oeego_msg_class, schema_data

IOS_BASE_TS = 1_788_000_000_000_000_000  # epoch ns
IOS_STEP = 33_333_333  # ~30fps


def _jpeg_bytes(i: int) -> bytes:
    """用 cv2 编码一张 16x16 的真 JPEG,供 video 重编码测试。"""
    img = np.full((16, 16, 3), (10 * (i + 1)) % 256, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def ios_color_msg(ts: int, i: int) -> FakeMsg:
    m = CompressedImage_pb2.CompressedImage()
    m.timestamp.seconds = ts // 1_000_000_000
    m.timestamp.nanos = ts % 1_000_000_000
    m.frame_id = "head_left_camera_color_optical_frame"
    m.format = "jpeg"
    m.data = _jpeg_bytes(i)
    return FakeMsg(m.SerializeToString(), ts)


def ios_raw_image_msg(ts: int, frame_id: str, w: int = 4, h: int = 4) -> FakeMsg:
    m = RawImage_pb2.RawImage()
    m.timestamp.seconds = ts // 1_000_000_000
    m.timestamp.nanos = ts % 1_000_000_000
    m.frame_id = frame_id
    m.width, m.height = w, h
    m.encoding = "8UC1"
    m.step = w
    m.data = bytes(range(w * h))
    return FakeMsg(m.SerializeToString(), ts)


def ios_camera_info_msg(ts: int, frame_id: str, w: int, h: int) -> FakeMsg:
    m = CameraCalibration_pb2.CameraCalibration()
    m.timestamp.seconds = ts // 1_000_000_000
    m.timestamp.nanos = ts % 1_000_000_000
    m.width, m.height = w, h
    m.distortion_model = "plumb_bob"
    m.D.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    m.K.extend([1343.3, 0, 959.1, 0, 1343.3, 718.7, 0, 0, 1])
    m.R.extend([1, 0, 0, 0, 1, 0, 0, 0, 1])
    m.P.extend([1343.3, 0, 959.1, 0, 0, 1343.3, 718.7, 0, 0, 0, 1, 0])
    m.frame_id = frame_id
    return FakeMsg(m.SerializeToString(), ts)


def ios_tf_msg(ts: int) -> FakeMsg:
    m = FrameTransforms_pb2.FrameTransforms()
    ft = m.transforms.add()
    ft.timestamp.seconds = ts // 1_000_000_000
    ft.timestamp.nanos = ts % 1_000_000_000
    ft.parent_frame_id = "world"
    ft.child_frame_id = "head_left_camera_color_optical_frame"
    ft.translation.x, ft.translation.y, ft.translation.z = -0.07, 0.35, 0.13
    ft.rotation.x, ft.rotation.y, ft.rotation.z, ft.rotation.w = -0.49, -0.06, -0.015, 0.87
    return FakeMsg(m.SerializeToString(), ts)


def ios_hand3d_msg(ts: int, n: int = 21) -> FakeMsg:
    m = PointCloud_pb2.PointCloud()
    m.timestamp.seconds = ts // 1_000_000_000
    m.timestamp.nanos = ts % 1_000_000_000
    m.frame_id = "head_left_camera_color_optical_frame"
    m.point_stride = 12
    for name, off, typ in [("x", 0, 9), ("y", 4, 9), ("z", 8, 9)]:
        f = m.fields.add()
        f.name, f.offset, f.type = name, off, typ
    pts = np.random.rand(n, 3).astype(np.float32)
    m.data = pts.tobytes()
    return FakeMsg(m.SerializeToString(), ts)


def ios_onboard_msg(ts: int) -> FakeMsg:
    cls = oeego_msg_class(
        "oeego.system_proto.MissionManagerProto.OnboardConfig",
        base64.b64decode(ONBOARD_CONFIG_SCHEMA_B64),
    )
    m = cls()
    m.mission_id = "1543953655491727360"
    m.target_frame_rate = 30
    return FakeMsg(m.SerializeToString(), ts)


IOS_SCHEMA_TOPICS = [
    ("foxglove.CompressedImage", "/cam_head_l/color/image_compressed_h265"),
    ("foxglove.RawImage", "/cam_head_l/depth/image_raw"),
    ("foxglove.RawImage", "/cam_head_l/depth/confidence"),
    ("foxglove.CameraCalibration", "/cam_head_l/color/camera_info"),
    ("foxglove.CameraCalibration", "/cam_head_l/depth/camera_info"),
    ("foxglove.FrameTransforms", "/tf"),
    ("foxglove.PointCloud", "/cam_head_l/ego_hand_pose/pointcloud_3d"),
    ("oeego.system_proto.MissionManagerProto.OnboardConfig",
     "/oeego/mission_manager/onboard_config"),
]


def build_ios_synth_mcap(p) -> Path:
    """3 帧 iPhone 风格 foxglove MCAP。"""
    p = Path(p)
    f = open(p, "wb")
    w = Writer(f)
    w.start()
    # schema: foxglove 用 schema_data();oeego 用 b64 解码
    schema_bytes = {}
    cids = {}
    for sch_name, topic in IOS_SCHEMA_TOPICS:
        if sch_name not in schema_bytes:
            if sch_name.startswith("oeego."):
                # OnboardConfig
                schema_bytes[sch_name] = base64.b64decode(ONBOARD_CONFIG_SCHEMA_B64)
            else:
                schema_bytes[sch_name] = schema_data(sch_name)
        sid = w.register_schema(name=sch_name, encoding="protobuf", data=schema_bytes[sch_name])
        cids[topic] = w.register_channel(topic=topic, message_encoding="protobuf", schema_id=sid)
    for i in range(3):
        ts = IOS_BASE_TS + i * IOS_STEP
        msgs = [
            ("/cam_head_l/color/image_compressed_h265", ios_color_msg(ts, i).data),
            ("/cam_head_l/depth/image_raw",
             ios_raw_image_msg(ts, "head_left_camera_depth_optical_frame").data),
            ("/cam_head_l/depth/confidence",
             ios_raw_image_msg(ts, "head_left_camera_depth_optical_frame").data),
            ("/cam_head_l/color/camera_info",
             ios_camera_info_msg(ts, "head_left_camera_color_optical_frame", 1920, 1440).data),
            ("/cam_head_l/depth/camera_info",
             ios_camera_info_msg(ts, "head_left_camera_depth_optical_frame", 256, 192).data),
            ("/tf", ios_tf_msg(ts).data),
            ("/cam_head_l/ego_hand_pose/pointcloud_3d", ios_hand3d_msg(ts).data),
            ("/oeego/mission_manager/onboard_config", ios_onboard_msg(ts).data),
        ]
        for topic, data in msgs:
            w.add_message(channel_id=cids[topic], log_time=ts, data=data, publish_time=ts)
    w.finish()
    f.close()
    return p
```

> 注:`FakeMsg` 已在 `_testing.py` 定义;`schema_data` 已从 `.decoders` import(顶部已有)。`Path` 已 import。

- [ ] **Step 4: 实现 conftest fixture**

在 `tests/conftest.py` 追加:

```python
from mcap_parser._testing import build_ios_synth_mcap


@pytest.fixture
def ios_synth_mcap(tmp_path) -> Path:
    """3 帧 iPhone 风格 foxglove MCAP。"""
    return build_ios_synth_mcap(tmp_path / "ios_synth.mcap")
```

- [ ] **Step 5: 运行验证通过**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_ios_synth_fixture_has_channels -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/mcap_parser/_testing.py tests/conftest.py tests/test_ios_parse.py
git commit -m "test(ios): iPhone 风格合成 mcap fixture"
```

---

## Task 3: IosMcapParser 骨架 + parse_report channels[]

**Files:**
- Create: `src/mcap_parser/ios_parse.py`
- Test: `tests/test_ios_parse.py`

- [ ] **Step 1: 写失败测试(骨架能跑完,写 parse_report,channels 齐全;本任务不验证数据产物)**

追加到 `tests/test_ios_parse.py`:

```python
import json

from mcap_parser.ios_parse import IosMcapParser


def test_ios_parse_writes_report_channels(ios_synth_mcap, tmp_path):
    out = tmp_path / "out"
    IosMcapParser(ios_synth_mcap).parse(out, overwrite=True)
    report = json.loads((out / "parse_report.json").read_text())
    assert report["case_id"] == "1543953655491727360"
    topics = {c["topic"] for c in report["channels"]}
    assert "/cam_head_l/color/image_compressed_h265" in topics
    assert "/tf" in topics
    assert len(report["channels"]) == 8  # 8 个通道
    # 每条 channel 记录都有 count/bytes/span_s/hz/schema
    for c in report["channels"]:
        assert c["count"] == 3
        assert c["schema"]
        assert c["span_s"] >= 0
        assert c["hz"] >= 0
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_ios_parse_writes_report_channels -v`
Expected: FAIL(`No module named 'mcap_parser.ios_parse'`)

- [ ] **Step 3: 实现 `IosMcapParser` 骨架**

创建 `src/mcap_parser/ios_parse.py`:

```python
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
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_ios_parse_writes_report_channels -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/mcap_parser/ios_parse.py tests/test_ios_parse.py
git commit -m "feat(ios): IosMcapParser 骨架 + parse_report channels[]"
```

---

## Task 4: camera_info.json 产物

**Files:**
- Modify: `src/mcap_parser/ios_parse.py`(`parse` 内 outputs 写入段)
- Test: `tests/test_ios_parse.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_ios_parse.py`:

```python
def test_ios_parse_camera_info(ios_synth_mcap, tmp_path):
    out = tmp_path / "out"
    IosMcapParser(ios_synth_mcap).parse(out, overwrite=True)
    ci = json.loads((out / "camera_info.json").read_text())
    assert set(ci.keys()) == {
        "head_left_camera_color_optical_frame",
        "head_left_camera_depth_optical_frame",
    }
    color = ci["head_left_camera_color_optical_frame"]
    assert color["width"] == 1920 and color["height"] == 1440
    assert color["distortion_model"] == "plumb_bob"
    assert len(color["K"]) == 9
    assert len(color["R"]) == 9
    assert len(color["P"]) == 12
    assert color["frame_id"] == "head_left_camera_color_optical_frame"
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_ios_parse_camera_info -v`
Expected: FAIL(`camera_info.json` 不存在)

- [ ] **Step 3: 实现产物写入**

在 `src/mcap_parser/ios_parse.py` 的 `parse` 方法中,把 `# 各产物写入在后续 Task 逐步填充` 注释行替换为:

```python
            if accum["camera_infos"]:
                (out_dir / "camera_info.json").write_text(
                    json.dumps(accum["camera_infos"], ensure_ascii=False, indent=2))
                outputs["camera_info"] = out_dir / "camera_info.json"
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_ios_parse_camera_info -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/mcap_parser/ios_parse.py tests/test_ios_parse.py
git commit -m "feat(ios): camera_info.json 产物"
```

---

## Task 5: transforms.npz 产物

**Files:**
- Modify: `src/mcap_parser/ios_parse.py`
- Test: `tests/test_ios_parse.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_ios_parse.py`:

```python
def test_ios_parse_transforms(ios_synth_mcap, tmp_path):
    out = tmp_path / "out"
    IosMcapParser(ios_synth_mcap).parse(out, overwrite=True)
    z = np.load(out / "transforms.npz")
    arr = z["transforms"]
    assert arr.shape == (3, 8)
    assert arr.dtype == np.float64
    # 前 7 列为位姿,第 8 列为 ts ns(epoch 量级)
    assert abs(arr[0, 0] - (-0.07)) < 1e-6
    assert abs(arr[0, 3] - (-0.49)) < 1e-6
    assert arr[0, 7] > 1_000_000_000  # epoch ns
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_ios_parse_transforms -v`
Expected: FAIL(`transforms.npz` 不存在)

- [ ] **Step 3: 实现产物写入**

在 `parse` 方法中 `camera_info` 写入块**之前**追加(顺序无关,但放在 camera_info 前便于读):

```python
            if accum["transforms"]:
                arr = np.array(accum["transforms"], dtype=np.float64)
                np.savez_compressed(out_dir / "transforms.npz", transforms=arr)
                outputs["transforms"] = out_dir / "transforms.npz"
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_ios_parse_transforms -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/mcap_parser/ios_parse.py tests/test_ios_parse.py
git commit -m "feat(ios): transforms.npz 产物"
```

---

## Task 6: depth_frames.npz / confidence_frames.npz 产物

**Files:**
- Modify: `src/mcap_parser/ios_parse.py`
- Test: `tests/test_ios_parse.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_ios_parse.py`:

```python
def test_ios_parse_depth_and_confidence(ios_synth_mcap, tmp_path):
    out = tmp_path / "out"
    IosMcapParser(ios_synth_mcap).parse(out, overwrite=True)
    for name, fmt_prefix in [("depth_frames", "8UC1;4x4"),
                             ("confidence_frames", "8UC1;4x4")]:
        z = np.load(out / f"{name}.npz", allow_pickle=True)
        assert z["timestamps"].shape == (3,)
        assert z["timestamps"].dtype == np.int64
        assert z["frames"].shape == (3,)
        assert z["frames"].dtype == object
        assert isinstance(z["frames"][0], bytes)
        assert z["formats"][0] == fmt_prefix
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_ios_parse_depth_and_confidence -v`
Expected: FAIL(`depth_frames.npz` 不存在)

- [ ] **Step 3: 实现产物写入**

在 `parse` 方法中追加(放在 transforms 块后):

```python
            if accum["depth"]:
                self._save_raw_frames(accum["depth"], out_dir / "depth_frames.npz")
                outputs["depth_frames"] = out_dir / "depth_frames.npz"
            if accum["conf"]:
                self._save_raw_frames(accum["conf"], out_dir / "confidence_frames.npz")
                outputs["confidence_frames"] = out_dir / "confidence_frames.npz"
```

并在 `IosMcapParser` 类中追加静态方法(放在 `_channel_entries` 之后):

```python
    @staticmethod
    def _save_raw_frames(frames, path: Path) -> None:
        """无损:逐帧原始 bytes + timestamp + format。"""
        ts = np.array([f[0] for f in frames], dtype=np.int64)
        raw = np.array([f[1] for f in frames], dtype=object)
        fmts = np.array([f[2] for f in frames], dtype=object)
        np.savez_compressed(path, timestamps=ts, frames=raw, formats=fmts)
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_ios_parse_depth_and_confidence -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/mcap_parser/ios_parse.py tests/test_ios_parse.py
git commit -m "feat(ios): depth/confidence frames npz 产物"
```

---

## Task 7: hand_keypoints_3d.npz 产物

**Files:**
- Modify: `src/mcap_parser/ios_parse.py`
- Test: `tests/test_ios_parse.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_ios_parse.py`:

```python
def test_ios_parse_hand_keypoints_3d(ios_synth_mcap, tmp_path):
    out = tmp_path / "out"
    IosMcapParser(ios_synth_mcap).parse(out, overwrite=True)
    z = np.load(out / "hand_keypoints_3d.npz", allow_pickle=True)
    lm = z["landmarks"]
    ts = z["timestamps"]
    assert lm.shape == (3,)
    assert lm.dtype == object
    assert lm[0].shape == (21, 3)
    assert lm[0].dtype == np.float32
    assert ts.shape == (3,)
    assert ts.dtype == np.int64
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_ios_parse_hand_keypoints_3d -v`
Expected: FAIL(`hand_keypoints_3d.npz` 不存在)

- [ ] **Step 3: 实现产物写入**

在 `parse` 方法中追加(放在 confidence 块后):

```python
            if accum["hand_3d"]:
                lm = np.array([p for p, _ in accum["hand_3d"]], dtype=object)
                ts = np.array([t for _, t in accum["hand_3d"]], dtype=np.int64)
                np.savez_compressed(out_dir / "hand_keypoints_3d.npz",
                                    landmarks=lm, timestamps=ts)
                outputs["hand_keypoints_3d"] = out_dir / "hand_keypoints_3d.npz"
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_ios_parse_hand_keypoints_3d -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/mcap_parser/ios_parse.py tests/test_ios_parse.py
git commit -m "feat(ios): hand_keypoints_3d.npz 产物"
```

---

## Task 8: video_head.mp4 产物(JPEG → libx264 重编码)

**Files:**
- Modify: `src/mcap_parser/ios_parse.py`
- Test: `tests/test_ios_parse.py`

- [ ] **Step 1: 写失败测试(无 ffmpeg 时 skip)**

追加到 `tests/test_ios_parse.py`:

```python
import shutil
import subprocess


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def test_ios_parse_video_head(ios_synth_mcap, tmp_path):
    if not _has_ffmpeg():
        pytest.skip("ffmpeg not installed")
    out = tmp_path / "out"
    IosMcapParser(ios_synth_mcap).parse(out, overwrite=True)
    mp4 = out / "video_head.mp4"
    assert mp4.exists() and mp4.stat().st_size > 0
    import cv2
    cap = cv2.VideoCapture(str(mp4))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    cap.release()
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_ios_parse_video_head -v`
Expected: FAIL(`video_head.mp4` 不存在) 或 skip

- [ ] **Step 3: 实现产物写入 + ffmpeg 重编码**

在 `ios_parse.py` 顶部 import 区追加:

```python
import subprocess
import tempfile
```

在 `parse` 方法中追加(放在 hand_3d 块后、camera_info 块前——顺序不影响):

```python
            nals = accum["video_nals"]
            jpegs = accum["video_jpegs"]
            if jpegs or nals:
                fps = accum["oc_fps"] or 0
                self._write_video(out_dir / "video_head.mp4", jpegs, nals, fps,
                                 keep_frames_on_fail)
                if (out_dir / "video_head.mp4").exists():
                    outputs["video_head"] = out_dir / "video_head.mp4"
                elif keep_frames_on_fail and (out_dir / "color_frames").exists():
                    outputs["video_head"] = out_dir / "color_frames"
```

在 `IosMcapParser` 类中追加方法(放在 `_save_raw_frames` 后):

```python
    def _write_video(self, dst: Path, jpegs, nals, fps: int,
                     keep_frames_on_fail: bool) -> None:
        """JPEG 序列或 HEVC NAL -> libx264 mp4。

        JPEG:image2 输入逐帧 jpg;HEVC:-f hevc demux。均 libx264 重编码,
        -bf 0 -g fps。fps 来自 onboard_config.target_frame_rate,缺则按 log_time 计算。
        """
        if not jpegs and not nals:
            return
        if fps <= 0:
            lts = [t for t, _ in (jpegs + nals)]
            span = (max(lts) - min(lts)) / 1e9
            fps = max(round(len(lts) / span), 1) if span > 0 else 30
        tmp = tempfile.mkdtemp(prefix="iosvideo_")
        try:
            if jpegs:
                for i, (_, b) in enumerate(jpegs):
                    (Path(tmp) / f"frame_{i:06d}.jpg").write_bytes(b)
                cmd = ["ffmpeg", "-y", "-framerate", str(fps),
                       "-i", str(Path(tmp) / "frame_%06d.jpg"),
                       "-c:v", "libx264", "-bf", "0", "-g", str(max(fps, 1)),
                       "-preset", "fast", "-pix_fmt", "yuv420p", "-an", str(dst)]
            else:
                inp = Path(tmp) / "stream.h265"
                inp.write_bytes(b"".join(b for _, b in nals))
                cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-f", "hevc",
                       "-i", str(inp),
                       "-c:v", "libx264", "-bf", "0", "-g", str(max(fps, 1)),
                       "-preset", "fast", "-pix_fmt", "yuv420p", "-an", str(dst)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
                if keep_frames_on_fail and jpegs:
                    import shutil as _sh
                    _sh.copytree(tmp, dst.parent / "color_frames",
                                 dirs_exist_ok=True)
        finally:
            import shutil as _sh
            _sh.rmtree(tmp, ignore_errors=True)
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_ios_parse_video_head -v`
Expected: PASS(或 skip 若无 ffmpeg)

- [ ] **Step 5: 提交**

```bash
git add src/mcap_parser/ios_parse.py tests/test_ios_parse.py
git commit -m "feat(ios): video_head.mp4(JPEG/HEVC -> libx264 重编码)"
```

---

## Task 9: parse_report outputs/counts 完整 + 回归

**Files:**
- Test: `tests/test_ios_parse.py`

- [ ] **Step 1: 写完整 parse_report 测试**

追加到 `tests/test_ios_parse.py`:

```python
def test_ios_parse_report_full(ios_synth_mcap, tmp_path):
    out = tmp_path / "out"
    IosMcapParser(ios_synth_mcap).parse(out, overwrite=True)
    report = json.loads((out / "parse_report.json").read_text())
    keys = {o["key"] for o in report["outputs"]}
    assert "parse_report" in keys
    assert "camera_info" in keys
    assert "transforms" in keys
    assert "depth_frames" in keys
    assert "confidence_frames" in keys
    assert "hand_keypoints_3d" in keys
    # video_head 仅在有 ffmpeg 时存在
    for key in ("camera_info", "transforms", "depth_frames",
                "confidence_frames", "hand_keypoints_3d", "parse_report"):
        e = next(o for o in report["outputs"] if o["key"] == key)
        assert e["desc"]
    assert report["case_id"] == "1543953655491727360"
    assert report["transform_count"] == 3
    assert report["hand_3d_count"] == 3
    assert report["hand_2d_count"] == 0
    assert report["depth_frame_count"] == 3
    assert report["confidence_frame_count"] == 3
    assert report["video_nal_count"] == 3
    assert set(report["camera_infos"]) == {
        "head_left_camera_color_optical_frame",
        "head_left_camera_depth_optical_frame",
    }
```

- [ ] **Step 2: 运行验证通过(实现已在 Task 3-8 完成,此为回归)**

Run: `.venv/bin/pytest tests/test_ios_parse.py -v`
Expected: PASS(全部 iPhone 单测)

- [ ] **Step 3: 提交**

```bash
git add tests/test_ios_parse.py
git commit -m "test(ios): parse_report 完整字段回归"
```

---

## Task 10: CLI --profile 路由

**Files:**
- Modify: `src/mcap_parser/cli.py`
- Test: `tests/test_ios_parse.py`

- [ ] **Step 1: 写失败测试(CLI 自动识别 iPhone 走 IosMcapParser)**

追加到 `tests/test_ios_parse.py`:

```python
import sys


def test_cli_auto_routes_iphone(ios_synth_mcap, tmp_path):
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "-m", "mcap_parser", str(ios_synth_mcap),
         "--out_dir", str(out), "--overwrite", "true"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    # iPhone 产物标志:parse_report.json + transforms.npz(非 PICO 的 manifest.json)
    assert (out / "parse_report.json").exists()
    assert (out / "transforms.npz").exists()
    assert not (out / "manifest.json").exists()


def test_cli_explicit_profile_pico_still_works(synth_mcap, tmp_path):
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "-m", "mcap_parser", str(synth_mcap),
         "--out_dir", str(out), "--overwrite", "true",
         "--profile", "pico"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "manifest.json").exists()
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_cli_auto_routes_iphone -v`
Expected: FAIL(PICO 路径产出 manifest.json,无 parse_report.json)

- [ ] **Step 3: 实现 CLI 路由**

把 `src/mcap_parser/cli.py` 整体替换为:

```python
"""CLI: python -m mcap_parser <mcap> --out_dir <dir> [--profile ...]。

--profile {auto,iphone,pico} 默认 auto:iPhone 文件(topic 含 /cam_head_l/)走
IosMcapParser,否则走 PICO McapFile。
"""
from __future__ import annotations

from pathlib import Path

from mcap.reader import make_reader

from .ios_parse import IosMcapParser
from .reader import McapFile


def _detect_profile(mcap_path: str) -> str:
    with open(mcap_path, "rb") as f:
        s = make_reader(f).get_summary()
    topics = {ch.topic for ch in s.channels.values()}
    if any(t.startswith("/cam_head_l/") for t in topics):
        return "iphone"
    return "pico"


def main(
    mcap_path: str,
    out_dir: str,
    side: str = "both",
    streams: str | None = None,
    per_frame_pcd: bool = True,
    profile: str = "auto",
    keep_frames_on_fail: bool = False,
    overwrite: bool = False,
) -> None:
    resolved = profile if profile != "auto" else _detect_profile(mcap_path)
    if resolved == "iphone":
        outs = IosMcapParser(mcap_path).parse(
            out_dir, overwrite=overwrite, keep_frames_on_fail=keep_frames_on_fail)
        print(f"[mcap_parser] iphone done: {len(outs)} outputs in {out_dir}")
        return
    stream_list = streams.split(",") if isinstance(streams, str) else streams
    ctx = McapFile(mcap_path).parse(
        out_dir, side=side, streams=stream_list,
        per_frame_pcd=per_frame_pcd, overwrite=overwrite,
    )
    print(f"[mcap_parser] pico done: {len(ctx.outputs)} outputs in {out_dir}")
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/pytest tests/test_ios_parse.py::test_cli_auto_routes_iphone tests/test_ios_parse.py::test_cli_explicit_profile_pico_still_works -v`
Expected: PASS

- [ ] **Step 5: 全量单测回归**

Run: `.venv/bin/pytest tests/test_ios_parse.py tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/mcap_parser/cli.py tests/test_ios_parse.py
git commit -m "feat(ios): CLI --profile 路由(auto/iphone/pico)"
```

---

## Task 11: 真实截断 mcap 集成测试

**Files:**
- Create: `tests/test_ios_integration.py`
- Test: 依赖 `data/head_pose_mcap.mcap` 截断版

- [ ] **Step 1: 造截断 mcap(若不存在)**

检查 `scripts/truncate_mcap.py` 是否能截断。先看它接口:

Run: `.venv/bin/python scripts/truncate_mcap.py --help 2>&1 | head -20 || true`

若脚本存在且支持按帧/秒截断,生成截断文件(前 ~900 帧,约 30s):

```bash
.venv/bin/python scripts/truncate_mcap.py data/head_pose_mcap.mcap \
    tests/data/head_pose_trunc.mcap --frames 900 || \
.venv/bin/python scripts/truncate_mcap.py data/head_pose_mcap.mcap \
    tests/data/head_pose_trunc.mcap --seconds 30
```

若脚本接口不同,读 `scripts/truncate_mcap.py` 源码调整命令。生成后 `tests/data/head_pose_trunc.mcap` 应 < 100 MB。

- [ ] **Step 2: 写集成测试**

创建 `tests/test_ios_integration.py`:

```python
"""真实 head_pose_mcap.mcap 截断版集成测试(不入 CI fast path,需文件存在)。"""
from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

TRUNC = Path(__file__).parent / "data" / "head_pose_trunc.mcap"
pytestmark = pytest.mark.skipif(
    not TRUNC.exists(), reason=f"截断 mcap 不存在: {TRUNC}"
)


def test_ios_parse_real_truncated(tmp_path):
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "-m", "mcap_parser", str(TRUNC),
         "--out_dir", str(out), "--overwrite", "true"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads((out / "parse_report.json").read_text())
    assert report["case_id"] == "1543953655491727360"
    assert report["transform_count"] > 0
    assert report["depth_frame_count"] > 0
    assert report["confidence_frame_count"] > 0
    assert report["hand_3d_count"] > 0
    assert set(report["camera_infos"]) == {
        "head_left_camera_color_optical_frame",
        "head_left_camera_depth_optical_frame",
    }
    # 形状
    tr = np.load(out / "transforms.npz")["transforms"]
    assert tr.shape[1] == 8
    df = np.load(out / "depth_frames.npz", allow_pickle=True)
    assert df["frames"][0]  # bytes 非空
    hk = np.load(out / "hand_keypoints_3d.npz", allow_pickle=True)
    assert hk["landmarks"][0].shape[1] == 3
    # video_head.mp4(若 ffmpeg)
    if (out / "video_head.mp4").exists():
        import cv2
        cap = cv2.VideoCapture(str(out / "video_head.mp4"))
        assert cap.isOpened()
        cap.release()
```

- [ ] **Step 3: 运行集成测试**

Run: `.venv/bin/pytest tests/test_ios_integration.py -v`
Expected: PASS(若截断文件已生成)。若 `scripts/truncate_mcap.py` 无法按需截断,见 Step 1 备注。

- [ ] **Step 4: 全量回归**

Run: `.venv/bin/pytest -v`
Expected: 全绿(集成测试在截断文件存在时跑,否则 skip)。

- [ ] **Step 5: 提交**

```bash
git add tests/test_ios_integration.py tests/data/head_pose_trunc.mcap
git commit -m "test(ios): 真实截断 mcap 集成测试"
```

> 注:若 `tests/data/head_pose_trunc.mcap` 体积过大不宜入 git,把它加进 `.gitignore`,集成测试在本地跑;CI 上 skip。

---

## Self-Review

**Spec coverage:**
- §3 video_head.mp4 → Task 8 ✓
- §3 depth_frames.npz / confidence_frames.npz → Task 6 ✓
- §3 camera_info.json → Task 4 ✓
- §3 transforms.npz → Task 5 ✓
- §3 hand_keypoints_3d.npz → Task 7 ✓
- §4 parse_report.json → Task 3(骨架)+ Task 9(完整)✓
- §5 架构 / IosMcapParser → Task 3 ✓
- §5.3 video JPEG→libx264 失败语义 → Task 8 ✓
- §5.3 oeego onboard_config 解码 → Task 1 + Task 3 `_feed` ✓
- §6 CLI --profile → Task 10 ✓
- §8 测试策略(单测 + 集成)→ Task 2-9 + Task 11 ✓
- §11 验收标准 → 对应各 Task 断言 ✓

**Placeholder scan:** 无 TBD/TODO。Task 1 Step 1 的占位 b64 片段已明确标注"实际用 _testing 里的完整常量",并落到 Step 4 命令提取——非占位。Task 11 Step 1 对 `truncate_mcap.py` 接口的不确定性已给出"读源码调整"的明确兜底。

**Type 一致性:**
- `IosMcapParser.parse(out_dir, *, overwrite, keep_frames_on_fail)` 全程一致。
- `_feed(channel, msg, chan_info, oc_cls, accum)` 全程一致。
- 累加器键名 `video_jpegs/video_nals/depth/conf/camera_infos/transforms/hand_3d/oc_fps/case_id/summary` 在 Task 3 定义、Task 4-8 使用,无漂移。
- `_save_raw_frames` / `_write_video` / `_channel_entries` / `_resolve_onboard_cls` / `_case_id_from_name` 命名一致。
- `oeego_msg_class(schema_name, schema_data)` 签名在 Task 1 定义、Task 2/3 使用一致。
