"""McapFile：单遍流式分发 reader。"""
from __future__ import annotations

import shutil
from pathlib import Path

from mcap.reader import make_reader

from .writers.base import ParseContext, Writer
from .writers.body_tf import BodyTfWriter
from .writers.camera_calib import CameraCalibWriter
from .writers.depth import DepthWriter
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
    "depth": DepthWriter,
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
        summary = r.get_summary()
        for schema, channel, msg in r.iter_messages():
            topic = channel.topic
            for w in writers:
                if w.owns_topic(topic):
                    try:
                        w.feed(topic, msg, ctx)
                    except Exception as e:  # 解码失败不中断
                        print(f"[warn] feed {topic} failed: {e}")
        ctx.frame_count = len(ctx.tf_cache) or len(ctx.head_pos)

        # finalize 顺序：body_tf（已写 tf_cache）-> pointcloud/head_trajectory 依赖 ctx
        for w in writers:
            w.finalize(out_dir, ctx)

        from .manifest import ManifestBuilder
        ManifestBuilder(self.path, summary, ctx, writers).write(out_dir)
        f.close()
        return ctx
