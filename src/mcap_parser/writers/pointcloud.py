"""点云 writer：ch6 -> pointclouds/frame_*.pcd + aligned_pointclouds.npz；ch7 -> static_scene.ply。"""
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
