"""手部网格 writer：ch5 -> left/right_hand_vertices.npz。"""
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
