import numpy as np

from mcap_parser._testing import hand_mesh_msg
from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.hand_mesh import HandMeshWriter


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
