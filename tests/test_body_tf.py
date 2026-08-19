import json

import numpy as np

from mcap_parser._testing import tf_msg
from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.body_tf import BodyTfWriter


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
