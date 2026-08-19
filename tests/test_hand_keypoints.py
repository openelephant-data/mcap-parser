import numpy as np

from mcap_parser._testing import hand_kp_2d_msg, hand_kp_3d_msg
from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.hand_keypoints import HandKeypointsWriter


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
    # 只给左手 3D，右手缺失 -> 右手应为全 nan
    w.feed("/pose_algo_body/hand_keypoints", hand_kp_3d_msg(ts, left=True, right=False), ctx)
    w.feed("/pose_algo_body/hand_keypoints_2d", hand_kp_2d_msg(ts), ctx)
    w.finalize(tmp_path, ctx)

    right = np.load(tmp_path / "right_keypoints_3d.npy")
    assert right.shape == (1, 21, 3)
    assert np.isnan(right).all()
