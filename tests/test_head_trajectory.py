import numpy as np

from mcap_parser._testing import head_traj_msg, tf_msg
from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.body_tf import BodyTfWriter
from mcap_parser.writers.head_trajectory import HeadTrajectoryWriter


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
