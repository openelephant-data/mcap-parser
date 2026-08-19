import numpy as np

from mcap_parser._testing import hand_kp_3d_msg, pointcloud_msg, tf_msg
from mcap_parser.writers.base import ParseContext
from mcap_parser.writers.body_tf import BodyTfWriter
from mcap_parser.writers.hand_keypoints import HandKeypointsWriter
from mcap_parser.writers.pointcloud import PointcloudWriter


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
    assert pcds[0].read_bytes().startswith(b"# .PCD v0.7")

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
