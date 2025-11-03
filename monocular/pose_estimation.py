import cv2
import numpy as np


class ConeEstimator:
    keypoints_3d = np.array(
        [
            [-0.1, 0, 0],
            [-0.075, 0, 0.1],
            [-0.05, 0, 0.2],
            [-0.025, 0, 0.3],
            [0.025, 0, 0.3],
            [0.05, 0, 0.2],
            [0.075, 0, 0.1],
            [0.1, 0, 0],
        ]
    )

    def __init__(self, K, D):
        self.K = K
        self.D = D

    def estimate_pose(self, keypoints_2d):
        print(keypoints_2d)
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            ConeEstimator.keypoints_3d,
            keypoints_2d,
            self.K,
            self.D,
        )

        return success, rvec, tvec
