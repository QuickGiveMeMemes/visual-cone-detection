import cv2
import numpy as np


R_DIFF = 0.0206375
HEIGHT = 0.2921 - 0.0444

class ConeEstimator:
    keypoints_3d = np.array(
        [
            [-(0.0889 - 0.00952), 0, 0],
            [-(0.0889 - 0.00952) + R_DIFF, 0, HEIGHT / 3],
            [-(0.0889 - 0.00952) + 2 * R_DIFF, 0, HEIGHT * 2/3],
            [-(0.0889 - 0.00952) + 3 * R_DIFF, 0, HEIGHT],
            [(0.0889 - 0.00952) - 3 * R_DIFF, 0, HEIGHT],
            [(0.0889 - 0.00952) - 2 * R_DIFF, 0, HEIGHT * 2/3],
            [(0.0889 - 0.00952) - 1 * R_DIFF, 0, HEIGHT / 3],
            [(0.0889 - 0.00952), 0, 0],
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
