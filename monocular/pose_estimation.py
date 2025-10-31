import cv2
import numpy as np



class ConeEstimator:
    keypoints_3d = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

    def __init__(self, K, D):
        self.K = K
        self.D = D

    def estimate_pose(self, keypoints_2d):
        success, rvec, tvec = cv2.solvePnP(
            ConeEstimator.keypoints_3d,
            keypoints_2d,
            self.K,
            self.D,
            flags=cv2.SOLVEPNP_ITERATIVE,  # TODO Try other methods
        )

        return success, rvec, tvec
