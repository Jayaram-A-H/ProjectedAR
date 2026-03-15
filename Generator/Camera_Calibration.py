import cv2
import numpy as np
import open3d as o3d

# -----------------------------
# Example matched image points
# -----------------------------

# -----------------------------
# Example camera matrices
# (normally from calibration)
# -----------------------------

class Camera_Calib():
    def __init__(self):                
        K = np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ])

        # Camera 1 projection matrix
        self.P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))

        # Camera 2 projection matrix (shifted along X)
        R = np.eye(3)
        t = np.array([[0.1], [0], [0]])

        self.P2 = K @ np.hstack((R, t))
    
    def get_vals(self):
        return self.P1, self.P2

