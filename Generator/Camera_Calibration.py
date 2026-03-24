import cv2
import numpy as np
import open3d as o3d
from CameraFeed import Feed
import threading
import socket
import glob
import time
import json
# ----------------------
# -------
# Example matched image points
# -----------------------------

# -----------------------------
# Example camera matrices
# (normally from calibration)
# -----------------------------

class Camera_Calib():
    def __init__(self,load=True):   
        if not load:
            self.feed=Feed()
            threading.Thread(target=self.feed.receive_camera, args=(4000,"Cam1"), daemon=True).start()
            threading.Thread(target=self.feed.receive_camera, args=(4001,"Cam2"), daemon=True).start()

            self.s_img = socket.socket()
            self.s_img.connect(("127.0.0.1", 5005))
            
            self.s_obj = socket.socket()
            self.s_obj.connect(("127.0.0.1", 5006))

    def move_obj(self, obj_no, posn):
        msg = f"move {float(posn[0])} {float(posn[1])} {float(posn[2])} {float(posn[3])} {float(posn[4])} {float(posn[5])} {float(posn[6])}\n"
        if obj_no==1:
            self.s_img.sendall(msg.encode()) # move right
            time.sleep(1)
        elif obj_no==2:
            self.s_obj.sendall(msg.encode())
            time.sleep(0.1) 
    
    def save_frames(self,camno):
        self.move_obj(2,[0,0,-1,0,0,0,1])
        arr=[[0,0,0.3,0,0,0,1],[0,0,0.4,15,0,0,1],[0,0,0.4,-15,0,0,1],[0,0,0.4,30,0,0,1],[0,0,0.4,-30,0,0,1],[0,0,0.4,0,15,0,1],[0,0,0.4,0,-15,0,1],[0,0,0.4,0,30,0,1],[0,0,0.4,0,-30,0,1],
             [0.1,0,0.3,0,0,0,1],[0.1,0,0.35,15,0,0,1],[0.1,0,0.35,-15,0,0,1],[0.1,0,0.35,30,0,0,1],[0.1,0,0.35,-30,0,0,1],[0.1,0,0.35,0,15,0,1],[0.1,0,0.35,0,-15,0,1],[0.1,0,0.35,0,30,0,1],[0.1,0,0.35,0,-30,0,1],1]
        arr2=[[1,0,0.3,0,0,0,1],[1,0,0.4,15,0,0,1],[1,0,0.4,-15,0,0,1],[1,0,0.4,30,0,0,1],[1,0,0.4,-30,0,0,1],[1,0,0.4,0,15,0,1],[1,0,0.4,0,-15,0,1],[1,0,0.4,0,30,0,1],[1,0,0.4,0,-30,0,1],
             [1.1,0,0.3,0,0,0,1],[1.1,0,0.35,15,0,0,1],[1.1,0,0.35,-15,0,0,1],[1.1,0,0.35,30,0,0,1],[1.1,0,0.35,-30,0,0,1],[1.1,0,0.35,0,15,0,1],[1.1,0,0.35,0,-15,0,1],[1.1,0,0.35,0,30,0,1],[1.1,0,0.35,0,-30,0,1]]
        
        i=0
        if camno==2: 
            arr=arr2
        for a in arr:
            self.move_obj(1,a)
            if self.feed.frames[f"Cam{camno}"] is not None:
                cv2.imshow(f"Cam{camno}", self.feed.frames[f"Cam{camno}"])
                cv2.imwrite(f"Calib_img{camno}/frame{i}.jpg", self.feed.frames[f"Cam{camno}"])
                print("written")
                i+=1
    def disp(self):        
        while True:
            if self.feed.frames["Cam1"] is not None:
                cv2.imshow("Cam1", self.feed.frames["Cam1"])
            if self.feed.frames["Cam2"] is not None:
                cv2.imshow("Cam2", self.feed.frames["Cam2"])

            if cv2.waitKey(1) == 27:
                break

    def calibInteranal(self,camno):        
        # Checkerboard dimensions (inner corners)
        CHECKERBOARD = (7, 7)
        # Termination criteria for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare 3D object points (0,0,0), (1,0,0), ..., (6,5,0)
        objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane

        # Load calibration images
        images = glob.glob(f'calib_img{camno}/*.jpg')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            print(ret)
            if ret:
                objpoints.append(objp)

                # Refine corner locations
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)

                # Draw corners (optional)
                cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
                cv2.imshow('Corners', img)
                cv2.waitKey(200)
        # cv2.destroyAllWindows()
        cv2.destroyWindow('Corners')
        print("wtf")

        # Perform calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        # Print results
        print("\nCamera matrix (Intrinsic parameters):")
        print(camera_matrix)

        print("\nDistortion coefficients:")
        print(dist_coeffs)

        print("\nReprojection error:", ret)

        return camera_matrix,dist_coeffs
    
    
    def load_intrinsics(self, camno):
        filename = f"camera_intrinsics{camno}.json"
        
        with open(filename, "r") as f:
            data = json.load(f)

        # Convert lists back to numpy arrays
        camera_matrix = np.array(data["camera_matrix"])
        dist_coeffs = np.array(data["distortion_coefficients"])

        print(f"Loaded intrinsics from {filename}")
        
        return camera_matrix, dist_coeffs
    
    def save_intrinsics(self,camera_matrix,dist_coeffs,camno):
        data = {
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.tolist()
        }

        # Save to file
        with open(f"camera_intrinsics{camno}.json", "w") as f:
            json.dump(data, f, indent=4)

        print("Saved to camera_intrinsics.json")

    def calib(self):
        self.save_frames(1)
        camera_matrix,dist_coeffs=self.calibInteranal(1)
        self.save_intrinsics(camera_matrix,dist_coeffs,1)
        self.save_frames(2)
        camera_matrix,dist_coeffs=self.calibInteranal(2)
        self.save_intrinsics(camera_matrix,dist_coeffs,2)

    
    def get_vals(self):
        K1,_ = self.load_intrinsics(camno=1)
        K2,_ = self.load_intrinsics(camno=2)

        self.P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))

        # Camera 2 projection matrix (shifted along X)
        R = np.eye(3)
        t = np.array([[1], [0], [0]])

        self.P2 = K2 @ np.hstack((R, t))
        return self.P1, self.P2

# cc= Camera_Calib()
# # cc.calib()

# t1 = threading.Thread(target=cc.disp)
# t2 = threading.Thread(target=cc.calib)

# t1.start()
# t2.start()

# t1.join()
# t2.join()

# print("donnne")
