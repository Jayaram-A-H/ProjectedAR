import cv2
import numpy as np
import Camera_Calibration as cc
import open3d as o3d
from CameraFeed import Feed
import numpy as np
#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.
import mediapipe as mp
import socket
import Mediapipe_landmarks as draw_mesh
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import matplotlib.pyplot as plt
import threading
import time


print("ohhhh")
class ProjectedAR:

    def __init__(self):
                
        self.feed=Feed()

        threading.Thread(target=self.feed.receive_camera, args=(4000,"Cam1"), daemon=True).start()
        threading.Thread(target=self.feed.receive_camera, args=(4001,"Cam2"), daemon=True).start()
        self.t=True
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(frame)
        self.vis.get_render_option().point_size = 8
        self.s_img = socket.socket()
        self.s_img.connect(("127.0.0.1", 5005))
        
        self.s_obj = socket.socket()
        self.s_obj.connect(("127.0.0.1", 5006))

    def Triangulate(self,coord1,coord2,pcd,P1,P2):
        
        # Convert to numpy arrays
        pts1 = np.array(coord1, dtype=np.float32).T
        pts2 = np.array(coord2, dtype=np.float32).T

        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
        points_3d = (points_4d[:3] / points_4d[3]).T
        points_3d = points_3d[np.isfinite(points_3d).all(axis=1)]
        self.pcd.points = o3d.utility.Vector3dVector(points_3d)
        return self.pcd
    
    def get_detector(self):
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)
        return detector


    def rotate_obj(self):
        ang=180
        while(self.t==True):
            msg= f"move 0.5 -2.5 1.1 0 {ang} 0 1.5"
            self.s_obj.sendall(msg.encode())
            ang+=4
            time.sleep(0.1) 
    
    def init_posn(self):
        # msg = f"move {float(posn[0])} {float(posn[1])} {float(posn[2])} {float(posn[3])} {float(posn[4])} {float(posn[5])} {float(posn[6])}\n"
        msg ="move 0 0 1 0 0 0 0.2"
        self.s_img.sendall(msg.encode()) # move right
        time.sleep(1)
        msg= "move 0.5 -2.5 1.1 0 -180 0 1.5"
        self.s_obj.sendall(msg.encode())
        time.sleep(0.1) 


    def run(self):
        # STEP 2: Create an FaceLandmarker object.
        self.init_posn()
        detector= self.get_detector()
        print(self.feed)

        # STEP 3: Load the input image.

        ccc=cc.Camera_Calib(load=True)
        P1,P2=ccc.get_vals()


        # cap1 = cv2.VideoCapture("../Assets/cam1_face.mp4")
        # cap2 = cv2.VideoCapture("../Assets/cam2_face.mp4")
        while(1):
            frame1,frame2=self.feed.frames["Cam1"],self.feed.frames["Cam2"]

            if  frame1 is None or frame2 is None:
                print("no frame")
                continue


            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame1_rgb
            )

            image2 = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame2_rgb
            )
            # image = mp.Image.create_from_file(f"cam1_frames/frame_{j:05}.png")
            # image2 = mp.Image.create_from_file(f"cam2_frames/frame_{j:05}.png")
            detection_result = detector.detect(image)
            detection_result2 = detector.detect(image2)
            # print(detection_result.face_landmarks[0][1])
            coord1=[]
            coord2=[]
            if(detection_result.face_landmarks==[] or detection_result2.face_landmarks==[]):
                print("no face")
                self.vis.update_geometry(self.pcd)
                self.vis.poll_events()
                self.vis.update_renderer()
                continue
            for i in detection_result.face_landmarks[0]:
                coord1.append([i.x *image.width,i.y *image.height])
            for i in detection_result2.face_landmarks[0]:
                coord2.append([i.x *image2.width,i.y *image2.height])

            annotated_image = draw_mesh.draw_landmarks_on_image(image.numpy_view(), detection_result)
            annotated_image2 = draw_mesh.draw_landmarks_on_image(image2.numpy_view(), detection_result2)
            cv2.imshow("fraem",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            cv2.imshow("fraem2",cv2.cvtColor(annotated_image2, cv2.COLOR_RGB2BGR))
            self.pcd=self.Triangulate(coord1=coord1,coord2=coord2,pcd=self.pcd,P1=P1,P2=P2)
            self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

            time.sleep(0.03)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # cap1.release()
        # cap2.release()
        detector.close()
        # cap.release()
        cv2.destroyAllWindows()

    def disp(self):        
        while True:
            print("disp")
            if self.feed.frames["Cam1"] is not None:
                print("halalla")
                cv2.imshow("Cam1", self.feed.frames["Cam1"])
            if self.feed.frames["Cam2"] is not None:
                cv2.imshow("Cam2", self.feed.frames["Cam2"])

            if cv2.waitKey(1) == 27:
                break

a=ProjectedAR()
# a.run()


t2 = threading.Thread(target=a.rotate_obj, daemon=True)
t2.start()

# t1 = threading.Thread(target=a.run)
# t2 = threading.Thread(target=a.rotate_obj)
a.run()
# t1.start()
# t2.start()

# t1.join()
# t2.join()
