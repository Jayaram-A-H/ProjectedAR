import cv2
import numpy as np
import Camera_Calibration as cc
import open3d as o3d
import numpy as np
#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.
import mediapipe as mp
import Mediapipe_landmarks as draw_mesh
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import matplotlib.pyplot as plt
import time


class ProjectedAR:

    def __init__(self):
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(frame)
        self.vis.get_render_option().point_size = 8

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

    def run(self):
        # STEP 2: Create an FaceLandmarker object.
        detector= self.get_detector()

        # STEP 3: Load the input image.

        ccc=cc.Camera_Calib()
        P1,P2=ccc.get_vals()


        cap1 = cv2.VideoCapture("../Assets/cam1_face.mp4")
        cap2 = cv2.VideoCapture("../Assets/cam2_face.mp4")
        while(1):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break


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
                continue
            for i in detection_result.face_landmarks[0]:
                coord1.append([i.x *image.width,i.y *image.height])
            for i in detection_result2.face_landmarks[0]:
                coord2.append([i.x *image2.width,i.y *image2.height])

            annotated_image = draw_mesh.draw_landmarks_on_image(image.numpy_view(), detection_result)
            annotated_image2 = draw_mesh.draw_landmarks_on_image(image2.numpy_view(), detection_result2)
            cv2.imshow("fraem",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            cv2.imshow("fraem2",cv2.cvtColor(annotated_image2, cv2.COLOR_RGB2BGR))
            pcd=self.Triangulate(coord1=coord1,coord2=coord2,pcd=self.pcd,P1=P1,P2=P2)
            self.vis.update_geometry(pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

            time.sleep(0.03)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap1.release()
        cap2.release()
        detector.close()
        # cap.release()
        cv2.destroyAllWindows()

a=ProjectedAR()
a.run()