import cv2
import numpy as np
import mediapipe as mp
import Camera_Calibration as cc


import open3d as o3d
import numpy as np

#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import matplotlib.pyplot as plt
import time

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.


    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()


# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 3: Load the input image.

'''

cap = cv2.VideoCapture("../cam1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    

    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )  # STEP 4: Detect face landmarks from the input image.'''


ccc=cc.Camera_Calib()
P1,P2=ccc.P1,ccc.P2

vis = o3d.visualization.Visualizer()
vis.create_window()

pcd = o3d.geometry.PointCloud()

# IMPORTANT: add geometry once
vis.add_geometry(pcd)

# make points visible
vis.get_render_option().point_size = 10

# optional coordinate frame
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
vis.add_geometry(frame)

vis.get_render_option().point_size = 8


for j in range(1,1080):

    image = mp.Image.create_from_file(f"cam1_frames/frame_{j:05}.png")
    image2 = mp.Image.create_from_file(f"cam2_frames/frame_{j:05}.png")
    detection_result = detector.detect(image)
    detection_result2 = detector.detect(image2)
    # print(detection_result.face_landmarks[0][1])
    coord1=[]
    coord2=[]
    if(detection_result.face_landmarks==[] or detection_result2.face_landmarks==[]):
        continue
    for i in detection_result.face_landmarks[0]:
        coord1.append([i.x *image.width,i.y *image.height])
    for i in detection_result2.face_landmarks[0]:
        coord2.append([i.x *image.width,i.y *image.height])

    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    annotated_image2 = draw_landmarks_on_image(image2.numpy_view(), detection_result2)
    cv2.imshow("fraem",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("fraem2",cv2.cvtColor(annotated_image2, cv2.COLOR_RGB2BGR))
    # Convert to numpy arrays
    pts1 = np.array(coord1, dtype=np.float32).T
    pts2 = np.array(coord2, dtype=np.float32).T

    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

    # Convert homogeneous → 3D
    points_3d = (points_4d[:3] / points_4d[3]).T
    # print(points_3d)
    # points_3d = [[0.1,0.1,0.1],[0.2,0.2,0.2],[0.3,0.3,0.3],[0,0,0]]
    # points_3d = np.array(points_3d)
    # Optional: remove invalid points
    points_3d = points_3d[np.isfinite(points_3d).all(axis=1)]

    print(pcd)
    # Update point cloud
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    print(pcd.points)
    # Color points red
    # colors = np.tile(np.array([[1,0,0]]), (len(points_3d),1))
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # Update Open3D windo
    # for i in range(1000):
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    time.sleep(0.03)

    # # ---- Triangulation ----
    # points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

    # # Convert homogeneous → 3D

    # points_3d = points_4d[:3] / points_4d[3]

    # points = points_3d.T

    # # pcd = o3d.geometry.PointCloud()
    # # pcd.points = o3d.utility.Vector3dVector(points)

    # #o3d.visualization.draw_geometries([pcd])

    # print("3D Points:\n", points_3d.T[:5])

    # cv2.imshow("cam1", frame1)
    # cv2.imshow("cam2", frame2)
        

    cv2.waitKey(1)

detector.close()
# cap.release()
cv2.destroyAllWindows()

