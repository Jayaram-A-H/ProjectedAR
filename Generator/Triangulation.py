import cv2
import numpy as np


import open3d as o3d
import numpy as np


# ---- Camera intrinsic matrices (from calibration) ----
K1 = np.array([[700, 0, 320],
               [0, 700, 240],
               [0,   0,   1]])

K2 = np.array([[700, 0, 320],
               [0, 700, 240],
               [0,   0,   1]])

# ---- Relative pose between cameras (from stereo calibration) ----
R = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]])

T = np.array([[0.1,0,0]]).T  # baseline between cameras

# ---- Projection matrices ----
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
P2 = K2 @ np.hstack((R, T))

# ---- Read camera frames ----
cap1 = cv2.VideoCapture('cam1.mp4')
cap2 = cv2.VideoCapture('cam2.mp4')

orb = cv2.ORB_create()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect features
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match features
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(des1, des2)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)

    good_matches = good_matches[:50]
    match_img = cv2.drawMatches(
    gray1, kp1,
    gray2, kp2,
    good_matches[:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow("Feature Matches", match_img)
    pts1 = []
    pts2 = []

    for m in good_matches[:50]:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.array(pts1).T
    pts2 = np.array(pts2).T
    img1_kp = cv2.drawKeypoints(
    gray1, kp1, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    img2_kp = cv2.drawKeypoints(
        gray2, kp2, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imshow("Frame 1 Keypoints", img1_kp)
    cv2.imshow("Frame 2 Keypoints", img2_kp)

    # ---- Triangulation ----
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

    # Convert homogeneous → 3D
    points_3d = points_4d[:3] / points_4d[3]
    
    points = points_3d.T

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)

    #o3d.visualization.draw_geometries([pcd])

    print("3D Points:\n", points_3d.T[:5])

    # cv2.imshow("cam1", frame1)
    # cv2.imshow("cam2", frame2)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            break
        elif key == ord('q'):
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
            exit()

    # if cv2.waitKey(1) == 27:
    #     break

cap1.release()
cap2.release()
cv2.destroyAllWindows()