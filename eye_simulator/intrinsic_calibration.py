import cv2
import numpy as np
import os


def calibrate_intrinsic_matrix(intrinsic_path, chessboard_square_size_mm=3, chessboard_dims=(5, 5), check=False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((chessboard_dims[0] * chessboard_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_dims[0], 0:chessboard_dims[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    # Open CSV with checked images
    if os.path.isfile("checked_images.csv"):
        with open("checked_images.csv", "r") as f:
            checked_images = f.read().splitlines()
    else:
        checked_images = []

    images = [f for f in os.listdir(intrinsic_path) if
              os.path.isfile(os.path.join(intrinsic_path, f)) and (f.endswith(".jpg") or f.endswith(".png"))]

    for image in images:
        img = cv2.imread(os.path.join(intrinsic_path, image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Invert image to make chessboard detection easier
        gray = cv2.bitwise_not(gray)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_dims, None)

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            if check:
                print("Image: " + image)
                cv2.drawChessboardCorners(img, chessboard_dims, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey()
        else:
            if check:
                print("Image " + image + " not detected")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(mtx, dist)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, gray.shape[::-1], 1, gray.shape[::-1])
    print(newcameramtx, roi)



    if check:
        cv2.destroyAllWindows()
