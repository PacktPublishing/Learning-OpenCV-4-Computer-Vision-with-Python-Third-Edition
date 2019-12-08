import numpy as np
import cv2


left_camera = cv2.VideoCapture(0)
right_camera = cv2.VideoCapture(1)

# Create windows.
cv2.namedWindow('Left Camera', cv2.WINDOW_NORMAL)
cv2.namedWindow('Right Camera', cv2.WINDOW_NORMAL)
cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

blockSize = 40

while(cv2.waitKey(1) == -1):

    ret1, left_frame = left_camera.read()
    ret2, right_frame = right_camera.read()

    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Left Camera', left_gray)
    cv2.imshow('Right Camera', right_gray)
    stereo = cv2.StereoSGBM_create(
        minDisparity = 1,
        numDisparities = 16,
        blockSize = blockSize,
        speckleWindowSize = 10,
        speckleRange = 32,
        disp12MaxDiff = 1,
        P1 = 8*3*blockSize**2,
        P2 = 32*3*blockSize**2)
    disparity = stereo.compute(left_gray, right_gray)
    disparity = cv2.normalize(
            disparity, disparity, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('Disparity', disparity)
