import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Capture several frames to allow the camera's autoexposure to adjust.
for i in range(10):
    success, frame = cap.read()
if not success:
    exit(1)

# Define an initial tracking window in the center of the frame.
frame_h, frame_w = frame.shape[:2]
w = frame_w//8
h = frame_h//8
x = frame_w//2 - w//2
y = frame_h//2 - h//2
track_window = (x, y, w, h)

# Calculate the normalized HSV histogram of the initial window.
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = None
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Define the termination criteria:
# 10 iterations or convergence within 1-pixel radius.
term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)

# Initialize the Kalman filter.
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]], np.float32) * 0.03
cx = x+w/2
cy = y+h/2
kalman.statePre = np.array(
    [[cx], [cy], [0], [0]], np.float32)
kalman.statePost = np.array(
    [[cx], [cy], [0], [0]], np.float32)


success, frame = cap.read()
while success:

    # Perform back-projection of the HSV histogram onto the frame.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Perform tracking with CamShift.
    rotated_rect, track_window = cv2.CamShift(
        back_proj, track_window, term_crit)

    kalman.predict()
    box_points = cv2.boxPoints(rotated_rect)
    box_points = np.int0(box_points)
    (cx, cy), radius = cv2.minEnclosingCircle(box_points)
    center = np.array([cx, cy], np.float32)
    estimate = kalman.correct(center)
    center_offset = estimate[:,0][:2] - center

    # Draw the tracking window.
    track_window = (track_window[0] + int(center_offset[0]),
                    track_window[1] + int(center_offset[1]),
                    track_window[2],
                    track_window[3])
    cv2.polylines(frame, [box_points], True, (255, 0, 0), 2)

    cv2.imshow('camshift', frame)

    k = cv2.waitKey(1)
    if k == 27:  # Escape
        break

    success, frame = cap.read()


cv2.destroyAllWindows()
cap.release()
