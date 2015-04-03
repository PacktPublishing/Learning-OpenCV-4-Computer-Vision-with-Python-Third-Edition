import cv2
import numpy as np

camera = cv2.VideoCapture(0)
cv2.namedWindow('dst')
while (True):
  
  ret, frame = camera.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray = np.float32(gray)
  dst = cv2.cornerHarris(gray, 2, 3, 0.04)
  frame[dst>0.01 * dst.max()] = [0, 0, 255]
  cv2.imshow('dst', frame)
  if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
    break

camera.release()
cv2.destroyAllWindows()
