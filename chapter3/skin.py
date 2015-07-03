import numpy as np
import cv2

camera = cv2.VideoCapture(0)

lower = np.array([0, 100, 0], dtype="uint8")
upper = np.array([50,255,255], dtype="uint8")

def click_handler(event, x, y, flags, params):
  global frame, lower, upper
  if event == cv2.EVENT_LBUTTONDOWN:
    point = frame[x, y]
    print point

cv2.namedWindow('HSV')
cv2.setMouseCallback('HSV', click_handler)


while (True):
  _, frame = camera.read()
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  skinMask = cv2.inRange(hsv, lower, upper)
  skinMask = cv2.GaussianBlur(skinMask, (9, 9), 0)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
  skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel, iterations = 3)
  skinMask = cv2.GaussianBlur(skinMask, (9, 9), 0)
  skin = cv2.bitwise_and(frame, frame, mask = skinMask)
  cv2.imshow("HSV", skin)
  key = cv2.waitKey(1000 / 12) & 0xff
  if key == ord("q"):
    break
  if key == ord("p"):
    cv2.imwrite("skin.jpg", skin) 

cv2.destroyAllWindows()
