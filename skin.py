import numpy as np
import cv2

camera = cv2.VideoCapture(0)

lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20,255,255], dtype="uint8")

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
 
  # apply a series of erosions and dilations to the mask
  # using an elliptical kernel
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
  skinMask = cv2.erode(skinMask, kernel, iterations = 2)
  skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
 
  # blur the mask to help remove noise, then apply the
  # mask to the frame
  skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
  #skin = cv2.bitwise_and(frame, frame, mask = skinMask)
  skin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  skin = cv2.adaptiveThreshold(skin, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  skin = cv2.GaussianBlur(skin, (3, 3), 0)
  cv2.imshow("HSV", skin)
  

  if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
    break

cv2.destroyAllWindows()
