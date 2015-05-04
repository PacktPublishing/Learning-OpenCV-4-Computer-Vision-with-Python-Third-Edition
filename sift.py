import cv2
import sys
import numpy as np

imgpath = sys.argv[1]
img = cv2.imread(imgpath)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detectAndCompute(gray,None)

img = cv2.drawKeypoints(image=gray, outImage=img, keypoints = kp, flags = 4, color = (51, 103, 236))
cv2.imwrite('sift_keypoints.jpg', img)

cv2.imshow('sift_keypoints.jpg', cv2.imread('sift_keypoints.jpg'))
while (True):
  if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
    break
cv2.destroyAllWindows()
