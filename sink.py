import numpy as np
import cv2

img = cv2.imread('/home/d3athmast3r/Documents/front.png')

roi = img[200:350, 320:450]
img[50:200, 320:450] = roi
while True:
  
  cv2.imshow("test" , img)
  if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
      break
cv2.destroyAllWindows()