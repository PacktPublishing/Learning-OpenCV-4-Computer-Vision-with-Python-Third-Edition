import cv2
import numpy as np

bs = cv2.createBackgroundSubtractorKNN()

camera = cv2.VideoCapture("/home/d3athmast3r/Downloads/traffic.flv")

while True:
  ret, frame = camera.read()
  fgmask = bs.apply(frame)
  cv2.imshow("mog", fgmask)
  cv2.imshow("diff", frame & cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR))
  k = cv2.waitKey(30) & 0xff
  if k == 27:
      break

camera.release()
cv2.destroyAllWindows()
