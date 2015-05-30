import cv2
import numpy as np

camera = cv2.VideoCapture(0)

es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,4))
kernel = np.ones((5,5),np.uint8)
firstFrame = None

while (True):
  ret, frame = camera.read()
  print "reading frame"
  if firstFrame is None:
    background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    firstFrame = cv2.GaussianBlur(background, (21, 21), 0)
    print "background set"
    continue
  
  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
  #diff = cv2.absdiff(gray_frame, gray_bg)
  diff = cv2.absdiff(background, gray_frame)
  diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
  diff = cv2.dilate(diff, None, iterations = 2)
  image, cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  print "%d contours detected" % len(cnts)
  for c in cnts:
    if cv2.contourArea(c) < 500:
      continue
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

  
  cv2.imshow("contours", frame)
  cv2.imshow("dif", diff)
  if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
      break

cv2.destroyAllWindows()
camera.release()
