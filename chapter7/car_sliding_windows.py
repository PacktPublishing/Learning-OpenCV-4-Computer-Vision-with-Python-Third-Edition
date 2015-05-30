import traceback
import cv2
import numpy as np
from car_detector.detector import car_detector, bow_features
from car_detector.pyramid import pyramid
from car_detector.non_maximum import non_max_suppression_fast as nms
from car_detector.sliding_window import sliding_window

def in_range(number, test, thresh=0.2):
  return abs(number - test) < thresh

test_image = "/home/d3athmast3r/dev/python/study/images/cars.jpg"
svm, extractor = car_detector()
detect = cv2.xfeatures2d.SIFT_create()

w, h = 100, 40
img = cv2.imread(test_image)
gray = cv2.imread(test_image, 0)

width, height, channels = img.shape

rectangles = []
counter = 1
scaleFactor = 1.25
scale = 1
cumul = 1

for resized in pyramid(img, scaleFactor):
  
  scale = float(img.shape[1]) / float(resized.shape[1])
  for (x, y, roi) in sliding_window(resized, 20, (100, 40)):
    
    if roi.shape[1] != w or roi.shape[0] != h:
      continue

    try:
      print x, y, roi.shape  
      bf = bow_features(roi, extractor, detect)
      _, result = svm.predict(bf)
      a, res = svm.predict(bf, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
      print "RESULT: %d" % result[0][0]
      if result[0][0] == 1:
        if in_range(-1, res[0][0], 0.5):
          cv2.rectangle(img, (int(x * scale) , int(y * scale)), (int((x+w) * scale), int((y + h) * scale)), (0, 255, 0), 1)
          cv2.imshow("found", img)
          cv2.waitKey(0)
          rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x+w) * scale), int((y+h) * scale)
          rectangles.append((rx, ry, rx2, ry2, (0, (40 * counter), int(255/counter)), counter, res[0][0]))
    except:
      print x, y, roi.shape

    counter += 1 

for (x, y, x2, y2, color, thick,score) in rectangles:
  print x, y, x2, y2, score

cv2.imshow("img", img)
cv2.waitKey(0)
