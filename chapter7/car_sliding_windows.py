import cv2
import numpy as np
from car_detector.detector import car_detector, bow_features
from car_detector.pyramid import pyramid
from car_detector.non_maximum import non_max_suppression_fast as nms

test_image = "/home/d3athmast3r/dev/python/study/images/cars.jpg"
svm, extractor = car_detector()
detect = cv2.xfeatures2d.SIFT_create()

w, h = 100, 40
x = 0
y = 0

hshift, vshift = 20, 20

img = cv2.imread(test_image)
gray = cv2.imread(test_image, 0)

width, height, channels = img.shape

rectangles = []
counter = 0
scaleFactor = 1.25
scale = 1
cumul = 1
roi = None
for resized in pyramid(img, scaleFactor):
  print "resize: %d" % counter
  x, y = 0, 0
  cumul *= pow(scaleFactor, counter)
  
  scale = img.shape[1] / resized.shape[1]

  print "SCALE %d %f %d" % (scale, cumul, counter)
  while (y + h < height):

    roi = resized[y:y+h, x:x+w]
    try:
      bf = bow_features(roi, extractor, detect)
      results = np.array([], dtype=np.float32)
      _, result = svm.predict(bf)
      a, res = svm.predict(bf, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
      if result[0][0] == 1:
        if res[0][0] < -0.8:
          rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x+w) * scale), int((y+h) * scale)
          rectangles.append((rx, ry, rx2, ry2, (0, (40 * counter), int(255/counter)), 1, res[0][0]))
    except:
      pass

    if (x + w > width):
      print "adding..."
      x = 0
      y += vshift
    else:
      x += hshift
  
  counter += 1

for (x, y, x2, y2, color, thick,score) in rectangles:
  if score < -0.9:
    print x, y, x2, y2
    cv2.rectangle(img, (x,y), (x2, y2), color, thick)

cv2.imshow("img", img)
cv2.waitKey(0)
