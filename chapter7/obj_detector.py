import cv2
import numpy as np
from os.path import join

basepath = "/home/d3athmast3r/dev/python/study/images/"

imagePaths = ["bb.jpg",
"color1.jpg",
"basil.jpg",
"bathory_album.jpg"]

extract = cv2.xfeatures2d.SIFT_create()
detect = cv2.xfeatures2d.SIFT_create()

hog = cv2.HOGDescriptor()
svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR,
                    svm_type = cv2.ml.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

train_data = np.array([])
features = []


def extract_sift(path):
  img = cv2.imread(path, 0)
  return extract.compute(img, detect.detect(img))[1]

for im in imagePaths:
  im_path = join(basepath, im)
  train_data = np.append(train_data, extract_sift(im_path))
  

svm = cv2.ml.SVM_create()
retval, results = svm.train(train_data.ravel(order="C"),
  np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
)
hog.setSVMDetector(results)

result = hog.detectMultiScale(cv2.imread("/home/d3athmast3r/dev/python/study/images/depth1.jpg"))

print result
