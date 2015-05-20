import cv2
import numpy as np
from os.path import join

basepath = "/home/d3athmast3r/dev/python/study/images/"

imagePaths = ["bb.jpg",
"color1.jpg",
"basil.jpg",
"bathory_album.jpg"]

hog = cv2.HOGDescriptor()
svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR,
                    svm_type = cv2.ml.SVM_C_SVC,
                    C=2.67, gamma=5.383 )
responses = np.float32(np.repeat(np.arange(10),250)[:,np.newaxis])

images = []


features = []


for im in imagePaths:
  img = cv2.imread(join(basepath, im), 0)
  d = np.array(img, dtype = np.float32)
  q = d.flatten()
  images.append(img)

svm = cv2.ml.SVM_create()
retval, results = svm.train(np.asarray(images), cv2.ml.ROW_SAMPLE, np.array([1, 1, -1, -1]))
hog.setSVMDetector(results)

result = hog.detectMultiScale(cv2.imread("/home/d3athmast3r/dev/python/study/images/depth1.jpg"))

print result
