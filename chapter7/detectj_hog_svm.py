import cv2
import numpy as np
from os.path import join


basepath = "/home/d3athmast3r/dev/python/CarData/TrainImages/"

extract = cv2.xfeatures2d.SIFT_create()
detect = cv2.xfeatures2d.SIFT_create()
flann_params = dict(algorithm = 1, trees = 5)      # flann enums are missing, FLANN_INDEX_KDTREE=1
matcher = cv2.FlannBasedMatcher(flann_params, {})
hog = cv2.HOGDescriptor()
svm = cv2.ml.SVM_create()
## 1.a setup BOW
bow_train   = cv2.BOWKMeansTrainer(8) # toy world, you want more.
bow_extract = cv2.BOWImgDescriptorExtractor( extract, matcher )

pos = "pos-"
neg = "neg-"
def path(cls,i): # "./left03.jpg"
  return "%s/%s%d.pgm"  % (basepath,cls,i+1)

def feature_bow(fn):
  im = cv2.imread(fn,0)
  return bow_extract.compute(im, detect.detect(im))

def extract_sift(img_path):
  img = cv2.imread(img_path, 0)
  return extract.compute(img, detect.detect(img))[1]

for i in range(20):
  bow_train.add(extract_sift(path(pos, i)))
  bow_train.add(extract_sift(path(neg, i)))

## 1.c kmeans cluster descriptors to vocabulary
voc = bow_train.cluster()
bow_extract.setVocabulary(voc)
print "bow vocab", np.shape(voc)

traindata, trainlabels = [], []
for i in range(20):
  traindata.append(feature_bow(path(pos, i)))
  trainlabels.append(1)
  traindata.append(feature_bow(path(neg, i)))
  trainlabels.append(-1)


print "svm items", len(traindata), len(traindata[0])

svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))
"""
hog.setSVMDetector(svm)

ret, results = hog.detectMultiScale(cv2.imread(join(basepath, "pos-100.jpg")))
print results
"""
