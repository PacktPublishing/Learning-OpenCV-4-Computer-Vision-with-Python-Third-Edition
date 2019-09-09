import cv2
import numpy as np
import os

if not os.path.isdir('CarData'):
    print('CarData folder not found. Please download and unzip '
          'http://l2r.cs.uiuc.edu/~cogcomp/Data/Car/CarData.tar.gz '
          'into the same folder as this script.')
    exit(1)

BOW_VOCABULARY_NUM_TRAINING_SAMPLES_PER_CLASS = 8
BOW_DESCRIPTORS_NUM_TRAINING_SAMPLES_PER_CLASS = 20

sift = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = {}
flann = cv2.FlannBasedMatcher(index_params, search_params)

bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
bow_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)

def get_pos_and_neg_paths(i):
    pos_path = 'CarData/TrainImages/pos-%d.pgm' % i+1
    neg_path = 'CarData/TrainImages/neg-%d.pgm' % i+1
    return pos_path, neg_path

def add_sample(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    bow_kmeans_trainer.add(descriptors)

for i in range(BOW_VOCABULARY_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_paths(i)
    add_sample(pos_path)
    add_sample(neg_path)
  
voc = bow_kmeans_trainer.cluster()
bow_extractor.setVocabulary(voc)

def extract_bow_descriptors(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    features = sift.detect(img)
    return bow_extractor.compute(img, features)

training_data = []
training_labels = []
for i in range(BOW_DESCRIPTORS_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_paths(i)
    pos_descriptors = extract_bow_descriptors(pos_path)
    training_data.extend(pos_descriptors)
    training_labels.append(1)
    neg_descriptors = extract_bow_descriptors(neg_path)
    training_data.extend(neg_descriptors)
    training_labels.append(-1)

svm = cv2.ml.SVM_create()
svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
          np.array(training_labels))

for test_img_path in ['../images/car.jpg', '../images/bb.jpg']:
    img = cv2.imread(test_img_path)
    descriptors = extract_bow_descriptors(path)
    prediction = svm.predict(descriptors)
    if predictiction[1][0][0] == 1.0:
        text = 'car'
        color = (0, 255, 0)
    else:
        text = 'not car'
        color = (0, 0, 255)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color, 2, cv2.LINE_AA)
    cv2.imshow(test_img_path, img)
cv2.waitKey(0)
