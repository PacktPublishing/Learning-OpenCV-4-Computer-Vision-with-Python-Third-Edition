import cv2
import numpy as np
import os

from car_detector.non_maximum import non_max_suppression_fast as nms

if not os.path.isdir('CarData'):
    print('CarData folder not found. Please download and unzip '
          'http://l2r.cs.uiuc.edu/~cogcomp/Data/Car/CarData.tar.gz '
          'into the same folder as this script.')
    exit(1)

BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 10
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 100

SVM_SCORE_THRESHOLD = 1.0
NMS_OVERLAP_THRESHOLD = 0.25

sift = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = {}
flann = cv2.FlannBasedMatcher(index_params, search_params)

bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
bow_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)

def get_pos_and_neg_paths(i):
    pos_path = 'CarData/TrainImages/pos-%d.pgm' % (i+1)
    neg_path = 'CarData/TrainImages/neg-%d.pgm' % (i+1)
    return pos_path, neg_path

def add_sample(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    bow_kmeans_trainer.add(descriptors)

for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_paths(i)
    add_sample(pos_path)
    add_sample(neg_path)

voc = bow_kmeans_trainer.cluster()
bow_extractor.setVocabulary(voc)

def extract_bow_descriptors(img):
    features = sift.detect(img)
    return bow_extractor.compute(img, features)

training_data = []
training_labels = []
for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_paths(i)
    pos_img = cv2.imread(pos_path, cv2.IMREAD_GRAYSCALE)
    pos_descriptors = extract_bow_descriptors(pos_img)
    training_data.extend(pos_descriptors)
    training_labels.append(1)
    neg_img = cv2.imread(neg_path, cv2.IMREAD_GRAYSCALE)
    neg_descriptors = extract_bow_descriptors(neg_img)
    training_data.extend(neg_descriptors)
    training_labels.append(-1)

svm = cv2.ml.SVM_create()
svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
          np.array(training_labels))

# TODO

def pyramid(img, scale_factor=1.25, min_size=(200, 80)):
    h, w = image.shape
    min_w, min_h = min_size
    while w >= min_w and h >= min_h:
        yield image
        w //= scale_factor
        h //= scale_factor
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)

def sliding_window(img, step=20, window_size=(100, 40)):
    img_h, img_w = img.shape
    window_w, window_h = window_size
    for y in range(0, img_w - window_w - (img_w % window_w) + 1, step):
        for x in range(0, img_h - window_h - (img_h % window_h) + 1, step):
            yield (x, y, img[y:y+window_h, x:x+window_w])

for test_img_path in ['CarData/TestImages/test-0.pgm',
                      'CarData/TestImages/test-1.pgm',
                      '../images/car.jpg',
                      '../images/haying.jpg',
                      '../images/statue.jpg',
                      '../images/woodcutters.jpg']:
    img = cv2.imread(test_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pos_rects = []
    for resized in pyramid(gray_img):
        for x, y, roi in sliding_window(resized):
            descriptors = extract_bow_descriptors(roi)
            prediction = svm.predict(descriptors)
            if prediction[1][0][0] == 1.0:
                raw_prediction = svm.predict(
                    descriptors, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
                score = -raw_prediction[1][0][0]
                if score > SVM_SCORE_THRESHOLD:
                    h, w = roi.shape
                    scale = gray_img.shape[0] / float(resized.shape[0])
                    pos_rects.append([int(x * scale),
                                      int(y * scale),
                                      int((x+w) * scale),
                                      int((y+h) * scale),
                                      score])
    nms(np.array(pos_rects), NMS_OVERLAP_THRESHOLD)
    for x0, y0, x1, y1, score in pos_rects:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)),
                      (0, 255, 0), 1)
        text = '%.2f' % score
        cv2.putText(img, text, (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow(test_img_path, img)
cv2.waitKey(0)
