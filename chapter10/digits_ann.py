import gzip
import pickle

import cv2
import numpy as np

"""OpenCV ANN Handwritten digit recognition example

Wraps OpenCV's own ANN by automating the loading of data and supplying default paramters,
such as 20 hidden layers, 10000 samples and 1 training epoch.

The load data code is adapted from http://neuralnetworksanddeeplearning.com/chap1.html
by Michael Nielsen
"""

def load_data():
    mnist = gzip.open('./digits_data/mnist.pkl.gz', 'rb')
    training_data, test_data = pickle.load(mnist)
    mnist.close()
    return (training_data, test_data)

def wrap_data():
    tr_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def create_ANN(hidden_nodes=20):
    ann = cv2.ml.ANN_MLP_create()
    ann.setLayerSizes(np.array([784, hidden_nodes, 10]))
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1, 0.1)
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0)
    ann.setTermCriteria(
        (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 1.0))
    return ann

def train(ann, samples=10000, epochs=1):

    tr, test = wrap_data()

    # Convert iterator to list so that we can iterate multiple times
    # in multiple epochs.
    tr = list(tr)

    for epoch in range(epochs):
        print("Completed %d/%d epochs" % (epoch, epochs))
        counter = 0
        for img in tr:
            if (counter > samples):
                break
            if (counter % 1000 == 0):
                print("Epoch %d: Trained on %d/%d samples" % \
                      (epoch, counter, samples))
            counter += 1
            sample, response = img
            data = cv2.ml.TrainData_create(
                np.array([sample.ravel()], dtype=np.float32),
                cv2.ml.ROW_SAMPLE,
                np.array([response.ravel()], dtype=np.float32))
            if ann.isTrained():
                ann.train(data, cv2.ml.ANN_MLP_UPDATE_WEIGHTS | cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
            else:
                ann.train(data, cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
    print("Completed all epochs!")

    return ann, test

def test(ann, test_data):
    sample = np.array(
        test_data[0][0].ravel(), dtype=np.float32).reshape(28, 28)
    cv2.imshow("sample", sample)
    cv2.waitKey()
    prediction = ann.predict(
        np.array([test_data[0][0].ravel()], dtype=np.float32))
    print(prediction)

def predict(ann, sample):
    resized = sample.copy()
    rows, cols = resized.shape
    if (rows != 28 or cols != 28) and rows * cols > 0:
        resized = cv2.resize(resized, (28, 28),
                             interpolation=cv2.INTER_LINEAR)
    return ann.predict(np.array([resized.ravel()], dtype=np.float32))

"""usage:
ann, test_data = train(create_ANN())
test(ann, test_data)
"""
