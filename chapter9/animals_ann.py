import cv2
import numpy as np

animals_net = cv2.ml.ANN_MLP_create()
animals_net.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
animals_net.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
animals_net.setLayerSizes(np.array([7, 5, 8]))

"""Input arrays
weight, legs, wings, tail, teeth, fins, breathe fire
"""

"""Output arrays
dog, cat, kangaroo, whale, eagle, rabbit, dolphin and dragon
"""

def dog():
  return [1, 0, 0, 0, 0, 0, 0, 0]

def cat():
  return [0, 1, 0, 0, 0, 0, 0, 0]

def kangaroo():
  return [0, 0, 1, 0, 0, 0, 0, 0]

def baleen():
  return [0, 0, 0, 1, 0, 0, 0, 0]

def eagle():
  return [0, 0, 0, 0, 1, 0, 0, 0]

def rabbit():
  return [0, 0, 0, 0, 0, 1, 0, 0]

def dolphin():
  return [0, 0, 0, 0, 0, 0, 1, 0]

def dragon():
  return [0, 0, 0, 0, 0, 0, 0, 1]

train_data = [
  # dogs
  ([10, 4, 0, 1, 42, 0, 0], dog()),
  ([6, 4, 0, 1, 41, 0, 0], dog()),
  ([12, 4, 0, 1, 39, 0, 0], dog()),
  ([7, 4, 0, 1, 40, 0, 0], dog()),
  # cats
  ([2, 4, 0, 1, 30, 0, 0], cat()),
  ([1.5, 4, 0, 1, 27, 0, 0], cat()),
  ([3, 4, 0, 1, 28, 0, 0], cat()),
  ([4, 4, 0, 1, 29, 0, 0], cat()),
  # kangaroo
  ([70, 2, 0, 1, 30, 0, 0], kangaroo()),
  ([65, 2, 0, 1, 22, 0, 0], kangaroo()),
  ([80, 2, 0, 1, 28, 0, 0], kangaroo()),
  ([65, 2, 0, 1, 29, 0, 0], kangaroo()),
  # baleen whale
  ([70000, 0, 0, 1, 0, 2, 0], baleen()),
  ([120000, 0, 0, 1, 0, 2, 0], baleen()),
  ([100000, 0, 0, 1, 0, 2, 0], baleen()),
  ([95000, 0, 0, 1, 0, 2, 0], baleen()),
  # eagle
  ([6, 2, 2, 1, 0, 0, 0], eagle()),
  ([8, 2, 2, 1, 0, 0, 0], eagle()),
  ([7, 2, 2, 1, 0, 0, 0], eagle()),
  ([8, 2, 2, 1, 0, 0, 0], eagle()),
  ([7, 2, 2, 1, 0, 0, 0], eagle()),
  ([6, 2, 2, 1, 0, 0, 0], eagle()),
  # rabbit
  ([1, 4, 0, 1, 28, 0, 0], rabbit()),
  ([3, 4, 0, 1, 26, 0, 0], rabbit()),
  ([2, 4, 0, 1, 28, 0, 0], rabbit()),
  ([3, 4, 0, 1, 26, 0, 0], rabbit()),
  ([1, 4, 0, 1, 28, 0, 0], rabbit()),
  ([2, 4, 0, 1, 26, 0, 0], rabbit()),
  ([1, 4, 0, 1, 28, 0, 0], rabbit()),
  # dolphin
  ([130, 0, 0, 1, 80, 2, 0], dolphin()),
  ([180, 0, 0, 1, 82, 2, 0], dolphin()),
  ([200, 0, 0, 1, 88, 2, 0], dolphin()),
  ([170, 0, 0, 1, 90, 2, 0], dolphin()),
  ([180, 0, 0, 1, 98, 2, 0], dolphin()),
  ([190, 0, 0, 1, 100, 2, 0], dolphin()),
  # dragon
  ([1000, 4, 2, 1, 250, 0, 1], dragon()),
  ([1200, 4, 2, 1, 290, 0, 1], dragon()),
  ([1100, 4, 2, 1, 230, 0, 1], dragon()),
  ([1200, 4, 2, 1, 250, 0, 1], dragon()),
  ([900, 4, 2, 1, 280, 0, 1], dragon()),
  ([1000, 4, 2, 1, 240, 0, 1], dragon())
]
for x in range(0, 1000):
  print "Epoch %d" % x
  for t, r in train_data:
    animals_net.train(np.array([t], dtype=np.float32), 
      cv2.ml.ROW_SAMPLE,
      np.array([r], dtype=np.float32)
    )

animals = ["dog", "cat", "kangaroo", "whale", "eagle", "rabbit", "dolphin", "dragon"]

pred1 = animals_net.predict(np.array([[1000, 4, 2, 1, 250, 0, 1]], dtype=np.float32))
pred2 = animals_net.predict(np.array([[2, 4, 0, 1, 42, 0, 0]], dtype=np.float32))
pred3 = animals_net.predict(np.array([[120000, 0, 0, 1, 0, 2, 0]], dtype=np.float32))

print animals[int(pred1[0])]
print animals[int(pred2[0])]
print animals[int(pred3[0])]
