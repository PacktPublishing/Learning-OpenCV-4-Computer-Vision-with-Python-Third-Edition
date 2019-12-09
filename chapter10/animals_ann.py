import cv2
import numpy as np
from random import randint

animals_net = cv2.ml.ANN_MLP_create()
animals_net.setTrainMethod(
    cv2.ml.ANN_MLP_RPROP | cv2.ml.ANN_MLP_UPDATE_WEIGHTS)
animals_net.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
animals_net.setLayerSizes(np.array([3, 6, 4]))
animals_net.setTermCriteria(
    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))

"""Input arrays
weight, length, teeth
"""

"""Output arrays
dog, condor, dolphin, dragon
"""

def dog_sample():
  return [randint(10, 20), 1, randint(38, 42)]

def dog_class():
  return [1, 0, 0, 0]

def condor_sample():
  return [randint(3, 10), randint(3, 5), 0]

def condor_class():
  return [0, 1, 0, 0]

def dolphin_sample():
  return [randint(30, 190), randint(5, 15), randint(80, 100)]

def dolphin_class():
  return [0, 0, 1, 0]

def dragon_sample():
  return [randint(1200, 1800), randint(30, 40), randint(160, 180)]

def dragon_class():
  return [0, 0, 0, 1]

def record(sample, classification):
  return (np.array([sample], np.float32),
          np.array([classification], np.float32))


RECORDS = 5000
records = []
for x in range(0, RECORDS):
  records.append(record(dog_sample(), dog_class()))
  records.append(record(condor_sample(), condor_class()))
  records.append(record(dolphin_sample(), dolphin_class()))
  records.append(record(dragon_sample(), dragon_class()))

EPOCHS = 2
for e in range(0, EPOCHS):
  print("epoch: %d" % e)
  for t, c in records:
    animals_net.train(t, cv2.ml.ROW_SAMPLE, c)


TESTS = 100

dog_results = 0
for x in range(0, TESTS):
  clas = int(animals_net.predict(
      np.array([dog_sample()], np.float32))[0])
  print("class: %d" % clas)
  if clas == 0:
      dog_results += 1

condor_results = 0
for x in range(0, TESTS):
  clas = int(animals_net.predict(
      np.array([condor_sample()], np.float32))[0])
  print("class: %d" % clas)
  if clas == 1:
      condor_results += 1

dolphin_results = 0
for x in range(0, TESTS):
  clas = int(animals_net.predict(
      np.array([dolphin_sample()], np.float32))[0])
  print("class: %d" % clas)
  if clas == 2:
      dolphin_results += 1

dragon_results = 0
for x in range(0, TESTS):
  clas = int(animals_net.predict(
      np.array([dragon_sample()], np.float32))[0])
  print("class: %d" % clas)
  if clas == 3:
      dragon_results += 1

print("Dog accuracy: %.2f%%" % (100.0 * dog_results / TESTS))
print("condor accuracy: %.2f%%" % (100.0 * condor_results / TESTS))
print("dolphin accuracy: %.2f%%" % (100.0 * dolphin_results / TESTS))
print("dragon accuracy: %.2f%%" % (100.0 * dragon_results / TESTS))
