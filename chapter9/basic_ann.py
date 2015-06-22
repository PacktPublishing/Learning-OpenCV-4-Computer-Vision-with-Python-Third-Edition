import cv2
import numpy as np


ann = cv2.ml.ANN_MLP_create()

ann.setLayerSizes(np.array([64, 18, 2], dtype=np.float32))

num4 = [
  1, 0, 0, 0, 0, 0, 0, 1, 
  1, 0, 0, 0, 0, 0, 0, 1, 
  1, 0, 0, 0, 0, 0, 0, 1,
  1, 0, 0, 0, 0, 0, 0, 1,
  1, 1, 1, 1, 1, 1, 1, 1,
  0, 0, 0, 0, 0, 0, 0, 1,
  0, 0, 0, 0, 0, 0, 0, 1,
  0, 0, 0, 0, 0, 0, 0, 1
]

num1 = [
  0, 0, 0, 0, 1, 0, 0, 0, 
  0, 0, 0, 1, 1, 0, 0, 0, 
  0, 0, 1, 1, 1, 0, 0, 0, 
  0, 0, 0, 0, 1, 0, 0, 0, 
  0, 0, 0, 0, 1, 0, 0, 0, 
  0, 0, 0, 0, 1, 0, 0, 0, 
  0, 0, 0, 0, 1, 0, 0, 0, 
  0, 0, 0, 0, 1, 0, 0, 0
]


train_data = [
  
  (num1, [1, 1]),
  (num4, [4, 4])
]

ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)

for x in range(0, 100):
  for t, r in train_data:
    ann.train(np.array([t], dtype=np.float32), 
      cv2.ml.ROW_SAMPLE,
      np.array([r], dtype=np.float32)
    )

print ann.predict(np.array([num1], dtype=np.float32))
print ann.predict(np.array([num4], dtype=np.float32))
