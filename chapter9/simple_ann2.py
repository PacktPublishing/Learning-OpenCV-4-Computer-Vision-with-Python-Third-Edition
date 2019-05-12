import pyneural
import gzip
import numpy as np
import cPickle
import cv2

def load_data():
  mnist = gzip.open('./data/mnist.pkl.gz', 'rb')
  training_data, classification_data, test_data = cPickle.load(mnist)
  mnist.close()
  return (training_data, classification_data, test_data)

def wrap_data():
  tr_d, va_d, te_d = load_data()
  training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
  training_results = [vectorized_result(y) for y in tr_d[1]]
  training_data = zip(training_inputs, training_results)
  validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
  validation_data = zip(validation_inputs, va_d[1])
  test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
  test_data = zip(test_inputs, te_d[1])
  return (training_data, validation_data, test_data)

def vectorized_result(j):
  e = np.zeros((10, 1))
  e[j] = 1.0
  return e

features, validation_data, test_data = wrap_data()

f_r = []
d_r = []
counter = 0

for f,d in features:
  if counter > 50000:
    break
  f_r.append(f.ravel())
  d_r.append(d.ravel())
  counter += 1

nn = pyneural.NeuralNet([784, 200, 10])
nn.train(np.array(f_r), np.array(d_r), 5, 1, 0.01, 0.0, 1.0)

def wrap_digit(rect):
  x, y, w, h = rect
  padding = 5
  hcenter = x + w/2
  vcenter = y + h/2
  roi = None
  if (h > w):
    w = h
    x = hcenter - (w/2)
  else:
    h = w
    y = vcenter - (h/2)
  return (x-padding, y-padding, w+padding, h+padding)

font = cv2.FONT_HERSHEY_SIMPLEX

# path = "./images/MNISTsamples.png"
path = "./images/numbers.jpg"
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
img = cv2.pyrDown(img)
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bw = cv2.GaussianBlur(bw, (7,7), 0)
ret, thbw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY_INV)
thbw = cv2.erode(thbw, np.ones((2,2), np.uint8), iterations = 2)
image, cntrs, hier = cv2.findContours(thbw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rectangles = []

for c in cntrs:
  r = x,y,w,h = cv2.boundingRect(c)
  a = cv2.contourArea(c)
  b = (img.shape[0]-3) * (img.shape[1] - 3)
  
  if not a == b:
    rectangles.append(r)


for r in rectangles:
  x,y,w,h = wrap_digit(r) 
  cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
  
  print f.shape
  
  try:
    roi = deskew(cv2.resize(thbw[y:y+h, x:x+w], (28, 28), interpolation = cv2.INTER_CUBIC))
    f = np.array([roi.copy().ravel()])
    label = nn.predict_label(f)[0]
    print label
    # digit_class = int(nn.predict_label([roi.copy().ravel()])[0])
    # print "Label: %d" % digit_class
    cv2.putText(img, "%s" % label, (x, y-1), font, 1, (0, 255, 0))
  except:
    continue
  

cv2.imshow("thbw", thbw)
cv2.imshow("contours", img)
cv2.imwrite("sample.jpg", img)
cv2.waitKey()
