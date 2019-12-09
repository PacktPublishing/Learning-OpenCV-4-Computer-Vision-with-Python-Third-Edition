import cv2
import numpy as np

import digits_ann as ANN

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])

def inside(r1, r2):
    x1,y1,w1,h1 = r1
    x2,y2,w2,h2 = r2
    if (x1 > x2) and (y1 > y2) and (x1+w1 < x2+w2) and \
            (y1+h1 < y2 + h2):
        return True
    else:
        return False

def wrap_digit(rect):
    x, y, w, h = rect
    hcenter = x + w//2
    vcenter = y + h//2
    roi = None
    if (h > w):
        w = h
        x = hcenter - (w//2)
    else:
        h = w
        y = vcenter - (h//2)
    padding = 5
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding
    return x, y, w, h

ann, test_data = ANN.train(ANN.create_ANN(58), 50000, 5)
font = cv2.FONT_HERSHEY_SIMPLEX

img_path = "./digit_images/digits_0.jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.GaussianBlur(gray, (7, 7), 0, gray)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
erode_kernel = np.ones((2, 2), np.uint8)
thresh = cv2.erode(thresh, erode_kernel, thresh, iterations=2)

if OPENCV_MAJOR_VERSION >= 4:
    # OpenCV 4 or a later version is being used.
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
else:
    # OpenCV 3 or an earlier version is being used.
    # cv2.findContours has an extra return value.
    # The extra return value is the thresholded image, which is
    # unchanged, so we can ignore it.
    _, contours, hier = cv2.findContours(thresh, cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE)

rectangles = []

for c in contours:
    r = cv2.boundingRect(c)
    a = cv2.contourArea(c)
    b = (img.shape[0] - 3) * (img.shape[1] - 3)

    is_inside = False
    for q in rectangles:
        if inside(r, q):
            is_inside = True
            break
    if not is_inside:
        if not a == b:
            rectangles.append(r)

for r in rectangles:
    x, y, w, h = wrap_digit(r) 
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
    roi = thresh[y:y+h, x:x+w]

    try:
        digit_class = int(ANN.predict(ann, roi)[0])
    except:
        continue
    cv2.putText(img, "%d" % digit_class, (x, y-1), font, 1,
                (0, 255, 0))

cv2.imshow("thresh", thresh)
cv2.imshow("detected and classified digits", img)
cv2.imwrite("detected_and_classified_digits.png", img)
cv2.waitKey()
