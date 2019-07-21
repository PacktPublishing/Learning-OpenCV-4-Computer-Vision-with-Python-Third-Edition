import cv2


face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
img = cv2.imread('../images/vikings.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
  
cv2.namedWindow('Vikings Detected!')
cv2.imshow('Vikings Detected!', img)
cv2.imwrite('./vikings_detected.jpg', img)
cv2.waitKey(0)
