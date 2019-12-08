import cv2

image = cv2.imread('MyPic.png')
cv2.imwrite('MyPic.jpg', image)