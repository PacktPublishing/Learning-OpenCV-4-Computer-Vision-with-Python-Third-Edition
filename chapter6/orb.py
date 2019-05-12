import numpy as np
import cv2
from matplotlib import pyplot as plt

# query and test images
img1 = cv2.imread('../images/manowar_logo.png',0)
img2 = cv2.imread('../images/manowar_single.jpg',0)

# create the ORB detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# brute force matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)

# Sort by distance.
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(img1,kp1,img2,kp2, matches[:25], img2,flags=2)

plt.imshow(img3),plt.show()
