import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img0 = cv2.imread('anchors/tattoo_seed.jpg',
                  cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('anchors/hush.jpg',
                  cv2.IMREAD_GRAYSCALE)

# Perform SIFT feature detection and description.
sift = cv2.xfeatures2d.SIFT_create()
kp0, des0 = sift.detectAndCompute(img0, None)
kp1, des1 = sift.detectAndCompute(img1, None)

# Define FLANN-based matching parameters.
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Perform FLANN-based matching.
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des0, des1, k=2)

# Find all the good matches as per Lowe's ratio test.
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

if len(good_matches) >= MIN_MATCH_COUNT:
    src_pts = np.float32(
        [kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask_matches = mask.ravel().tolist()

    h,w = img0.shape
    pts = np.float32(
        [[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
else:
    print("Not enough matches are found - %d/%d" % \
          (len(good_matches), MIN_MATCH_COUNT))
    mask_matches = None

# Draw the matches that passed the ratio test.
img_matches = cv2.drawMatches(
    img0, kp0, img1, kp1, good_matches, None,
    matchColor=(0, 255, 0), singlePointColor=None,
    matchesMask=mask_matches, flags=2)

# Show the matches.
plt.imshow(img_matches)
plt.show()
