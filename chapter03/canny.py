import cv2
import numpy as np

img = cv2.imread("../images/statue_small.jpg",
                 cv2.IMREAD_GRAYSCALE)
canny_img = cv2.Canny(img, 200, 300)
cv2.imshow("canny", canny_img)
cv2.waitKey()
cv2.destroyAllWindows()
