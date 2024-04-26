import cv2
import numpy as np

img = cv2.imread('../images/houghlines5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 120)

lines = cv2.HoughLinesP(edges, rho=1,
                        theta=np.pi/180.0,
                        threshold=20,
                        minLineLength=40,
                        maxLineGap=5)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("edges", edges)
cv2.imshow("lines", img)
cv2.waitKey()
cv2.destroyAllWindows()
