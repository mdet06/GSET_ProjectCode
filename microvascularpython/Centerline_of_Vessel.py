import cv2
import numpy as np
from skimage import morphology

image = cv2.imread('vessel_image.jpg')
original_image = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blurred, 30, 150)

skeleton = morphology.skeletonize(edges / 255)

skeleton = skeleton.astype(np.uint8) * 255
contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

contour = max(contours, key=cv2.contourArea)
rows, cols = skeleton.shape[:2]
[vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)

cv2.line(original_image, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
cv2.imshow('Original Image', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#This has not been tested yet so we should prob check - also it has gray scale so instead of using that we could just use the function we had already completed for that
