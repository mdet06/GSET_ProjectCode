import cv2
import numpy as np

img = cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/retina_deep1.png")
rows, cols = img.shape[:2]

#kernel blurring (not as effective)
kernel_25 = np.ones((25, 25), np.float32) / 625.0
output_kernel = cv2.filter2D(img, -1, kernel_25)

#Boxfilter and blur function blurring.
output_blur = cv2.blur(img, (25, 25))
output_box = cv2.boxFilter(img, -1, (5,5), normalize=False)

#gaussian blur
output_gaus = cv2.GaussianBlur(img, (5,5), 0)

'''median blur (noise reduction): median blur replaces pixel values with the median of all
values lying in kernel area:'''
output_med = cv2.medianBlur(img, 5)

'''Bilateral filtering (noise reduction + preserving edges).
Makes sure that only those pixels having intensity almost same as target
pixel are confirmed.'''
output_bil = cv2.bilateralFilter(img, 5,6,6)

cv2.imshow("kernel blur", output_kernel)
cv2.imshow("Blur() output", output_blur)
cv2.imshow("Box filter", output_box)
cv2.imshow("Gaussian", output_gaus)
cv2.imshow("Bilateral", output_bil)
cv2.imshow("Median Blur", output_med)

cv2.imshow("Original", img)
cv2.waitKey(0)

#Use control z to end program (until coded in)