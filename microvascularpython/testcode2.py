import cv2
import numpy as np

img = cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/deep_layer.png")
rows, cols = img.shape[:2]

#kernel blurring (not as effective)
kernel_25 = np.ones((25, 25), np.float32) / 625.0
output_kernel = cv2.filter2D(img, -1, kernel_25)

#Boxfilter and blur function blurring.
output_blur = cv2.blur(img, (25, 25))
output_box = cv2.boxFilter(img, -1, (5,5), normalize=False)
output_gaus = cv2.GaussianBlur(img, (5,5), 0)

'''median blur (noise reduction): median blur replaces pixel values with the median of all
values lying in kernel area:'''
output_med = cv2.medianBlur(img, 5)

'''Bilateral filtering (noise reduction + preserving edges).
Makes sure that only those pixels having intensity almost same as target
pixel are confirmed.'''
output_bil = cv2.bilateralFilter(img, 5,6,6)

#def gaus
gaussian_blur = cv2.GaussianBlur(img, (7,7), 2)

#Sharpening with addWeighted()
sharpened1 = cv2.addWeighted(img,1.5,gaussian_blur, -0.5,0)
sharpened2 = cv2.addWeighted(img,3.5,gaussian_blur, -2.5,0)
sharpened3 = cv2.addWeighted(img,7.5,gaussian_blur, -6.5,0)

cv2.imwrite("z.png", output_blur)

img = cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/z.png")
rows, cols = img.shape[:2]

cv2.imwrite("za.png", output_box)

img = cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/za.png")
rows, cols = img.shape[:2]

def main():
    image_color = cv2.imread("za.png", cv2.IMREAD_GRAYSCALE)
    thres = 128
    img_bw = cv2.threshold(image_color, thres, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite("zb.png", img_bw)
main()

'''
cv2.imwrite

cv2.imshow("two blur", output_med)
cv2.imshow("one blur", output_gaus)
cv2.imshow("Original", img)
cv2.waitKey(0)'''