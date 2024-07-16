import cv2
import numpy as np

#Reading img
img =cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/retina_superficial2.png")

#Gaussian kernel for sharpening
    

#Sharpening with addWeighted()
def to_Sharpened(input_image):
    gaussian_blur = cv2.GaussianBlur(img, (7,7), 2)
    sharpened1 = cv2.addWeighted(img,1.5,gaussian_blur, -0.5,0)
    sharpened2 = cv2.addWeighted(img,3.5,gaussian_blur, -2.5,0)
    sharpened3 = cv2.addWeighted(img,7.5,gaussian_blur, -6.5,0)
    cv2.imshow("Sharpened 3", sharpened3)
    cv2.imshow("Sharpened 2", sharpened2)
    cv2.imshow("Sharpened 1", sharpened1)


#Showing sharpened images
to_Sharpened(img)

cv2.imshow("Original", img)
cv2.waitKey(0)
