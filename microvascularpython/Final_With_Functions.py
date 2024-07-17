#may or may not work

import cv2
import numpy as np
from numpy import append
import matplotlib.pyplot as plt



img = cv2.imwrite(input("Write filename: "), input("Write name of image: "))
h = int(input("height of image in micrometers: "))
w = int(input("Width of image in micrometers: "))
rows, cols = img.shape[:2]

#for white pixel coordinates
pixel_array = plt.imread(img)
white_pixel_array = []


#kernel blurring (not as effective)


def to_Kernel(input_image):
    kernel_25 = np.ones((25, 25), np.float32) / 625.0
    output_kernel = cv2.filter2D(input_image, -1, kernel_25)
    #cv2.imshow("kernel blur", output_kernel)
    return output_kernel



def to_BoxFilter(input_image):
    output_blur = cv2.blur(input_image, (25, 25)) #this does nothing unless you want to add another function for blurring
    output_box = cv2.boxFilter(input_image, -1, (5,5), normalize=False)
    output_gaus = cv2.GaussianBlur(input_image, (5,5), 0) #this does nothing unless you want to right another function for guassian blur
    #cv2.imshow("Box filter", output_box)
    return output_box




'''median blur (noise reduction): median blur replaces pixel values with the median of all
values lying in kernel area:'''


def to_Median(input_image):
    output_med = cv2.medianBlur(input_image, 5)
    #cv2.imshow("Median Blur", output_med)
    return output_med

'''Bilateral filtering (noise reduction + preserving edges).
Makes sure that only those pixels having intensity almost same as target
pixel are confirmed.'''

def to_Bilateral(input_image):
    output_bil = cv2.bilateralFilter(input_image, 5,6,6)
    #cv2.imshow("Bilateral", output_bil)
    return output_bil

#def gaus

#gaussian blur
def to_Guassian(input_image):
    gaussian_blur = cv2.GaussianBlur(input_image, (7,7), 2) # - why is this one different ---- output_gaus = cv2.GaussianBlur(img, (5,5), 0)
    #cv2.imshow("Gaussian", gaussian_blur)
    return gaussian_blur

def to_Sharpened(input_image):
    gaussian_blur = to_Guassian(input_image)
    sharpened1 = cv2.addWeighted(img,1.5,gaussian_blur, -0.5,0)
    sharpened2 = cv2.addWeighted(img,3.5,gaussian_blur, -2.5,0)
    sharpened3 = cv2.addWeighted(img,7.5,gaussian_blur, -6.5,0)
    # cv2.imshow("Sharpened 3", sharpened3)
    # cv2.imshow("Sharpened 2", sharpened2)
    # cv2.imshow("Sharpened 1", sharpened1)
    return sharpened3 #I dont know which one we should choose srry


def main():
    image_color = cv2.imread("za.png", cv2.IMREAD_GRAYSCALE)
    thres = 128
    img_bw = cv2.threshold(image_color, thres, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite("zb.png", img_bw)
    return img_bw
    
def calculate_vessel_diameter(image_path, threshold_low=30, threshold_high=150):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #instead we should call function for black and white

    edges = cv2.Canny(img, threshold_low, threshold_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    diameters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        diameter = max(w, h)
        diameters.append(diameter)
    if diameters:
        max_diameter = max(diameters)
    else:
        max_diameter = 0

    return max_diameter


# this is incase you want to put the image into multiple filters
a = main(img)
b = to_Kernel(a)
c = to_BoxFilter(b)
d = to_Median(c)
e = to_Bilateral(d)
f  = to_Guassian(e)
g = to_Sharpened(f)
cv2.imwrite("test.png", g)










