#This may or may not work

import cv2
import numpy as np

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
