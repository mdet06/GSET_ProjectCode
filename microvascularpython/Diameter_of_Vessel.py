import cv2
import numpy as np


image = cv2.imread('image name.jpg')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


blurred = cv2.GaussianBlur(gray, (5, 5), 0)


edges = cv2.Canny(blurred, 50, 150)


contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for contour in contours:
    ((x, y), radius) = cv2.minEnclosingCircle(contour)
    
    diameter = radius * 2
    cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
    cv2.putText(image, f'Diameter: {diameter:.2f}', (int(x - radius), int(y - radius - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow('Vessel Diameter Measurement', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#not tested - may need to be checked
