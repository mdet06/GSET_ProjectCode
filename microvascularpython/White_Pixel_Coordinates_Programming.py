from numpy import append
import matplotlib.pyplot as plt


pixel_array = plt.imread(img)
white_pixel_array = []

height, width, depth = img.shape

for h in range(height):
    for w in range(width):
        for c in range (depth):
            if pixel_array[h,w,c] == (255,255,255):
                append(h,w,c)

#not sure if this works - cant upload image
