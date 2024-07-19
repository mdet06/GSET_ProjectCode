
################################################################Archibald aka Archie###################################################################

import cv2
import numpy as np
from PIL import Image

#Input image path from user.
file = input("Paste file path: ")

h  = int(input("height of the image in physical dimensions (um): "))
w  = int(input("width of the image in physical dimensions (um): "))

Coord = []

#Read image file.
img = cv2.imread(file)
rows, cols = img.shape[:2]


im = Image.open(file)
x = im.size[0]
y = im.size[1]

#Defines different functions for denoising and sharpening.
def kernel():
    kernel_25 = np.ones((25, 25), np.float32) / 625.0
    output_kernel = cv2.filter2D(img, -1, kernel_25)
    cv2.imwrite("img.png", output_kernel)
def blur():
    output_blur = cv2.blur(img, (25, 25))
    cv2.imwrite("img.png", output_blur)
def box():
    output_box = cv2.boxFilter(img, -1, (5,5), normalize=False)
    cv2.imwrite("img.png", output_box)
def gaus():
    output_gaus = cv2.GaussianBlur(img, (5,5), 0)
    cv2.imwrite("img.png", output_gaus)
def med():
    output_med = cv2.medianBlur(img, 5)
    cv2.imwrite("img.png", output_med)
def bil():
    output_bil = cv2.bilateralFilter(img, 5,6,6)
    cv2.imwrite("img.png", output_bil)

gaussian_blur = cv2.GaussianBlur(img, (7,7), 2)

def sharp1():
    sharpened1 = cv2.addWeighted(img,1.5,gaussian_blur, -0.5,0)
    cv2.imwrite("img.png", sharpened1)

def sharp2():
    sharpened2 = cv2.addWeighted(img,3.5,gaussian_blur, -2.5,0)
    cv2.imwrite("img.png", sharpened2)

sharpened3 = cv2.addWeighted(img,7.5,gaussian_blur, -6.5,0)

def sharp3():
    cv2.imwrite("img.png", sharpened3)

def binary():
    image_color = cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/img.png", cv2.IMREAD_GRAYSCALE)
    thres = 128
    img_bw = cv2.threshold(image_color, thres, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite("superficial_bw.png", img_bw)



def record_white_pixel_coordinates(input_image, output_file):
   # Load the image
   image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)




   # Ensure the image is binary
   _, binary_image = cv2.threshold(image, 225, 255, cv2.THRESH_BINARY)


   # Save the binary image
   cv2.imwrite(binary_image_output_path, binary_image)
   white_pixel_coords = np.column_stack(np.where(binary_image == 255))
   put_Coord_Dimen_Arr(white_pixel_coords, w, h, binary_image_output_path)
   # Write coordinates to the file

   print(Coord)
   with open(output_file, 'w') as f:
       for coord in Coord:
           f.write(f"{coord[0]},{coord[1]}\n")

def calculate_vessel_diameters(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.medianBlur(img, 5)
    
    im = Image.open(image_path)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    diameters = []

    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        diameter = radius * 2
        diameters.append(diameter)
        
    Lx = im.size[0]

    for d in range(int(len(diameters))):
        diameters[d] = get_Coord_Dimen(diameters[d], Lx, w)
   

    return diameters

def put_Coord_Dimen_Arr(arrays, dimen_w, dimen_h, input_path):
    # im = Image.open(file)
    # pixel_x = im.size[0]
    # pixel_y = im.size[1]

    im = Image.open(input_path)
    Lx = im.size[0]
    Ly = im.size[1]

    for x in range(Lx):
        for y in range(Ly):
            Coord.append([get_Coord_Dimen(x, Lx,dimen_w), get_Coord_Dimen(y, Ly, dimen_h)])


def put_Coord_Dimen(dimen_w, dimen_h, input_path):
   # im = Image.open(file)
   # pixel_x = im.size[0]
   # pixel_y = im.size[1]

   im = Image.open(input_path)
   Lx = im.size[0]
   Ly = im.size[1]

   for x in range(Lx):
       for y in range(Ly):
           Coord.append([get_Coord_Dimen(x, Lx,dimen_w), get_Coord_Dimen(y, Ly, dimen_h)])
#Coord.append(get_Coord_Dimen_x( dimen_w, x), get_Coord_Dimen_y(dimen_h, y))

def get_Coord_Dimen(x, P_L, R_L):
   return (x/P_L) * R_L


#problem - i am not sure if the diameter is always form the x axis  could be a different proprtion depending on the vessel's rotation
def calculate_vessel_diameter(image_path):
   # Load image
   img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
   Lx = im.size[0]
  
   # Example preprocessing (you might need different steps based on your images)
   img = cv2.medianBlur(img, 5)
  
   # Example vessel detection (you might need different methods based on your images)
   _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
   contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
   # Find the contour with maximum area (assuming it's the vessel)
   max_contour = max(contours, key=cv2.contourArea)
  
   # Fit a circle to the contour to find diameter
   (x, y), radius = cv2.minEnclosingCircle(max_contour)
   diameter = radius * 2


   a = get_Coord_Dimen(diameter, Lx, w)#is it width


  
   return a

def code():
    cv2.imwrite("sharpened3.png", sharpened3)

    supersharp = cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/sharpened3.png")
    
    output_box = cv2.boxFilter(supersharp, -1, (5,5), normalize=False)

    cv2.imwrite("box_wo_blur.png", output_box)

    image_color = cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/box_wo_blur.png", cv2.IMREAD_GRAYSCALE)
    thres = 225
    img_bw = cv2.threshold(image_color, thres, 255, cv2.THRESH_BINARY)[1]

    cv2.imshow("box, sharp bw", img_bw)
    cv2.imwrite("box_sharp_bw.png", img_bw)
    cv2.imshow("original", img)

    cv2.waitKey(0)

    record_white_pixel_coordinates(image_path, output_file)

code()
