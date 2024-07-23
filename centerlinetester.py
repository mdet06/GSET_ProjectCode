import cv2
import numpy as np

#Input image path from user.
file = input("Paste file path: ")

#Read image file.
img = cv2.imread(file)
rows, cols = img.shape[:2]

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

def record_white_pixel_coordinates(image_path, output_file):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded correctly
    if image is None:
        print(f"Error loading image {image_path}")
        return

    # Ensure the image is binary
    _, binary_image = cv2.threshold(image, 225, 255, cv2.THRESH_BINARY)

    # Save the binary image
    cv2.imwrite(binary_image_output_path, binary_image)
    print(f"Binary image saved as {binary_image_output_path}")

    # Find coordinates of white pixels
    white_pixel_coords = np.column_stack(np.where(binary_image == 255))

    # Write coordinates to the file
    with open(output_file, 'w') as f:
        for coord in white_pixel_coords:
            f.write(f"{coord[0]},{coord[1]}\n")

# Example usage
image_path = '/Users/daisymaturo/Downloads/microvascularpython/box_wo_blur.png'
output_file = 'white_pixel_coords.txt'
binary_image_output_path = 'binary_image.png'
record_white_pixel_coordinates(image_path, output_file)


def find_centerlines(binary_image_path):
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    
    skeleton = cv2.ximgproc.thinning(binary_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.circle(output_image, (cX, cY), 5, (0, 0, 255), -1)

    cv2.imwrite('segmented_vessels.png', output_image)
    cv2.imshow('Segmented Vessels and Centerlines', output_image)
    cv2.waitKey(0)



def code():
    cv2.imwrite("sharpened3.png", sharpened3)

    supersharp = cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/sharpened3.png")
    
    output_box = cv2.boxFilter(supersharp, -1, (5,5), normalize=False)

    cv2.imshow("original", img)
    cv2.imwrite("box_wo_blur.png", output_box)
    cv2.imshow("boxbox", output_box)
    
    image_color = cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/box_wo_blur.png", cv2.IMREAD_GRAYSCALE)
    thres = 225
    img_bw = cv2.threshold(image_color, thres, 255, cv2.THRESH_BINARY)[1]

    cv2.imshow("box, sharp bw", img_bw)
    cv2.imwrite("box_sharp_bw.png", img_bw)

    blurry = cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/box_sharp_bw.png")
    output_blur = cv2.blur(blurry, (25, 25))

    cv2.imwrite("box_sharp_bw_blur.png", output_blur)
    cv2.imshow("box_sharp_bw_blur.png", output_blur)

    sharpy = cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/box_sharp_bw_blur.png")
    
    gaussian_blur = cv2.GaussianBlur(sharpy, (7,7), 2)

    sharpened1 = cv2.addWeighted(sharpy,1.5,gaussian_blur, -0.5,0)

    cv2.imwrite("bsbbsharp.png", sharpened1)

    rebinary = cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/bsbbsharp.png")
    thres = 100
    img_bw = cv2.threshold(rebinary, thres, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite("binary.png", img_bw)
    cv2.imshow("final binary", img_bw)
    record_white_pixel_coordinates(image_path, output_file)
    find_centerlines('/Users/daisymaturo/Downloads/microvascularpython/binary.png')

code()
