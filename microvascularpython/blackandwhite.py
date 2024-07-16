import cv2

file = input("File path and name: ")

def main():
    image_color = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    thres = 128
    img_bw = cv2.threshold(image_color, thres, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite("bwgoose.png", img_bw)

convert = input("Convert to black and white? ")

if convert == "yes" or "Yes":
    main()