import cv2

def Capture_Event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")


if __name__=="__main__":
    img = cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/retina_superficial2.png")
    cv2.imshow()