
from PIL import Image

#make it so the new physical dimension coordinates are stored on array

file = input("Paste file path: ")

h  = int(input("height of the image in physical dimeninsions (um): "))
w  = int(input("width of the image in physical dimeninsions (um): "))
im = Image.open(file)

#r, g, b = im.getpixel((x,y))
            # if (r==255) and (g == 255) and (b == 255):
            #     final_White_Coord.append(get_Coord_Dimen_x( w, x), get_Coord_Dimen_y(h, y))

Coord = []

#implement h and w variable
def put_Coord_Dimen(dimen_w, dimen_h):
    pixel_x = im.size[0]
    pixel_y = im.size[1]
    for x in range (pixel_x):
        for y in range(pixel_y):
            Coord.append((get_Coord_Dimen_x( dimen_w, x), get_Coord_Dimen_y(dimen_h, y)))


#Coord.append(get_Coord_Dimen_x( dimen_w, x), get_Coord_Dimen_y(dimen_h, y))

def get_Coord_Dimen_x( a, x):
   pixel_length_x = im.size[1]
   
   return (x/pixel_length_x) *a
   


def get_Coord_Dimen_y( b, y):
   pixel_length_y = im.size[0]
   return (y/pixel_length_y) * b
   
    
put_Coord_Dimen(h,w)

print(Coord)
