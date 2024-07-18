from PIL import Image


image_file = "/Users/madisondetrick/Documents/VisualStudios/Flower.png"
im = Image.open(image_file)
x = im.size[0] # return value is a tuple, ex.: (1200, 800)

y = im.size[1]



#a is micro meter length for x, b is micro meter length for y, x is pixel length for x coord, y is pixel length for y coord



def get_Coord_Dimen_x( a, x):
   pixel_length_x = im.size[1]
   
   return (x/pixel_length_x) *a
   


def get_Coord_Dimen_y( b, y):
   pixel_length_y = im.size[0]
   return (y/pixel_length_y) * b
   
    

ab = get_Coord_Dimen_x( 5, 6)
cd = get_Coord_Dimen_y( 5,6)

print(ab)
print(cd)
