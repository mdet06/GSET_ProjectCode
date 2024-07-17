from PIL import Image

def to_micrometers(width_micrometers, height_micrometers):

    dpi_per_um = 39.3701  # Conversion factor from inches to micrometers
    width_pixels = int(width_micrometers * dpi_per_um)
    height_pixels = int(height_micrometers * dpi_per_um)

    # Resize the image
    img = img.resize((width_pixels, height_pixels))
    return img

#not tested
