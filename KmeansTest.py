
import numpy as np
from PIL import Image
import cv2
# Load the image
img = cv2.imread("testImage2.jpg")

# Reshape the image for clustering

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Split the LAB channels
l, a, b = cv2.split(lab)

# Create a CLAHE object and apply it to the L channel
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
l_clahe = clahe.apply(l)

# Merge the CLAHE-adjusted L channel with the original A and B channels
lab_clahe = cv2.merge((l_clahe, a, b))

# Convert the LAB image back to RGB color space
rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)



def scaleImage(image):
    scale_percent = 20 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    new_img_size = (resized.shape[1] - (resized.shape[1] % 32), resized.shape[0] - (resized.shape[0] % 32))
    resized_img = cv2.resize(resized, new_img_size)
    return resized_img

# Display the original and contrast-adjusted images side-by-side
printIMage =  scaleImage(rgb_clahe)
cv2.imshow("Contrast-Adjusted Image",printIMage)
cv2.waitKey(0)
cv2.destroyAllWindows()