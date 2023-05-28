import cv2
import numpy as np
from RemoveOutliers import replace_outliers_with_surrounding_color
from ImagePixelManipulateTest import imagePixelManipulate as PM
import matplotlib as plt

def imagePixelManipulate(img, threshold, shadow_threshold=50, shadow_boost_factor=30):
    image = img.copy()
    data = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Boost pixel values if they are below a certain threshold (for shadows)
    mask = data[..., 0] < shadow_threshold
    data[mask, 0] += shadow_boost_factor
    data[mask, 1] = (data[mask, 1] + shadow_boost_factor*0.1).astype('uint8')
    data[mask, 2] =  (data[mask, 2] + shadow_boost_factor*0.1).astype('uint8')

    pixelValueMeans = np.round(np.median(data, axis=(0,1))).astype(np.uint8)

    diff = np.abs(data - pixelValueMeans)
    mask = np.sum(diff, axis=-1) <= threshold
    data[mask] = pixelValueMeans

    return cv2.cvtColor(data, cv2.COLOR_LAB2RGB)


def scaleImage(image):
    scale_percent = 100 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    new_img_size = (resized.shape[1] - (resized.shape[1] % 32), resized.shape[0] - (resized.shape[0] % 32))
    resized_img = cv2.resize(resized, new_img_size)
    lab = cv2.cvtColor(resized_img, cv2.COLOR_BGR2LAB)
    

    # Split the LAB channels
    l, a, b = cv2.split(lab)

    # Create a CLAHE object and apply it to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)

    # Merge the CLAHE-adjusted L channel with the original A and B channels
    lab_clahe = cv2.merge((l_clahe, a, b))

    # Convert the LAB image back to RGB color space
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return rgb_clahe

image = cv2.imread("DL_Photos\WIN_20230329_10_13_42_Pro (2).jpg")
kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])

# Apply the convolution operation
#image = cv2.filter2D(image, -3, kernel)

resized_img = scaleImage(image=image)
resized_img = cv2.fastNlMeansDenoisingColored(resized_img,None,10,10,7,21)
resized_img =  cv2.filter2D(resized_img, -1, kernel)
#resized_img = imagePixelManipulate(resized_img, 100, shadow_threshold=100, shadow_boost_factor=0)
#resized_img = PM(resized_img, 100, shadow_threshold=150, shadow_boost_factor=0)
#resized_img = replace_outliers_with_surrounding_color(resized_img,100)
resized_img = replace_outliers_with_surrounding_color(resized_img,80)
newImage = resized_img

pixel_values = newImage.reshape((-1,3))
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 6
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()

segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(newImage.shape)


masked_image = np.copy(segmented_image)
# convert to the shape of a vector of pixel values
masked_image = masked_image.reshape((-1, 3))
# color (i.e cluster) to disable

mask = [2,3,4,5]

# for x in range(k):
#     if(x not in mask):
#         masked_image[labels == x] = [0, 0, 0]




# convert back to original shape
masked_image = masked_image.reshape(resized_img.shape)

#masked_image = replace_outliers_with_surrounding_color(masked_image,100)

img = cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY)

img_blur = cv2.medianBlur(img, 5)

# Detect circles using HoughCircles function
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 0.4, 20, param1=50, param2=30, minRadius=10, maxRadius=50)

# Draw detected circles on the original image
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(masked_image, (x, y), r, (0, 0, 255), 2)


print(circles)

cv2.imshow('kmeansIM', resized_img)
# add wait key. window waits until user presses a key
cv2.waitKey(0)
# and finally destroy/close all open windows
cv2.destroyAllWindows()
