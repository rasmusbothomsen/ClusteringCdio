import cv2
import numpy as np

# Load the image
img = cv2.imread('input_image.jpg')

# Reshape the image to a 2D array of pixels
pixel_vals = img.reshape((-1,3))

# Convert pixel values to float32
pixel_vals = np.float32(pixel_vals)

# Define stopping criteria for the algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# Run k-means clustering on the pixel values
k = 8
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert the centers back to uint8 and reshape them to the original image size
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(img.shape)

# Display the segmented image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
