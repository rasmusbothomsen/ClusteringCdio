import cv2
import numpy as np

img = cv2.imread('testImage2.jpg')

# Define a kernel for convolution
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

# Apply the convolution operation
result = cv2.filter2D(img, -1, kernel)

# Display the original and the resulting image
cv2.imshow('Convolved Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()