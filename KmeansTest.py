from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import cv2
# Load the image
image = np.array(Image.open('testImage2.jpg'))

# Reshape the image for clustering

h, w, d = image.shape
pixels = np.reshape(image, (h*w, d))

# Perform k-means clustering with k=8 (for example)
kmeans = KMeans(n_clusters=8).fit(pixels)

# Assign a color to each cluster (e.g. using the cluster centers)
colors = kmeans.cluster_centers_.astype(int)

# Assign each pixel to its corresponding cluster color
labels = kmeans.predict(pixels)
clustered_pixels = colors[labels]

# Reshape the image back to its original shape
clustered_image = np.reshape(clustered_pixels, (h, w, d))

# Save the clustered image
Image.fromarray(clustered_image.astype(np.uint8)).save('clustered_image.jpg')