import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import cv2

#   Rasmus Bo Thomsen  S211708                    Mathilde Shalimon Elia S215811

def scaleImage(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    new_img_size = (resized.shape[1] - (resized.shape[1] % 32), resized.shape[0] - (resized.shape[0] % 32))
    resized_img = cv2.resize(resized, new_img_size)
    return resized_img
# Load and preprocess the image
image = cv2.imread("DL_Photos\WIN_20230329_10_13_33_Pro.jpg") 
image = scaleImage(image, 10)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
X = image.reshape(-1, 3)  # Reshape the image to a 2D array
# Perform Agglomerative Clustering
n_clusters = 5  # Number of clusters
clustering = AgglomerativeClustering(n_clusters=n_clusters)
cluster_labels = clustering.fit_predict(X)

# Reshape the cluster labels back to the original image shape
cluster_labels = cluster_labels.reshape(image.shape[:2])

# Display each cluster of the image
plt.figure(figsize=(12, 6))
for cluster in range(n_clusters):
    plt.subplot(1, n_clusters, cluster + 1)
    cluster_image = np.zeros_like(image)
    cluster_image[cluster_labels == cluster] = image[cluster_labels == cluster]
    plt.imshow(cluster_image)
    plt.axis('off')
    plt.title(f'Cluster {cluster + 1}')

plt.suptitle('Agglomerative Clustering: Clustered Images')
plt.tight_layout()
plt.show()