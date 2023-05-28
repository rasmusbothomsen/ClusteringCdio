import os
import random
import cv2
from cv2 import Mat
import numpy as np
from RemoveOutliers import replace_outliers_with_surrounding_color
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from skimage import morphology
from sklearn.metrics import silhouette_samples
from skimage.segmentation import mark_boundaries



def setImageVariables(image):
    global masked_image,centers,labels,pixel_values,segmented_image
    masked_image,centers,labels,pixel_values,segmented_image = k_means(image)

def k_means(image, showClusters=False):
    
    newImage = image
    # newImage = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # ret,thresh = cv2.threshold(newImage,140,255,cv2.THRESH_BINARY)
    # cv2.normalize(thresh, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # newImage = thresh
    pixel_values = newImage.reshape((-1,3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 9
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

    maxMask = 0.0
    mask = 0
    for x in range(k):
        maxCenter = np.max(centers[x])
        if(maxCenter> maxMask):
            mask = x
            maxMask = maxCenter
        if showClusters:
            tmpimg = masked_image.copy()
            tmpimg[labels != x] = [0,0,0]
            tmpimg = tmpimg.reshape(newImage.shape)
            showImage(tmpimg)
    
    
    masked_image[labels != mask] = [0,0,0]
    masked_image = masked_image.reshape(newImage.shape)
    
    return masked_image,centers,labels,pixel_values,segmented_image
def showImage(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def kmeans_dendrogram(image, k, sample_size):
    np.random.seed(10)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    sample_indices = random.sample(range(len(pixel_values)), sample_size)
    sampled_pixel_values = pixel_values[sample_indices]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    _, labels, centers = cv2.kmeans(sampled_pixel_values, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()

    linked = linkage(sampled_pixel_values, method='average')
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()


def kMeansVisualization(image):

    # Create meaningful graphs
    plt.figure(figsize=(10, 6))

    # Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Segmented Image
    plt.subplot(2, 2, 2)
    plt.imshow(segmented_image)
    plt.title('Segmented Image')
    plt.axis('off')

    # Color Palette
    plt.subplot(2, 2, 3)
    plt.imshow([centers])
    plt.title('Color Palette')
    plt.axis('off')

    # Masked Image
    plt.subplot(2, 2, 4)
    plt.imshow(masked_image)
    plt.title('Masked Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def clusterVisualization(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create meaningful graphs
    plt.figure(figsize=(12, 10))

    # Original Image
    plt.subplot(3, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # Segmented Image
    plt.subplot(3, 3, 2)
    plt.imshow(masked_image)
    plt.title('Segmented Image')
    plt.axis('off')

    # Color Palette
    plt.subplot(3, 3, 3)
    plt.imshow([centers])
    plt.title('Color Palette')
    plt.axis('off')

    # Cluster Distribution
    plt.subplot(3, 3, 4)
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.bar(unique_labels, counts)
    plt.title('Cluster Distribution')
    plt.xlabel('Cluster')
    plt.ylabel('Pixel Count')

    # Cluster Centroids
    plt.subplot(3, 3, 5)
    color_patches = np.reshape(centers, (1, -1, 3))
    plt.imshow(color_patches)
    plt.title('Cluster Centroids')
    plt.axis('off')

    # Clustered Pixels
    plt.subplot(3, 3, 6)
    plt.scatter(pixel_values[:, 0], pixel_values[:, 1], c=labels, cmap='rainbow', s=1)
    plt.title('Clustered Pixels')
    plt.xlabel('Red')
    plt.ylabel('Green')

    # Color Histogram
    plt.subplot(3, 1, 3)
    color_labels = ['Red', 'Green', 'Blue']
    for i in range(3):
        hist, bins, _ = plt.hist(segmented_image[:, :, i].flatten(), bins=256, color=color_labels[i], alpha=0.7, label=color_labels[i])
    plt.title('Color Histogram')
    plt.xlabel('Color Intensity')
    plt.ylabel('Pixel Count')
    plt.legend()

    plt.tight_layout()
    plt.show()


def ColorDistibutionCompare(image):

    original_hist, _ = np.histogram(image.reshape(-1, 3), bins=256, range=[0, 255])

    # Calculate the color distribution for the segmented image
    segmented_hist, _ = np.histogram(segmented_image.reshape(-1, 3), bins=256, range=[0, 255])
    showImage(segmented_image)
    # Plot the color distribution comparison
    plt.plot(original_hist, label='Original Image')
    plt.plot(segmented_hist, label='Segmented Image')
    plt.title('Color Distribution Comparison')
    plt.xlabel('Color Intensity')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.show()

def ClusterBoundaries(image):

    boundaries_image = mark_boundaries(image, labels.reshape(image.shape[:2]), color=(0, 0, 0), outline_color=(1, 1, 1))

    # Display the original image with cluster boundaries
    plt.imshow(boundaries_image)
    plt.title('Cluster Boundaries')
    plt.axis('off')
    plt.show()


def Colormapping(image):
    color_mapped_image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

    # Convert the color mapped image from BGR to RGB
    color_mapped_image_rgb = cv2.cvtColor(color_mapped_image, cv2.COLOR_BGR2RGB)

    # Display the color mapped image
    plt.imshow(color_mapped_image_rgb)
    plt.axis('off')
    plt.title('Color Mapped Image')
    plt.show()
def ClusterPixelcount(image):
    cluster_counts = np.bincount(labels)
    # Plot the pixel counts for each cluster
    plt.bar(range(len(cluster_counts)), cluster_counts)
    plt.title('Cluster Pixel Counts')
    plt.xlabel('Cluster')
    plt.ylabel('Pixel Count')
    plt.show()

def main():
    image = cv2.imread("DL_Photos\WIN_20230329_10_13_33_Pro.jpg")
    #setImageVariables(image)


    Colormapping(image)

if __name__ == "__main__":
    main()