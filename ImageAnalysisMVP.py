import os
import random
import cv2
import numpy as np
from RemoveOutliers import replace_outliers_with_surrounding_color
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from skimage import morphology



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def scaleImage(image,scale):
    scale_percent = scale # percent of original size
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
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)

    # Merge the CLAHE-adjusted L channel with the original A and B channels
    lab_clahe = cv2.merge((l_clahe, a, b))

    # Convert the LAB image back to RGB color space
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return rgb_clahe

# Performs Convolutions on image
def convolutions(image):
    kernel = np.array([
                        [0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]]
                        )
    resized_img =  cv2.filter2D(image, -1, kernel)
    return resized_img

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

def imagePixelManipulate2(img, threshold, shadow_threshold=50, shadow_boost_factor=30):
    image = img.copy()
    data = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    height, width, channels = image.shape
    printProgressBar(0, height, prefix='Progress:', suffix='Complete', length=50)
    visited = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            if not visited[y, x]:
                pixelValueMean = np.round(np.median(data[y, x], axis=0)).astype(np.uint8)
                # Boost pixel values if they are below a certain threshold (for shadows)
                if np.sum(data[y,x,0]) < shadow_threshold:
                    data[y, x, 0] += shadow_boost_factor
                    data[y, x, 1] += shadow_boost_factor*0.1
                    data[y, x, 2] += shadow_boost_factor*0.1

                queue = [(y, x)]
                visited[y, x] = True
                while queue:
                    curr_y, curr_x = queue.pop(0)
                    if np.sum(np.abs(data[curr_y, curr_x] - pixelValueMean)) <= threshold:
                        data[curr_y, curr_x] = pixelValueMean
                        for ny, nx in [(curr_y-1, curr_x), (curr_y+1, curr_x), (curr_y, curr_x-1), (curr_y, curr_x+1)]:
                            if ny >= 0 and ny < height and nx >= 0 and nx < width and not visited[ny, nx]:
                                queue.append((ny, nx))
                                visited[ny, nx] = True
        printProgressBar(y + 1, height, prefix='Progress:', suffix='Complete', length=50)

    return cv2.cvtColor(data, cv2.COLOR_LAB2BGR)

# This was only for testing
def sklearnKulster(image):
    pixels = image.reshape(-1, 3)

    # Perform K-means clustering
    n_clusters = 6  # Specify the number of clusters
    kmeans = KMeans(n_clusters=n_clusters,random_state=0,n_init="auto")
    kmeans.fit(pixels)
    kmeans.predict(pixels)
    kmeans.get_params(deep=True)
    labels = kmeans.labels_
   
    selected_cluster = 4  # Adjust this to the desired cluster index

# Filter the labels to keep only the selected cluster
    selected_labels = np.where(labels == selected_cluster, 255, 0).astype(np.uint8)

# Reshape the labels to the original image shape
    selected_pixels = selected_labels.reshape(image.shape[:2])

    showImage(selected_pixels)

def kmeans_dendrogram(image, k, sample_size):
    np.random.seed(10)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    sample_indices = random.sample(range(len(pixel_values)), sample_size)
    sampled_pixel_values = pixel_values

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


def k_means(image, showClusters=False):
    # Sets random seed to 0 to generate the same sequence
    np.random.seed(0)

    newImage = image
    newImage = scaleImage(image, 80)

    # newImage = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # ret,thresh = cv2.threshold(newImage,140,255,cv2.THRESH_BINARY)
    # cv2.normalize(thresh, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # newImage = thresh

    # Reshapes image into 2D array
    pixel_values = newImage.reshape((-1,3))
    # Converts pixel values to float32 datatype
    pixel_values = np.float32(pixel_values)

    # Gives criteria for when k-means should stop.
    # Set to 100 iterations or when the difference is less than 0.2.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Sets number of clusters to 9
    k = 9
    # Uses k-means algorithm. Returns label for each pixel and cluster centers.
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Converts centers to uint8 datatype
    centers = np.uint8(centers)
    # Flatten labels array to 1D array
    labels = labels.flatten()

    # Creates segmented image where each pixel is assigned a cluster.
    segmented_image = centers[labels.flatten()]
    # Reshapes image into its previous shape
    segmented_image = segmented_image.reshape(newImage.shape)

    # Creates copy of segmented image.
    masked_image = np.copy(segmented_image)
    # Reshapes image into 2D array.
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to disable

    # Finds cluster containing the white elements.
    maxMask = 0.0
    # Index for cluster containing balls.
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

    # Makes all other clusters black
    masked_image[labels != mask] = [0,0,0]
    # Reshapes image back.
    masked_image = masked_image.reshape(newImage.shape)

    return masked_image




# Draw the filtered contours on the image
def findCirclesAndBoxes(image):

    # Converts image from BGR to grascale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduces noice by blurring image
    img_blur = cv2.medianBlur(img, 5)

    # HoughCircles is used to find circles in the image.
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=10, minRadius=10, maxRadius=15)

    # Finds edges.
    edged = cv2.Canny(img_blur, 30, 200)
    # Finds contours.
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Removes small contours and sort the remaining based on area.
    boxes = [cv2.boxPoints(cv2.minAreaRect(contour)).astype(int) for contour in contours if len(contour) >= 3 and cv2.contourArea(contour) > 100]
    boxes = sorted(boxes, key=cv2.contourArea, reverse=True)

    # If circles and boxes overlap - remove circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            circle_center = (int(x), int(y))
            for box in boxes:
                distances = [cv2.pointPolygonTest(box, circle_center, True) for box in boxes]
                if max(distances) >= -r:
                    circles = np.delete(circles, np.where((circles == [x, y, r]).all(axis=1))[0], axis=0)
                    break

    # Add remaining circles and boxes on the original image
    if circles is not None:
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 0, 255), 2)
    if boxes:
        for box in boxes:
            cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    
    return boxes, image

#Calculates the relative pixel value
def convert_to_yolo_format(image_width, image_height, box):
    rect = cv2.minAreaRect(box)
    points = cv2.boxPoints(rect)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    x_center = (x_min + x_max) / (2 * image_width)
    y_center = (y_min + y_max) / (2 * image_height)
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return x_center, y_center, width, height



def showImage(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







def PrepareImages():
    directory = "train_images"
    output_dir = "yolo_labels"  # Directory to save the YOLO labels
    imageTestDir = "outputimageTest"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(imageTestDir, exist_ok=True)
    for fileName in os.listdir(directory):
        filePath = os.path.join(directory,fileName)
        outPath = os.path.join(imageTestDir,fileName)
        print(filePath)
        image = cv2.imread(filePath)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        image = scaleImage(image,80)
        image = k_means(image)

        image = convolutions(image)
        image = replace_outliers_with_surrounding_color(image, 60)
        #image = imagePixelManipulate2(image,100,0,0)
        image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
        image = convolutions(image)

        h,w,c = image.shape
        mask = image
        boxes, imagebox = findCirclesAndBoxes(mask)
        yolo_labels = list()

        for  box in boxes:
            label = convert_to_yolo_format(w,h,box)
            yolo_labels.append(label)

        with open(os.path.join(output_dir, f"{fileName[:-4]}.txt"), "w") as f:
            for index, label in enumerate(yolo_labels):
                x_center, y_center, width, height = label
                if(index == len(yolo_labels)-1):
                    line = f"1 {x_center} {y_center} {width} {height}\n"
                else:
                    line = f"0 {x_center} {y_center} {width} {height}\n"
                
                f.write(line)
        cv2.imwrite(outPath,imagebox)


def imageTest():

    image = cv2.imread("DL_Photos\WIN_20230329_10_13_33_Pro.jpg")
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    image = scaleImage(image,80)
    image = convolutions(image)
    image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    image = k_means(image, True)
    image = replace_outliers_with_surrounding_color(image, 60)
    image = convolutions(image)

    h,w,c = image.shape
    mask = image
    boxes, imagebox = findCirclesAndBoxes(mask)
    showImage(image)


def DendoGram():
    image = cv2.imread("DL_Photos\WIN_20230329_10_13_33_Pro.jpg")
    image = scaleImage(image,10)
    image = kmeans_dendrogram(image,9,0)
    showImage(image)

DendoGram()

