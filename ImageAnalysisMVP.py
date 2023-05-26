import cv2
import numpy as np
from RemoveOutliers import replace_outliers_with_surrounding_color



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


def convolutions(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
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

def k_means(image):
    np.random.seed(0)
    newImage = image
    pixel_values = newImage.reshape((-1,3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
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

    maxMask = 0.0
    mask = 0
    for x in range(k):
        maxCenter = np.max(centers[x])
        if(maxCenter> maxMask):
            mask = x
            maxMask = maxCenter
    
    masked_image[labels != mask] = [0,0,0]


    masked_image = masked_image.reshape(newImage.shape)

    return masked_image

def unDistort(image):
    focal_length = 13870.866142 # mm
    image_width = 3024
    image_height = 4032
    principal_point = (image_width/2, image_height/2)
    distortion_coefficients = np.array([-0.029, 0.144, 0.001, -0.002, 0.0])
    fx = fy = focal_length * image_width / 36
    camera_matrix = np.array([[fx, 0, principal_point[0]],
                          [0, fy, principal_point[1]],
                          [0, 0, 1]])

    undistorted_img = cv2.undistort(image, camera_matrix, distortion_coefficients)
    return undistorted_img


def findCircles(image):
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = image

    img_blur = cv2.medianBlur(img, 5)

    # Detect circles using HoughCircles function
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=10, minRadius=10, maxRadius=15)

    # Draw detected circles on the original image
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 0, 255), 2)


    print(circles)
    return image

def findBoxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            boxes.append(box)
            cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    
    # sort boxes by area, smallest to largest
    boxes = sorted(boxes, key=cv2.contourArea)
    return image, boxes


def findCirclesAndBoxes(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.medianBlur(img, 5)

    # Detect circles using HoughCircles function
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=10, minRadius=10, maxRadius=15)

    # Find boxes using Canny edge detection and contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Remove small boxes and sort the remaining boxes by area in descending order
    boxes = [cv2.boxPoints(cv2.minAreaRect(contour)).astype(int) for contour in contours if len(contour) >= 3 and cv2.contourArea(contour) > 100]
    boxes = sorted(boxes, key=cv2.contourArea, reverse=True)
    
    # Remove circles that intersect with boxes
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            circle_center = (int(x), int(y))
            for box in boxes:
                distances = [cv2.pointPolygonTest(box, circle_center, True) for box in boxes]
                if max(distances) >= -r:
                    circles = np.delete(circles, np.where((circles == [x, y, r]).all(axis=1))[0], axis=0)
                    break

    # Draw remaining circles and boxes on the original image
    if circles is not None:
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 0, 255), 2)
    if boxes:
        for box in boxes:
            cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    
    return image




def showImage(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




image = cv2.imread(r"C:\Users\rasmu\OneDrive\Billeder\Filmrulle\WIN_20230331_11_09_17_Pro.jpg")
image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
image = scaleImage(image,80)
image = k_means(image)

image = convolutions(image)
image = replace_outliers_with_surrounding_color(image, 60)
#image = imagePixelManipulate2(image,100,0,0)
image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
image = convolutions(image)


mask = image
# opening_kernel_size = 2
# closing_kernel_size = 10

# # Create rectangular kernels for opening and closing
# opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (opening_kernel_size, opening_kernel_size))
# closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (closing_kernel_size, closing_kernel_size))

# # Perform opening to remove stray pixels
# opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)

# # Perform closing to fill in gaps
# closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, closing_kernel)

# Display the original and closed masks side by side
showImage(findCirclesAndBoxes(mask))