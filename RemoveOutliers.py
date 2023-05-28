import cv2
import numpy as np
from ImagePixelManipulateTest import printProgressBar

#   Rasmus Bo Thomsen  S211708                    Mathilde Shalimon Elia S215811


def replace_outliers_with_surrounding_color(img, threshold):
    # Convert image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split the LAB channels
    l, a, b = cv2.split(lab)

    # Compute the median pixel value in each channel
    l_median = np.median(l)
    a_median = np.median(a)
    b_median = np.median(b)

    # Compute the absolute difference between each pixel value and the median
    l_diff = np.abs(l - l_median)
    a_diff = np.abs(a - a_median)
    b_diff = np.abs(b - b_median)

    # Create a mask where pixel values are greater than the threshold
    mask = np.logical_or(np.logical_or(l_diff > threshold, a_diff > threshold), b_diff > threshold)
    printProgressBar(0, img.shape[0], prefix='Progress:', suffix='Complete', length=50)

    # Replace outlier pixels with the median pixel value of the surrounding pixels
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i, j]:
                # Compute the median pixel value of the surrounding pixels
                l_median = np.median(l[max(i-1, 0):min(i+2, img.shape[0]), max(j-1, 0):min(j+2, img.shape[1])])
                a_median = np.median(a[max(i-1, 0):min(i+2, img.shape[0]), max(j-1, 0):min(j+2, img.shape[1])])
                b_median = np.median(b[max(i-1, 0):min(i+2, img.shape[0]), max(j-1, 0):min(j+2, img.shape[1])])
                
                # Replace the outlier pixel with the median pixel value of the surrounding pixels
                l[i, j] = l_median
                a[i, j] = a_median
                b[i, j] = b_median
        printProgressBar(i+1, img.shape[0], prefix='Progress:', suffix='Complete', length=50)

    # Merge the LAB channels
    lab = cv2.merge((l, a, b))

    # Convert the LAB image back to RGB color space
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


