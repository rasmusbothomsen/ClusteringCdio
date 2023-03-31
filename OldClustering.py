import cv2
import numpy as np


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

# def imagePixelManipulate(img , threshold):
#     image = img
#     data = image
#     height, width, channels = image.shape
#     printProgressBar(0, height, prefix = 'Progress:', suffix = 'Complete', length = 50)
#     pixelList = numpy.array([data[0][0]])
#     indexList = [[0,0]]
#     for y in range(height):
#         for x in range(1,width):
#             pixelValueMean = numpy.round(numpy.mean(pixelList, axis=0)).astype((numpy.uint8))
#             tempMean = numpy.copy(pixelValueMean)
#             tempMean[0][1] = 100

#             pixel = data[y][x]
#             diff = abs(pixel-pixelValueMean)
#             if(numpy.sum(diff) <= threshold):
#                 pixelList = numpy.append(pixelList,pixel)
#                 indexList.append([y,x])
#             else:
#                 for pixelIndex in indexList:
#                     if(numpy.sum(pixelValueMean)>10):
#                         data[pixelIndex[0],pixelIndex[1]] = pixelValueMean
#                 pixelList = numpy.zeros((1,3))
#                 indexList = []
            
#         printProgressBar(y + 1, height, prefix = 'Progress:', suffix = 'Complete', length = 50)

#     return data




def imagePixelManipulate(img, threshold):
    image = img.copy()
    height, width, channels = image.shape
    visited = set()  # keep track of visited pixels
    pixelList = [image[0, 0]]  # start with the first pixel
    visited.add((0, 0))  # mark it as visited
    while pixelList:
        # calculate the convolution value of the current pixel list
        convolution_value = np.mean(pixelList, axis=0)
        # find new pixels to add to the list
        new_pixels = []
        for pixel in pixelList:
            # check all 4 adjacent pixels
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                y, x = pixel[0] + dy, pixel[1] + dx
                if y >= 0 and y < height and x >= 0 and x < width and (y, x) not in visited:
                    visited.add((y, x))  # mark the pixel as visited
                    diff = np.abs(image[y, x] - convolution_value)
                    if np.sum(diff) <= threshold:
                        new_pixels.append(image[y, x])
        # add the new pixels to the list
        if new_pixels:
            pixelList.extend(new_pixels)
        else:
            # set all pixels in the list to the convolution value
            for pixel in pixelList:
                image[pixel[0], pixel[1]] = convolution_value
            # start a new list with the next unvisited pixel
            pixelList = []
            for y in range(height):
                for x in range(width):
                    if (y, x) not in visited:
                        pixelList.append(image[y, x])
                        visited.add((y, x))
                        break
                if pixelList:
                    break
    return image

image = cv2.imread("testImage2.jpg")


scale_percent = 10 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
new_img_size = (resized.shape[1] - (resized.shape[1] % 32), resized.shape[0] - (resized.shape[0] % 32))
resized_img = cv2.resize(resized, new_img_size)


resized_img = imagePixelManipulate(resized_img,100)

cv2.imshow('image window', resized_img)
# add wait key. window waits until user presses a key
cv2.waitKey(0)
# and finally destroy/close all open windows
cv2.destroyAllWindows()