from PIL import Image
import numpy
import cv2
import math

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



def imagePixelManipulate(img, threshold, shadow_threshold=50, shadow_boost_factor=30):
    image = img.copy()
    data = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    height, width, channels = image.shape
    printProgressBar(0, height, prefix='Progress:', suffix='Complete', length=50)
    visited = numpy.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            if not visited[y, x]:
                pixelValueMean = numpy.round(numpy.median(data[y, x], axis=0)).astype(numpy.uint8)
                
                # Boost pixel values if they are below a certain threshold (for shadows)
                if numpy.sum(data[y,x,0]) < shadow_threshold:
                    data[y, x, 0] += shadow_boost_factor
                    data[y, x, 1] += shadow_boost_factor*0.1
                    data[y, x, 2] += shadow_boost_factor*0.1

                queue = [(y, x)]
                visited[y, x] = True
                while queue:
                    curr_y, curr_x = queue.pop(0)
                    if numpy.sum(numpy.abs(data[curr_y, curr_x] - pixelValueMean)) <= threshold:
                        data[curr_y, curr_x] = pixelValueMean
                        for ny, nx in [(curr_y-1, curr_x), (curr_y+1, curr_x), (curr_y, curr_x-1), (curr_y, curr_x+1)]:
                            if ny >= 0 and ny < height and nx >= 0 and nx < width and not visited[ny, nx]:
                                queue.append((ny, nx))
                                visited[ny, nx] = True
        printProgressBar(y + 1, height, prefix='Progress:', suffix='Complete', length=50)

    return cv2.cvtColor(data, cv2.COLOR_LAB2BGR)



# scale_percent = 40 # percent of original size
# width = int(image.shape[1] * scale_percent / 100)
# height = int(image.shape[0] * scale_percent / 100)
# dim = (width, height)
  
# # resize image
# resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
# new_img_size = (resized.shape[1] - (resized.shape[1] % 32), resized.shape[0] - (resized.shape[0] % 32))
# resized_img = cv2.resize(resized, new_img_size)
# lab = cv2.cvtColor(resized_img, cv2.COLOR_BGR2LAB)

# # Split the LAB channels
# l, a, b = cv2.split(lab)

# # Create a CLAHE object and apply it to the L channel
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# l_clahe = clahe.apply(l)

# # Merge the CLAHE-adjusted L channel with the original A and B channels
# lab_clahe = cv2.merge((l_clahe, a, b))

# # Convert the LAB image back to RGB color space
# rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# resized_img = imagePixelManipulate(rgb_clahe,200 , shadow_threshold=100, shadow_boost_factor=60)

# cv2.imshow('image window', resized_img)
# # add wait key. window waits until user presses a key
# cv2.waitKey(0)
# # and finally destroy/close all open windows
# cv2.destroyAllWindows()