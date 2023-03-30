from PIL import Image
import numpy
import cv2

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

image = cv2.imread("testImage2.jpg")

data = numpy.asarray(image)
def RGBImage(image):
    data = image
    blue = [0,0,255]
    red = [0,255,0]
    green = [255,0,0]
    colorList = [blue,red,green]
    height, width, channels = image.shape
    printProgressBar(0, height, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for y in range(height):
        for x in range(width):
            pixel = data[y][x]
            data[y][x] = colorList[numpy.argmin(pixel)]
        printProgressBar(y + 1, height, prefix = 'Progress:', suffix = 'Complete', length = 50)
    return data



image = data
scale_percent = 20 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
new_img_size = (resized.shape[1] - (resized.shape[1] % 32), resized.shape[0] - (resized.shape[0] % 32))
resized_img = cv2.resize(resized, new_img_size)
resized_img = RGBImage(resized_img)

cv2.imshow('image window', resized_img)
# add wait key. window waits until user presses a key
cv2.waitKey(0)
# and finally destroy/close all open windows
cv2.destroyAllWindows()