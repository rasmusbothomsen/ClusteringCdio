import os
import shutil

import cv2

imageDir = r"C:\Users\Rasmus\Desktop\outputimageTest"




for fileName in os.listdir("train_images"):
    counter = 0
    img = cv2.imread(f"train_images/{fileName}")
    img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(f"train_images/{counter}{fileName}",img)
    counter +=1
    img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(f"train_images/{counter}{fileName}",img)
    counter +=1
    img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(f"train_images/{counter}{fileName}",img)
    counter +=1

