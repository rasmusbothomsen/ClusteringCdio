import os
import shutil

imageDir = r"C:\Users\Rasmus\Desktop\outputimageTest"



os.makedirs("dataDir", exist_ok=True)

for fileName in os.listdir(imageDir):
    shutil.copy(f"train_images/{fileName}", f"dataDir/{fileName}")
    shutil.copy(f"yolo_labels/{fileName[:-3]}txt", f"dataDir/{fileName[:-3]}txt")

