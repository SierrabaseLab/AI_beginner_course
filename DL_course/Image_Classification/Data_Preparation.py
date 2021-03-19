import cv2
import glob
import numpy as np
import os
import shutil

image_paths = sorted(glob.glob('*.jpg'))
print("There are ",len(image_paths), "images")

if not(os.path.isdir('PreProcessed')):
    os.mkdir('PreProcessed')
    print("PreProcessed folder made")
else:
    shutil.rmtree('PreProcessed')
    os.mkdir('PreProcessed')
    print("PreProcessed folder remade")

for i, image_path in enumerate(image_paths):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    th = 255 - img
    cv2.imwrite(f'PreProcessed/Preprocessed_{i}.jpg', th)

