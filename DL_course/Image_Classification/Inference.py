import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import glob
import numpy as np

# Caution :: You have to classify your directories. Keep in mind!!
image_paths = sorted(glob.glob('PreProcessed/*.jpg'))

model_path = 'mnist_99acc_model.h5'
model = load_model(model_path)

print("\n\nIn PreProcessed folder, there are files :\n", image_paths, "\n\n")

for image_path in image_paths:
    img = cv2.imread(image_path, 0)
    test_img = img.reshape((1, 28, 28, 1))
    print(f"From {image_path}, We predicted : ", np.argmax(model.predict(test_img), axis=1), ".\n")
