import tensorflow.keras
import tensorflow as tf
import time
import numpy as np
import cv2

label_name = "labels.txt"
count = 0 
labels = []
with open(label_name, "r") as file:
	lines = file.readlines()
	for line in lines:
		_, label = line.split(' ')
		labels.append(label.replace('\n', ''))

print(f"There are in {label_name} : {labels}")

cap = cv2.VideoCapture(0)
model = tensorflow.keras.models.load_model('keras_model.h5')
model.summary()
start_time = time.time()
while True:
	ret, frame = cap.read()
	count +=1

	frame = cv2.resize(frame, (224, 224), cv2.INTER_AREA)
	data = tf.expand_dims(tf.convert_to_tensor(frame, dtype=tf.float32), 0)
	cv2.imshow('test', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
	if count % 10 == 0:
		count = 0		
		print("Predicted : ", labels[np.argmax(model.predict(data))], "\tFPS : ", int(1 / (time.time() - start_time)))
	start_time = time.time()

cap.release()
cv2.destroyAllWindows()

