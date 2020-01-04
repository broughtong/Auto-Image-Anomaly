import cv2
import os
import numpy as np
from keras.models import load_model
import json

path = "eval-images"

valfn = "annotations/val-distribution-noppl-multi.json"
autoencoder = load_model('models/autoencoder-79-10.h5')

size = 80
if "-noppl-" in valfn:
	size = 79

#maps index from coco to contiguous
categories = []
with open("annotations/instances_val2017.json", "r") as f:
	categories = json.load(f)
catmap = categories["categories"]
catmap = [x["id"] for x in catmap]
catmap = dict(zip(catmap, range(size+1)))

annos = []
with open(valfn, "r") as f:
	annos = json.load(f)

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
	classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

files = []
for i in os.walk(path):
	files = i[2]

annos = dict(annos)

for imgfn in files:

	img = cv2.imread(os.path.join(path, imgfn))
	height, width, channels = img.shape

	blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)

	thresh = 0.1

	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > thresh:
				# Object detected
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)
				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)

	inputs = [0] * 80

	for i in range(len(class_ids)):
		inputs[class_ids[i]] = confidences[i]
	# 	print(classes[class_ids[i]], class_ids[i], confidences[i])
	# print(imgfn)

	if size == 79:
		del inputs[0]

	inputs = np.array([inputs], dtype=np.float32)
	predictions = autoencoder.predict(inputs)[0]

	zipped = list(zip(inputs[0], predictions, classes[1:]))

	try:
		imgfn = imgfn[:-4].lstrip("0")
		annotation = annos[int(imgfn)]
	except:
		continue

	for i in range(len(zipped)):
		if zipped[i][0] > 0.3 and zipped[i][0] < 0.5 and zipped[i][1] > 0.1:
			if annotation[i] != 1:
				print("Yolo failed, but boosted: ", zipped[i], imgfn)
			else:
				print("Maybe bad annotation: ", zipped[i], imgfn)
				
				font = cv2.FONT_HERSHEY_PLAIN
				for i in range(len(boxes)):
					x, y, w, h = boxes[i]
					label = str(classes[class_ids[i]])
					color = colors[i]
					cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
					cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
				cv2.imshow("Image", img)
				cv2.waitKey(0)
				cv2.destroyAllWindows()


	# print(boxes)
	# print(class_ids)
	# print(boxes)
	# print(imgfn)
	# for i in class_ids:
	# 	print(classes[i])

	# font = cv2.FONT_HERSHEY_PLAIN
	# for i in range(len(boxes)):
	# 	if i in indexes:
	# 		x, y, w, h = boxes[i]
	# 		label = str(classes[class_ids[i]])
	# 		color = colors[i]
	# 		cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
	# 		cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
	# cv2.imshow("Image", img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# break