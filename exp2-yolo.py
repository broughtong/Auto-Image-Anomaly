import cv2
import os
import numpy as np
from keras.models import load_model
import json
import pickle
import darknet

#important!
#above this threshold is taken as the truth for the ae
aeMaxTreshold = 0.5
#and this for what defines assoc
aeInfluence = 0.1

#get classes
classes = []
with open("yolo/coco.names", "r") as f:
	classes = [line.strip() for line in f.readlines()]
#remove people from classes
del classes[0]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#load autoencoder
autoencoder = load_model('models/autoencoder-79-10.h5')

#load annotations
annotations = []
with open("annotations/val-distribution-noppl-multi.json", "r") as f:
	annotations = dict(json.load(f))

#get evaluation image files
files = []
for i in os.walk("eval-images"):
	files = i[2]

#create yolo network
net = darknet.load_net("yolo/yolov3.cfg", "yolo/yolov3.weights", 0)
meta = darknet.load_meta("yolo/coco.data")

yoloResults = []
aeResults = []

#perform experiment for range of thresholds
#more detail at lower thresholds, less especially above 0.5
# thresholds = np.concatenate([np.array([0.000001, 0.0001]), np.arange(0.01, 0.05, 0.01), np.arange(0.05, 0.5, 0.05), np.arange(0.5, 1, 0.1)])
thresholds = np.concatenate([np.arange(0.05, 0.5, 0.05), np.array([0.5, 0.9])])
#thresholds = np.arange(0.05, 1.00, 0.05)
for threshold in thresholds:

	#build accuracy and confusionMatrix for this threshold
	yoloConfmat = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
	yoloCorrect = 0
	yoloTotal = 0

	aeConfmat = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
	aeCorrect = 0
	aeTotal = 0

	imgN = 0

	for imagefn in files:

		#open coco annotation for the image
		#it may not exist as the annotations file skips images with low number of detections etc
		imgid = ""
		annotation = []
		try:
			imgid = imagefn[:-4].lstrip("0")
			annotation = annotations[int(imgid)]
		except:
			continue

		nAnnotations = 0
		for i in annotation:
			if i == 1:
				nAnnotations += 1

		if nAnnotations < 3:
			continue

		#open image
		img = cv2.imread(os.path.join("eval-images", imagefn))
		height, width, channels = img.shape

		#yolo detection
		results = darknet.detect(net, meta, img, thresh=threshold)
		print(results)

		#gather detections together that are above the threshold
		class_ids = []
		confidences = []
		boxes = []

		coconames = []
		cocomap = {}
		with open("yolo/coco.names", "r") as f:
			coconames = f.read()
		coconames = coconames.split("\n")
		coconames = filter(None, coconames)
		del coconames[0]
		for i in range(len(coconames)):
			cocomap[coconames[i]]  = i
		print(cocomap)

		for result in results:
			if result[0] == "person":
				continue
			boxes.append(result[2])
			confidences.append(result[1])
			class_ids.append(cocomap[result[0]])
		classList = [classes[x] for x in class_ids]

		uniqueIDs = list(set(class_ids))

		#mark correct ids
		for classID in uniqueIDs:
			if annotation[classID] == 1:
				yoloCorrect += 1
			yoloTotal += 1

		#build array for confusion matrix and autoencoder
		inputs = [0] * len(classes)

		for i in range(len(class_ids)):
			inputs[class_ids[i]] = max(inputs[class_ids[i]], confidences[i])

		#fill in yolo only confusion matrix
		for idx in range(len(classes)):
			if inputs[idx] != 0:
				#yolo predicted the object
				if annotation[idx] == 0:
					#but it wasn't there
					yoloConfmat["fp"] += 1
				else:
					#and it was in the annotation
					yoloConfmat["tp"] += 1
			else:
				#yolo didn't see the object
				if annotation[idx] == 0:
					#and it wasn't there
					yoloConfmat["tn"] += 1
				else:
					#yolo missed it
					yoloConfmat["fn"] += 1

		#feed data into autoencoder
		inputs = np.array([inputs], dtype=np.float32)
		predictions = autoencoder.predict(inputs)[0]

		uniqueIDs = []

		#mark any results above ae cutoff the same as yolo
		#else, only use the yolo results if ae also present
		for idx in range(len(predictions)):
			if inputs[0][idx] > aeMaxTreshold:
				uniqueIDs.append(idx)
			elif inputs[0][idx] > threshold:
				if predictions[idx] > aeInfluence:
					uniqueIDs.append(idx)

		#mark correct ids
		for classID in uniqueIDs:
			if annotation[classID] == 1:
				aeCorrect += 1
			aeTotal += 1

		for idx in range(len(classes)):
			if idx in uniqueIDs:
				#ae predicted the object
				if annotation[idx] == 0:
					#but it wasn't there
					aeConfmat["fp"] += 1
				else:
					#and it was in the annotation
					aeConfmat["tp"] += 1
			else:
				#ae didn't see the object
				if annotation[idx] == 0:
					#and it wasn't there
					aeConfmat["tn"] += 1
				else:
					#yolo missed it
					aeConfmat["fn"] += 1

		# print(imgid)
		# print(annotation)
		# print(class_ids)
		# print(classList)
		# print(confidences)
		# print(boxes)

		imgN += 1

		if imgN % 20 == 0:
			print(imgN)

		if imgN == 3:
			break

	yoloResults.append((threshold, yoloCorrect, yoloTotal, yoloConfmat))
	aeResults.append((threshold, aeCorrect, aeTotal, aeConfmat))

	print("======")
	print(threshold)
	print(yoloCorrect, yoloTotal)
	print(yoloConfmat)
	print(aeCorrect, aeTotal)
	print(aeConfmat)

	with open('exp2-results-' + str(aeInfluence) + '.pickle', 'wb') as f:
		pickle.dump([yoloResults, aeResults], f)
