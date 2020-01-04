###experiment 1 - injecting anomalous positive classes
from keras.models import load_model
import json
import numpy as np
import random

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

cats = []
for i in categories["categories"]:
	cats.append(i["name"])
if "-noppl-" in valfn:
	del cats[0]

data = []
with open(valfn, "r") as f:
	data = json.load(f)

anomsPerImage = 1

total = 0
n = 0

for frame in data:

	inputs = np.array([frame[1]], dtype=np.float32)
	predictions = autoencoder.predict(inputs)

	nAnnotations = 0
	for i in inputs[0]:
		if i == 1:
			nAnnotations += 1

	if nAnnotations < 5:
		continue

	# anomImage = False
	# for i in range(0, len(inputs[0])):
	# 	if abs(inputs[0][i] - predictions[0][i]) > 0.5:
	# 		anomImage = True
	# 		break
	# if anomImage == True:
	# 	continue

	changedVals = []
	for i in range(anomsPerImage):
		changed = False
		while changed == False:
			anom = random.randint(0, size - 1)
			if frame[1][anom] == 1:
				changedVals.append(anom)
				frame[1][anom] = 0
				changed = True

	inputs = np.array([frame[1]], dtype=np.float32)
	predictions = autoencoder.predict(inputs)

	zipped = list(zip(inputs[0], predictions[0], cats))

	#16.5%
	poppedIdxs = []
	for i in changedVals:
		maxDiff = 0
		maxDiffIdx = -1
		for item in range(len(zipped)):
			if item in poppedIdxs:
				continue
			diff = -zipped[item][0] + zipped[item][1]
			# diff = abs(zipped[item][0] - zipped[item][1])
			if diff > maxDiff:
				maxDiff = diff
				maxDiffIdx = item
		poppedIdxs.append(maxDiffIdx)

	correct = 0
	for i in poppedIdxs:
		if i in changedVals:
			correct += 1

	n += 1
	total += correct

	print("Correct: " + str(correct) + "/" + str(anomsPerImage))

print("Avg: " + str(total / float(n)) + "/" + str(anomsPerImage) + " or " + str((((total / float(n)) / anomsPerImage) * 100)) + "%")
print(n)