###experiment 1 - injecting anomalous positive classes
from keras.models import load_model
import json
import numpy as np
import random

#20 least common object ids in coco
idsToAdd = [22, 88, 79, 23, 86, 24, 21, 35, 89, 19, 13, 41, 4, 20, 33, 59, 57, 34, 11, 37]

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

for i in range(len(idsToAdd)):
	idsToAdd[i] = catmap[idsToAdd[i]]

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

	changedVals = []
	for i in range(anomsPerImage):
		changed = False
		while changed == False:
			anom = random.randint(0, len(idsToAdd) - 1)
			if frame[1][anom] == 0:
				changedVals.append(anom)
				frame[1][anom] = 1
				changed = True

	inputs = np.array([frame[1]], dtype=np.float32)
	predictions = autoencoder.predict(inputs)

	zipped = list(zip(inputs[0], predictions[0], cats))

	#62% of the time, strongest pos anom is correct
	#51% of the time, strongest outright anom is correct
	poppedIdxs = []
	for i in changedVals:
		maxDiff = 0
		maxDiffIdx = -1
		for item in range(len(zipped)):
			if item in poppedIdxs:
				continue
			diff = zipped[item][0] - zipped[item][1]
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
