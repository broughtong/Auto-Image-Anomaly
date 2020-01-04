###experiment 1 - injecting anomalous positive classes
from keras.models import load_model
import json
import numpy as np
import random

#20 least common object ids in coco
# idsToAdd = [22, 88, 79, 23, 86, 24, 21, 35, 89, 19, 13, 41, 4, 20, 33, 59, 57, 34, 11, 37]
idsToAdd = [22, 88, 79, 23, 86, 24, 21, 35, 89, 19, 13, 41, 4, 20, 33, 59, 57, 34, 11, 37, 77, 18, 40, 55, 87, 10, 15, 52, 54, 39, 51, 56, 8, 38, 6, 73, 32, 67, 81, 53, 31, 75, 44, 78, 85, 42, 60, 3, 84, 58, 64, 74, 49, 27, 47, 16, 5, 17, 36, 9, 80, 63, 76, 72, 48, 62, 14, 70, 25, 7, 82, 28, 50, 43, 46, 65, 2, 61]

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
print(catmap)

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

changedVals = []
anom = random.randint(0, len(idsToAdd) - 1)
changedVals.append(anom)

for frame in data:

	inputs = np.array([frame[1]], dtype=np.float32)
	predictions = autoencoder.predict(inputs)

	anomImage = False
	for i in range(0, len(inputs[0])):
		if abs(inputs[0][i] - predictions[0][i]) > 0.2:
			anomImage = True
			break
	if anomImage == True:
		continue

	if frame[1][anom] == 1:
		continue
	else:
		frame[1][anom] = 1

	# changedVals = []
	# for i in range(anomsPerImage):
	# 	changed = False
	# 	while changed == False:
	# 		anom = random.randint(0, len(idsToAdd) - 1)
	# 		if frame[1][anom] == 0:
	# 			changedVals.append(anom)
	# 			frame[1][anom] = 1
	# 			changed = True

	inputs = np.array([frame[1]], dtype=np.float32)
	predictions = autoencoder.predict(inputs)

	zipped = list(zip(inputs[0], predictions[0], cats))

	correct = 0
	for i in changedVals:
		#at 0.5 (more likely than not) ~ 71% accuracy
		if zipped[i][1] < 0.5:
		#at 0.2 (very sure of the anomaly) ~ 61% acc
		# if zipped[i][1] < 0.2:
			correct += 1

	n += 1
	total += correct

	print("Correct: " + str(correct) + "/" + str(anomsPerImage))

print("Avg: " + str(total / float(n)) + "/" + str(anomsPerImage) + " or " + str((((total / float(n)) / anomsPerImage) * 100)) + "%")
print(n)
print(changedVals)