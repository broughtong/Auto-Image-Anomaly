from keras.models import load_model
import json
import numpy as np

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

for frame in data:

	inputs = np.array([frame[1]], dtype=np.float32)
	predictions = autoencoder.predict(inputs)

	zipped = list(zip(inputs[0], predictions[0], cats))

	first = True
	for i in zipped:
		if i[0] == 0 and i[1] > 0.9:
			if first:
				print("=== Image %s ===" % (frame[0]))
				first = False
			print("\tExpected a %s (%f)" % (i[2], i[1]))
		elif i[0] == 1 and i[1] < 0.001:
			if first:
				print("=== Image %s ===" % (frame[0]))
				first = False
			print("\tUnexpected %s (%f)" % (i[2], i[1]))