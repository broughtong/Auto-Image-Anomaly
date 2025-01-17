###experiment 1 - injecting anomalous positive classes
from keras.models import load_model
import json
import numpy as np
import random
import sys

valfn = "annotations/val-distribution-noppl-multi.json"

encDimsValues = list(range(5, 16))
autoencoders = [load_model('models/autoencoder-79-' + str(i) +'.h5') for i in encDimsValues]

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
#
# for frame in data:
#
# 	inputs = np.array([frame[1]], dtype=np.float32)
# 	predictions = autoencoder.predict(inputs)
#
# 	nAnnotations = 0
# 	for i in inputs[0]:
# 		if i == 1:
# 			nAnnotations += 1
#
# 	if nAnnotations < 5:
# 		continue
#
# 	anomImage = False
# 	for i in range(0, len(inputs[0])):
# 		if abs(inputs[0][i] - predictions[0][i]) > 0.2:
# 			anomImage = True
# 			break
# 	if anomImage == True:
# 		continue
#
# 	changedVals = []
# 	for i in range(anomsPerImage):
# 		changed = False
# 		while changed == False:
# 			anom = random.randint(0, size - 1)
# 			if frame[1][anom] == 1:
# 				changedVals.append(anom)
# 				frame[1][anom] = 0
# 				changed = True
#
# 	inputs = np.array([frame[1]], dtype=np.float32)
# 	predictions = autoencoder.predict(inputs)
#
# 	zipped = list(zip(inputs[0], predictions[0], cats))
#
# 	correct = 0
# 	for i in changedVals:
# 		#at 0.1 (some low prob of being there) ~ 13.5% accuracy
# 		# if zipped[i][1] > 0.1:
# 		#at 0.25 (low prob of being there) ~ 9.1% accuracy
# 		# if zipped[i][1] > 0.2:
# 		#at 0.5 (more likely than not) ~ 3.5% accuracy
# 		if zipped[i][1] > 0.5:
# 		#at 0.8 (very sure it should be there) ~ 1.3% acc
# 		# if zipped[i][1] > 0.8:
# 			correct += 1
#
# 	n += 1
# 	total += correct
#
# 	print("Correct: " + str(correct) + "/" + str(anomsPerImage))
#
# print("Avg: " + str(total / float(n)) + "/" + str(anomsPerImage) + " or " + str((((total / float(n)) / anomsPerImage) * 100)) + "%")
# print("N of images: " + str(n))




###
##
##  With statistics
##
###



anomsToTest = 5


changedVals = []
seed_max = 20
for seed in range(0, seed_max):
	totals = []
	ns = []
	random.seed(seed)
	# anom = random.randint(0, len(idsToAdd) - 1)
	# anoms = np.random.choice(list(range(0, len(idsToAdd) - 1)), anomsToTest)
	# print(anoms)
	# changedVals.append(anom)
	print("# seed {} out of {}".format(seed, seed_max))
	print("# seed {} out of {}".format(seed, seed_max), file=sys.stderr)

	for autoencoder in autoencoders:
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

			anomImage = False
			for i in range(0, len(inputs[0])):
				if abs(inputs[0][i] - predictions[0][i]) > 0.5:
					anomImage = True
					break
			if anomImage == True:
				continue
			
			anoms = np.random.choice(np.argwhere(np.array(frame[1]) == 1)[:, 0], anomsToTest, replace=False)
			#print(anoms)
			for a in anoms:
				# if frame[1][anom] == 1:
				#	 continue
				# else:
				#	 frame[1][anom] = 1
				#if frame[1][a] == 1:
				#	continue
				#else:
				frame[1][a] = 0

				#changedVals = []
				#for i in range(anomsPerImage):
				#	changed = False
				#	while changed == False:
				#		anom = random.randint(0, len(idsToAdd) - 1)
				#		if frame[1][anom] == 1:
				#			changedVals.append(anom)
				#			frame[1][anom] = 0
				#			changed = True

				inputs = np.array([frame[1]], dtype=np.float32)
				predictions = autoencoder.predict(inputs)

				zipped = list(zip(inputs[0], predictions[0], cats))

				correct = 0
				# for i in changedVals:
				# 	#at 0.1 (some low prob of being there) ~ 13.5% accuracy
				# 	# if zipped[i][1] > 0.1:
				# 	#at 0.25 (low prob of being there) ~ 9.1% accuracy
				# 	# if zipped[i][1] > 0.2:
				# 	#at 0.5 (more likely than not) ~ 3.5% accuracy
				# 	if zipped[i][1] > 0.5:
				# 	#at 0.8 (very sure it should be there) ~ 1.3% acc
				# 	# if zipped[i][1] > 0.8:
				# 		correct += 1
				#at 0.1 (some low prob of being there) ~ 13.5% accuracy
				#if zipped[a][1] > 0.1:
				#at 0.25 (low prob of being there) ~ 9.1% accuracy
				if zipped[a][1] > 0.25:
				#at 0.5 (more likely than not) ~ 3.5% accuracy
				#if zipped[a][1] > 0.5:
				#at 0.8 (very sure it should be there) ~ 1.3% acc
				# if zipped[a][1] > 0.8:
					correct += 1

				n += 1
				total += correct
				
				frame[1][a] = 1

				#print("Correct: " + str(correct) + "/" + str(anomsPerImage))
		totals.append(total)
		ns.append(n)

	for i in range(len(encDimsValues)):
		#print("Avg: " + str(total / float(n)) + "/" + str(anomsPerImage) + " or " + str((((total / float(n)) / anomsPerImage) * 100)) + "%")
		try:
			acc = (((totals[i] / float(ns[i])) / anomsPerImage) * 100)
			print("==" + str(encDimsValues[i]) + "== " + str(acc) + "% accuracy over " + str(ns[i]) + " images")
			sys.stdout.flush()
		except ZeroDivisionError:
			print("==" + str(encDimsValues[i]) + "== " + str(ns[i]) + " images")
			sys.stdout.flush()
		#print(n)
	#print(changedVals)
