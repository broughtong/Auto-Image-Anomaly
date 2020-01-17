#lock seed for nn weight init, so results are reproducible
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

import os
import pickle
import random
import numpy as np
from keras.layers import Input, Dense, Concatenate
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras import regularizers
import numpy as np

encoders = [80, 70, 60, 50, 40, 30, 25, 20, 15, 10, 8, 5, 4, 3, 2, 1]
encoders = range(1, 40) + [40, 50, 60, 70, 80]

activationFunction = "relu"
activationFunction = "sigmoid"
regulariser = False
ratio = 10
epochs = 100

history = []

for encodeN in encoders:

	encoding_dim = encodeN

	input_img = Input(shape=(80,))

	if regulariser == False:
		encoded = Dense(encoding_dim, activation=activationFunction)(input_img)
	else:
		encoded = Dense(encoding_dim, activation=activationFunction, activity_regularizer=regularizers.l1(10e-5))(input_img)

	decoded = Dense(80, activation=activationFunction)(encoded)

	autoencoder = Model(input_img, decoded)
	encoder = Model(input_img, encoded)

	encoded_input = Input(shape=(encoding_dim,))
	decoder_layer = autoencoder.layers[-1]
	decoder = Model(encoded_input, decoder_layer(encoded_input))

	autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

	data = []
	try:
		with open("list_of_categories_by_image.pickle", "rb") as f:
			data = pickle.load(f)
	except:
		print("Error opening coco data. Make sure your running python3, and have run the dataset collecter first")
		break

	indexmap = {}
	with open("indexmap", "rb") as f:
		indexmap = pickle.load(f)

	for key in indexmap:
		indexmap[key] = 0

	lTrainSet = 0
	lTestSet = 0
	ind = 0
	for _ in data:
		if ind % ratio == 0:
			lTestSet += 1
		else:
			lTrainSet += 1
		ind += 1

	trainSet = np.zeros((lTrainSet, 80), dtype=np.float32)
	testSet = np.zeros((lTestSet, 80), dtype=np.float32)

	print(trainSet.shape)
	print(testSet.shape)

	currentTrainIdx = 0
	currentTestIdx = 0

	for idx in range(len(data)):
		observations = indexmap.copy()
		for detection in data[idx]:
			classLabel = detection
			if classLabel == "sofa":
				classLabel = "couch"
			if classLabel == "diningtable":
				classLabel = "dining table"
			if classLabel == "aeroplane":
				classLabel = "airplane"
			if classLabel == "tvmonitor":
				classLabel = "tv"
			if classLabel == "motorbike":
				classLabel = "motorcycle"
			if classLabel == "pottedplant":
				classLabel = "potted plant"
			observations[classLabel] = 1.0

		ind = 0
		for key in observations:
			if idx % ratio == 0:
				# print(currentTestIdx)
				testSet[currentTestIdx][ind] = observations[key]
				# currentTestIdx += 1
			else:
				# print(currentTrainIdx)
				trainSet[currentTrainIdx][ind] = observations[key]
				# currentTrainIdx += 1
			ind += 1

		if idx % ratio == 0:
			currentTestIdx += 1
		else:
			currentTrainIdx += 1

	history = autoencoder.fit(trainSet, trainSet,
		epochs=epochs,
		batch_size=256,
		shuffle=True,
		validation_data=(testSet, testSet))

	regString = ""
	if regulariser == True:
		regString = "-reg"

	with open('../trainedModels/history-' + str(encoding_dim) + "-" + str(epochs) + "-" + activationFunction + regString + ".pickle", 'wb') as fp:
		pickle.dump(history.history, fp)
	autoencoder.save('../trainedModels/autoencoder-' + str(encoding_dim) + "-" + str(epochs) + "-" + activationFunction + regString + '.h5')
	encoder.save('../trainedModels/encoder-' + str(encoding_dim) + "-" + str(epochs) + "-" + activationFunction + regString + '.h5')
	decoder.save('../trainedModels/decoder-' + str(encoding_dim) + "-" + str(epochs) + "-" + activationFunction + regString + '.h5')
