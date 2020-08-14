from keras.layers import Input, Dense
from keras.models import Model
import json
import numpy as np
import os

fn = "annotations/distribution-noppl-multi.json"
edgeDim = 80
if "-noppl-" in fn:
	edgeDim = 79
	
for i in range(15, 26):
	encodedDim = i

	input_img = Input(shape=(edgeDim,))
	encoded = Dense(encodedDim, activation='relu')(input_img)
	decoded = Dense(edgeDim, activation='sigmoid')(encoded)
	autoencoder = Model(input_img, decoded)
	encoder = Model(input_img, encoded)
	encoded_input = Input(shape=(encodedDim,))
	decoder_layer = autoencoder.layers[-1]
	decoder = Model(encoded_input, decoder_layer(encoded_input))
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	data = []
	test = []
	train = []
	with open(fn) as f:
		data = json.load(f)

	ratio = 5
	for i in range(len(data)):
		if i % ratio == 0:
			test.append(data[i])
		else:
			train.append(data[i])

	testDataset = np.array(test, dtype=np.float32)
	trainDataset = np.array(train, dtype=np.float32)

	autoencoder.fit(trainDataset, trainDataset, epochs=150, batch_size=256, shuffle=True, validation_data=(testDataset, testDataset))
	try:
		os.mkdir("models")
	except:
		pass
	autoencoder.save("models/autoencoder-" + str(edgeDim) + "-" + str(encodedDim) + ".h5")
