from matplotlib import pyplot as plt
import pickle
import os

plt.title('Model Loss ')
plt.ylabel('Loss')
plt.xlabel('Epoch')

ignoreActivations = []
useRegularisedVersion = False

files = []
for i in os.walk("../trainedModels"):
	for fn in i[2]:
		if ".pickle" not in fn:
			continue
		if useRegularisedVersion == True and "-reg." not in fn:
			continue
		activation = fn.split("-")[3]
		if activation in ignoreActivations:
			continue
		files.append(fn)
	break
files.sort()

processedFiles = []
for file in files:
	processedFiles.append(file)
	history = []
	with open(os.path.join("..", "trainedModels", file), "rb") as f:
		history = pickle.load(f)

	# plt.plot(history['loss'])
	plt.plot(history['val_loss'])

plt.legend(processedFiles, loc='upper right')
plt.show()
