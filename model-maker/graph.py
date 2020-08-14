from matplotlib import pyplot as plt
import pickle
import os

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot()
ax.set_title('Model Loss ')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')

act = "relu"
act = "sig"

files = []
for i in os.walk("."):
	for fn in i[2]:
		if ".pickle" in fn and act in fn and not ("-r." in fn):
			files.append(fn)
	break
files.sort()
files = filter(lambda x: 15 <= int(x[5:-7].split('-')[0]) <= 25, files)

pFiles = []
for file in files:
	pFiles.append(file)
	history = []
	with open(file, "rb") as f:
		history = pickle.load(f)

	# plt.plot(history['loss'])
	ax.plot(history['val_loss'])
	
leg = map(lambda x: x[5:-7].split('-'), pFiles)
#leg = map(lambda x: x[0] + "-" + x[2], leg)
leg = map(lambda x: x[0], leg)
ax.legend(list(leg), loc='upper right')
plt.savefig("training_loss.eps")
plt.show()
