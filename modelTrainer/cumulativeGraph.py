from matplotlib import pyplot as plt
import pickle
import os

plt.title('Model Loss After Training')
plt.ylabel('Loss')
plt.xlabel('Encoding neurons')

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
points = []
for file in files:
	processedFiles.append(file)
	history = []
	with open(os.path.join("..", "trainedModels", file), "rb") as f:
		history = pickle.load(f)

	n = int(file.split("-")[1])

	diff = history["val_loss"][-1]
	diff = abs(diff)

	points.append((n, diff))

points.sort()
points = list(zip(*points))

plt.scatter(points[0], points[1])
plt.plot(points[0], points[1])
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,0.125))
plt.show()
