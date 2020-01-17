from matplotlib import pyplot as plt
import pickle
import os

plt.title('Model Loss Difference ')
plt.ylabel('Loss Difference relative to n+1')
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

	fn2 = file.split("-")
	fn2[1] = str(n+1)
	fn2 = "-".join(fn2)
	history2 = []
	try:
		with open(os.path.join("..", "trainedModels", fn2), "rb") as f:
			history2 = pickle.load(f)
	except:
		continue

	diff = history2["val_loss"][-1] - history["val_loss"][-1]
	diff = abs(diff)

	points.append((n, diff))

points.sort()
points = list(zip(*points))

plt.scatter(points[0], points[1])
plt.plot(points[0], points[1])
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,0.02))
plt.show()
