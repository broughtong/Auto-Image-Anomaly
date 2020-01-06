import numpy as np
import matplotlib.pyplot as plt
import pickle

data = []
with open('exp2-results-05.pickle', 'rb') as f:
	data = pickle.load(f)

#yolo
yolo = data[0]
x = []
y = []
for i in yolo:
	#data format: (threshold, yoloCorrect, yoloTotal, yoloConfmat)
	x.append(i[0])
	accuracy = float(i[1] / i[2])
	y.append(accuracy)
plt.plot(x, y, '-x', label='YOLO Only')

#yolo w/ ae
ae = data[1]
x = []
y = []
for i in ae:
	#data format: (threshold, yoloCorrect, yoloTotal, yoloConfmat)
	x.append(i[0])
	accuracy = float(i[1] / i[2])
	y.append(accuracy)

plt.plot(x, y, '-x', label='YOLO w/ Autoencoder')

plt.axvline(x=0.25, color='red', linestyle='--', label="YOLO default threshold")
plt.axvline(x=0.5, color='blue', linestyle='--', label="Autoencoder cut-off")
plt.legend(loc="lower right")
plt.xlabel('Threshold', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.savefig('Experiment2.jpg')
plt.show()
