import json

includePeople = False
rejectSingle = True

size = 80
if includePeople == False:
	size = 79

#maps index from coco to contiguous
categories = []
with open("annotations/instances_val2017.json", "r") as f:
	categories = json.load(f)
categories = categories["categories"]
categories = [x["id"] for x in categories]
categories = dict(zip(categories, range(size+1)))

data = []

with open("annotations/instances_train2017.grouped.json", "r") as f:
	data = json.load(f)

output = []
for i in data.keys():

	i = data[i]
	dist = [0] * size

	for j in i:
		if includePeople:
			dist[categories[j["category_id"]]] = 1
		else:
			if j["category_id"] != 1:
				dist[categories[j["category_id"]]-1] = 1

	nObjects = 0
	for j in dist:
		if j == 1:
			nObjects += 1

	if rejectSingle == True:
		if nObjects > 1:
			output.append(dist)
	else:
		output.append(dist)

p = "ppl"
if includePeople == False:
	p = "noppl"

m = "all"
if rejectSingle == True:
	m = "multi"

with open("annotations/distribution-" + p + "-" + m + ".json", "w") as f:
	json.dump(output, f)