import json

fn = "annotations/instances_train2017.json"

data = []

with open(fn, "r") as f:
	data = json.load(f)

data = data["annotations"]

for i in data:
	del i["segmentation"]
	del i["area"]

organised = {}

for i in data:
	if i["image_id"] in organised:
		organised[i["image_id"]].append(i)
	else:
		organised[i["image_id"]] = [i]

with open("annotations/instances_train2017.grouped.json", "w") as f:
	json.dump(organised, f)