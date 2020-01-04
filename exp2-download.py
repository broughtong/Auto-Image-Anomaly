import os
import json
import threading
from urllib.request import urlretrieve
import urllib

data = []
with open("annotations/instances_val2017.json", "r") as f:
	data = json.load(f)

try:
	os.mkdir("eval-images")
except FileExistsError:
	pass

def getter(url1, url2, dest):
	try:
		urlretrieve(url1, dest)
	except urllib.error.HTTPError:
		urlretrieve(url2, dest)

threads = []
for i in data["images"]:
	t = threading.Thread(target=getter, args=(i["flickr_url"], i["coco_url"], os.path.join("eval-images", i["file_name"])))
	t.start()
	threads.append(t)

map(lambda t: t.join(), threads)
