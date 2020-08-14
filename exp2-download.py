import os
import json
import threading
import time
import requests

data = []
with open("annotations/instances_val2017.json", "r") as f:
	data = json.load(f)

try:
	os.mkdir("eval-images")
except FileExistsError:
	pass

def getter(url1, url2, dest):
	try:
		r = requests.get(url1, allow_redirects=True)
		open(dest, 'wb').write(r.content)
	except:
		try:
			r = requests.get(url2, allow_redirects=True)
			open(dest, 'wb').write(r.content)
		except:
			pass

threads = []
for i in data["images"]:
	t = threading.Thread(target=getter, args=(i["flickr_url"], i["coco_url"], os.path.join("eval-images", i["file_name"])))
	t.start()
	time.sleep(0.01)
	threads.append(t)

map(lambda t: t.join(), threads)
