import pickle
from pycocotools.coco import COCO

data_dir = "./data/coco_annotations"
data_type = "val2014"
coco = COCO("{}/annotations/instances_{}.json".format(data_dir, data_type))
imgs = coco.getImgIds()
annotations_ids = [coco.getAnnIds(img) for img in imgs]
annotations = [coco.loadAnns(a) for a in annotations_ids]
categories_in_image = [list({coco.loadCats(a['category_id'])[0]['name'] for a in ans}) for ans in annotations]

with open("list_of_categories_by_image.pickle", "wb") as f:
	pickle.dump(categories_in_image, f)
