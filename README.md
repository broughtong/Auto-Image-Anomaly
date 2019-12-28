# Auto Image Anomaly Detector

Looks at objects in images, and spots anomalies.

Seen a laptop, keyboard, table, surfboard? Surfboard is unexpected, mouse was expected but not seen.

Works by training an autoencoder against the class categories in a dataset.
You can then run the autoencoder against eg. Yolo for real time info.
It compares what the autoencoder says to the actual output. If it's entirely an 'expected' scene, they should be pretty similar.
If any classes are strongly different, they are probably anomalous (compared to the training data).

If you want to skip to the results based on coco, read `results.txt`

Else:

## Get dataset
`wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip`

`unzip annotations_trainval2017.zip`

## Prep the dataset
`python3 prep.py`

## Encode the dataset
`python3 datasetEncoder.py`

## Train autoencoder
`python3 auto.py`

If the val_loss keeps dropping, we're in business :D

## Prep validation
`python3 validate.py`

## Find anomalies
`python3 compare.py`

For examining results, take the image ids from above, and put them in the search box on here:
`http://cocodataset.org/#explore`

Remember, the actual images themselves are irrelevant, compare to the categories of annotations

