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

## Experiments

To reproduce the experiments from our paper follow the instructions below.

### Experiment 1 - Injecting objects into scenes

To randomly add uncommon objects into the scene that weren't there and evaluate how often they are detected as anomalous:
`python3 exp1-pos-anom.py`

And to find out how often they are detected as the most anomalous object in the scene:
`python3 exp1-pos-most-anom.py`

To randomly remove an object from the scene and evaluate how strongly it was missing run:
`python3 exp1-neg-anom.py`

And to find out how often it was the most missing object from the scene:
`python3 exp1-neg-most-anom.py`

### Experiment 2 - How often the autoencoder agrees with YOLO that an object is not annotated

Download yolo weights:
`cd yolo;wget https://pjreddie.com/media/files/yolov3.weights;cd ../`

Download the images for COCO:
`python3 exp2-download.py`
