# Fahad Alsehami
# Age Estimation from the Adult Pubic Symphysis project

# import the required  libraries for the feature extraction

from sklearn.preprocessing import LabelEncoder
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from settings import config
from imutils import paths
import numpy as np
import pickle
import random
import os

# Init the labels
# Load a specific network : Here will try VGG16, and in the version 1.01 will try different network for benchmar purposes
print("VGG16 processing now...")
model = VGG16(weights="imagenet", include_top=False)
labelEncoder = None

# define the loop method to handle the spllited data
for x in (config.TRAIN_FOLDER, config.TESTING_FOLDER, config.VALIDATION):

	# get image route in the split
	print("processing split...".format(x))
	r = os.path.sep.join([config.PRIMARY_ROUT, x])

	imageRout = list(paths.list_images(r))

	# randomly shuffle the image paths and then extract the class
	# labels from the file paths
	random.shuffle(imageRout)
	labels = [p.split(os.path.sep)[-2] for r in imageRout]
# in the second version will add augmented method and resampling method as well.
	# if the label encoder is None, create it
	if labelEncoder is None:
		labelEncoder = LabelEncoder()
		labelEncoder.fit(labels)

	# open the output CSV file for writing
	csvRout = os.path.sep.join([config.PARENT_CVS_ROUT,
		"{}.csv".format(x)])
	csv = open(csvRout, "w")

	# Define loop for batch
	for (b, i) in enumerate(range(0, len(imagePaths), config.BATCH_SIZE)):
		# extract the batch of images and labels, then initialize the
		# list of actual images that will be passed through the network
		# for feature extraction
		print("[processing batch {}/{}".format(b + 1,
			int(np.ceil(len(imagePaths) / float(config.BATCH_SIZE)))))
		batchRout = imagePaths[i:i + config.BATCH_SIZE]
		batchLabels = le.transform(labels[i:i + config.BATCH_SIZE])
		batchImages = []

		# loop over the images and labels in the current batch
		for imageRout in batchRout:

			image = load_img(imageRout, target_size=(224, 224))
			image = img_to_array(image)


			image = np.expand_dims(image, axis=0)
			image = imagenet_utils.preprocess_input(image)

			# add the image to the batch
			batchImages.append(image)


		batchImages = np.vstack(batchImages)
		features = model.predict(batchImages, batch_size=config.BATCH_SIZE)
		features = features.reshape((features.shape[0], 7 * 7 * 512))

		# loop method to extract label
		for (label, vec) in zip(batchLabels, features):
			# construct a row that exists of the class label and
			# extracted features
			vec = ",".join([str(v) for v in vec])
			csv.write("{},{}\n".format(label, vec))


	csv.close()

# setup a serlization for the label
f = open(config.MODEL_ROUT, "wb")
f.write(pickle.dumps(labelEncoder))
f.close()
