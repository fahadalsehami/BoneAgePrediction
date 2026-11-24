# Fahad Alsehami
# Age Estimation from the Adult Pubic Symphysis project

# import the required  libraries for the feature extraction

from sklearn.preprocessing import LabelEncoder
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import img_to_array, load_img
from settings import config
from settings.files import list_images
import numpy as np
import pickle
import random
import os

# Init the labels
# Load a specific network : Here will try VGG16, and in the version 1.01 will try different network for benchmar purposes
print("VGG16 processing now...")
model = VGG16(weights="imagenet", include_top=False)
labelEncoder = None
os.makedirs(config.PARENT_CVS_ROUT, exist_ok=True)

# define the loop method to handle the spllited data
for x in (config.TRAIN_FOLDER, config.TESTING_FOLDER, config.VALIDATION):

	# get image route in the split
	print("processing '{}' split...".format(x))
	r = os.path.sep.join([config.PRIMARY_ROUT, x])

	imageRout = list(list_images(r))

	# randomly shuffle the image paths and then extract the class
	# labels from the file paths
	random.shuffle(imageRout)
	labels = [p.split(os.path.sep)[-2] for p in imageRout]
# in the second version will add augmented method and resampling method as well.
	# if the label encoder is None, create it from the class names in the config,
	# so every split is encoded the same way even when a class is missing from one split
	if labelEncoder is None:
		labelEncoder = LabelEncoder()
		labelEncoder.fit(config.CLASSES)

	# open the output CSV file for writing
	csvRout = os.path.sep.join([config.PARENT_CVS_ROUT,
		"{}.csv".format(x)])
	with open(csvRout, "w") as csv:

		# Define loop for batch
		batchCount = int(np.ceil(len(imageRout) / float(config.BATCH_SIZE)))
		for (b, i) in enumerate(range(0, len(imageRout), config.BATCH_SIZE)):
			# extract the batch of images and labels, then initialize the
			# list of actual images that will be passed through the network
			# for feature extraction
			print("processing batch {}/{}".format(b + 1, batchCount))
			batchRout = imageRout[i:i + config.BATCH_SIZE]
			batchLabels = labelEncoder.transform(labels[i:i + config.BATCH_SIZE])
			batchImages = []

			# loop over the images and labels in the current batch
			for imageFile in batchRout:

				image = load_img(imageFile, target_size=(224, 224))
				image = img_to_array(image)

				image = np.expand_dims(image, axis=0)
				image = preprocess_input(image)

				# add the image to the batch
				batchImages.append(image)

			batchImages = np.vstack(batchImages)
			features = model.predict(batchImages, batch_size=config.BATCH_SIZE, verbose=0)
			features = features.reshape((features.shape[0], 7 * 7 * 512))

			# loop method to extract label
			for (label, vec) in zip(batchLabels, features):
				# construct a row that exists of the class label and
				# extracted features
				vec = ",".join([str(v) for v in vec])
				csv.write("{},{}\n".format(label, vec))

# setup a serlization for the label
with open(config.LABEL_ROUT, "wb") as f:
	f.write(pickle.dumps(labelEncoder))
