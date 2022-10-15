# Fahad Alsehami
# Age Estimation from the Adult Pubic Symphysis project


# Library required for the 1st version of the bone age estimate project:
from settings import config
import shutil
import os
from imutils import paths


# split-loop over the data
for split in (config.TRAIN_FOLDER, config.TESTING_FOLDER, config.VALIDATION):
	# grab all image paths in the current split
	print("[Now dataset is processing '{} split'...".format(split))
	p = os.path.sep.join([config.PARENT_DATASET, split])
	imagePaths = list(paths.list_images(p))

	# loop over the image route
	for imagePath in imagePaths:
		# extract class label from the filename
		filename = imagePath.split(os.path.sep)[-1]
		label = config.CLASSES[int(filename.split("_")[0])]

		# construct the rout to the output path
		dirPath = os.path.sep.join([config.PRIMARY_ROUT, split, label])

		# create output in case we have not yet
		if not os.path.exists(dirPath):
			os.makedirs(dirPath)

		# construct and copy the route to the output image files
		p = os.path.sep.join([dirPath, filename])
		shutil.copy2(imagePath, p)
