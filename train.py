# Fahad Alsehami
# Age Estimation from the Adult Pubic Symphysis project

# Import the libraries required for the training model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from settings import config
import numpy as np
import pickle
import os


# Define a function that load and split image dataset
def load_and_split(splitingRout):
	# load_data_split   # splitPath
	# initialize the data and labels
	data = []
	labels = []

	# loop over the rows in the data split file
	for row in open(splitingRout):
		# extract the class label and features from the row
		row = row.strip().split(",")
		label = row[0]
		features = np.array(row[1:], dtype="float")

		# update the data and label lists
		data.append(features)
		labels.append(label)

	# convert the data and labels to NumPy arrays
	data = np.array(data)
	labels = np.array(labels)

	# return a tuple of the data and labels
	return (data, labels)

# derive the paths to the training and testing CSV files
trainingRout = os.path.sep.join([config.PARENT_CVS_ROUT,
	"{}.csv".format(config.TRAIN_FOLDER)])
testingRout = os.path.sep.join([config.PARENT_CVS_ROUT,
	"{}.csv".format(config.TESTING_FOLDER)])

# Local dataset loading
print("Scan dataset loading...")
(trainX, trainY) = load_and_split(trainingRout)
(testX, testY) = load_and_split(testingRout)

# Local the label encoder
le = pickle.loads(open(config.LABEL_ROUT, "rb").read())

# building training model
print("Training has been started....")
model = LogisticRegression(solver="lbfgs", multi_class="auto")
model.fit(trainX, trainY)

# building evaluating model
print("Evaulation has been started....")
pred = model.predict(testX)
print(classification_report(testY, pred, target_names=le.classes_))

# Serialize the model to the local disk
print("Saving The Model ..")
f = open(config.MODEL_PATH, "wb")
f.write(pickle.dumps(model))
f.close()
