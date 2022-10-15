# Fahad Alsehami
# Age Estimation from the Adult Pubic Symphysis project


# import os library to access our dataset path
import os

# init the parent folder of image
PARENT_DATASET = "MyData"

# initialize the parent path after the training and testing split proceeded
PRIMARY_ROUT = "dataset"

# Set directories:

TRAIN_FOLDER = "training"
TESTING_FOLDER = "evaluation"
VALIDATION = "validation"


# init. the density class labels
CLASSES = ["GOOD", "MEDIUM", "POOR"]

# Here will add the other four components on the TCM poster to compute the composite method
# setting the batch size
BATCH_SIZE = 180



# Here will construct the  label encoder file rout to the extracted features in cvs format along with the output dirct..
LABEL_ROUT = os.path.sep.join(["output", "label.cpickle"])

PARENT_CVS_ROUT = "output"



# setting the serialized rout
MODEL_ROUT = os.path.sep.join(["output", "model.cpickle"])
