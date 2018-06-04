import os
import sys
import time
import numpy as np
import sys
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
#
# A quick one liner to install the library
# !pip install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from mrcnn.evaluate import build_coco_results, evaluate_coco
from mrcnn.dataset import MappingChallengeDataset

import zipfile
import urllib.request
import shutil

ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


PRETRAINED_MODEL_PATH = os.path.join(sys.argv[1])
LOGS_DIRECTORY = os.path.join(ROOT_DIR, "../logs")

class MappingChallengeConfig(Config):
    """Configuration for training on data in MS COCO format.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "crowdai-mapping-challenge"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    #Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # 1 Backgroun + 1 Building

    STEPS_PER_EPOCH=1000
    VALIDATION_STEPS=50
    LEARNING_RATE=1e-5

    IMAGE_MAX_DIM=256
    IMAGE_MIN_DIM=256

config = MappingChallengeConfig()
config.display()

model = modellib.MaskRCNN(mode="training", config=config, model_dir=LOGS_DIRECTORY)
# Load pretrained weights
model_path = PRETRAINED_MODEL_PATH
print(model_path)
model.load_weights(model_path, by_name=True)

# Load training dataset
dataset_train = MappingChallengeDataset()
dataset_train.load_dataset(dataset_dir=os.path.join("../", "train"), load_small=True)
dataset_train.prepare()

# Load validation dataset
dataset_val = MappingChallengeDataset()
val_coco = dataset_val.load_dataset(dataset_dir=os.path.join("../", "val"), load_small=True, return_coco=True)
dataset_val.prepare()

#Training - Stage 1
print("Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=5,
            layers='heads')
model_path = os.path.join(MODEL_DIR, "mask_rcnn_5.h5")
model.keras_model.save_weights(model_path)
# Training - Stage 2
#Finetune layers from ResNet stage 4 and up
print("Fine tune Resnet stage 4 and up")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='4+')

model_path = os.path.join(MODEL_DIR, "mask_rcnn_10.h5")
model.keras_model.save_weights(model_path)
# Training - Stage 3
# Fine tune all layers
print("Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=10,
            layers='all')
model_path = os.path.join(MODEL_DIR, "mask_rcnn_20.h5")
model.keras_model.save_weights(model_path)

print("Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=10,
            layers='all')
model_path = os.path.join(MODEL_DIR, "mask_rcnn_30.h5")
model.keras_model.save_weights(model_path)
