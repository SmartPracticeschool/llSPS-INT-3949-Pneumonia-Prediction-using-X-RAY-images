import os
import zipfile
import logging

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Train:
    def __init__(self, force_download=False):
        self.force_download = force_download
        self.data_path = os.path.join(os.getcwd(), 'classifier/data/')
        self.model_path = os.path.join(os.getcwd(), 'classifier/models/')
        self.dataset_name = 'paultimothymooney/chest-xray-pneumonia'
        self.model: Sequential
        self.val: ImageDataGenerator
        self.train: ImageDataGenerator
        self.test: ImageDataGenerator
        self.epochs = 5