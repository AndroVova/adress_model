from tensorflow.keras.models import load_model
from .class_names_layer import ClassNamesLayer
from .bert_layer import BertLayer
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)

def load_custom_model(model_path):
    return load_model(
        model_path, custom_objects={'ClassNamesLayer': ClassNamesLayer, 'BertLayer': BertLayer}
    )
