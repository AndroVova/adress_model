from tensorflow.keras import layers # type: ignore

import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)


class ClassNamesLayer(layers.Layer):
    def __init__(self, class_names, **kwargs):
        super(ClassNamesLayer, self).__init__(**kwargs)
        self.class_names = list(class_names) if not isinstance(class_names, str) else class_names

    def call(self, inputs, **kwargs):
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"class_names": ','.join(self.class_names) if isinstance(self.class_names, list) else self.class_names})
        return config

    @classmethod
    def from_config(cls, config):
        config['class_names'] = config['class_names'].split(',') if isinstance(config['class_names'], str) else config['class_names']
        return cls(**config)
