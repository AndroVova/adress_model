from tensorflow.keras.layers import Layer
from transformers import TFAlbertModel
from keras.saving import register_keras_serializable
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)


@register_keras_serializable()
class BertLayer(Layer):
    def __init__(self, model_name='albert-base-v2', **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert = TFAlbertModel.from_pretrained(model_name)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        return self.bert([input_ids, attention_mask])[0]

    def get_config(self):
        config = super(BertLayer, self).get_config()
        config.update({
            "model_name": self.bert.name
        })
        return config
