from tensorflow.keras.layers import Layer # type: ignore
from transformers import TFAlbertModel
import tensorflow as tf
from keras.saving import register_keras_serializable # type: ignore

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
            "model_name": 'albert-base-v2'
        })
        return config
    