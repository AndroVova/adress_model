from tensorflow.keras.layers import Layer
from transformers import TFBertModel
import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class BertLayer(Layer):
    def __init__(self, model_name='bert-base-uncased', **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained(model_name)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        return self.bert([input_ids, attention_mask])[0]

    def get_config(self):
        config = super(BertLayer, self).get_config()
        config.update({
            "model_name": self.bert.name
        })
        return config
