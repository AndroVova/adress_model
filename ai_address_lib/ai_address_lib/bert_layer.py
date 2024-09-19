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
    """
    A custom Keras layer that integrates a pre-trained ALBERT model from Hugging Face's transformers library.

    Args:
        model_name (str): Name of the pre-trained ALBERT model to use. Defaults to 'albert-base-v2'.
        **kwargs: Additional keyword arguments for the Keras Layer.
    """
    def __init__(self, model_name='albert-base-v2', **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert = TFAlbertModel.from_pretrained(model_name)

    def call(self, inputs):
        """
        Performs a forward pass through the ALBERT model.

        Args:
            inputs (tuple): A tuple containing two elements:
                - input_ids: The input token IDs for the text sequences.
                - attention_mask: The mask indicating which tokens should be attended to.

        Returns:
            Tensor: The output of the ALBERT model, representing the contextual embeddings of the input tokens.
        """
        input_ids, attention_mask = inputs
        return self.bert([input_ids, attention_mask])[0]

    def get_config(self):
        """
        Returns the configuration of the BertLayer, including the model name used.

        Returns:
            dict: The configuration dictionary containing the model name.
        """
        config = super(BertLayer, self).get_config()
        config.update({
            "model_name": self.bert.name
        })
        return config
