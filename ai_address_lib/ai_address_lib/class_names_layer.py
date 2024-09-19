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
    """
    A custom Keras layer that stores and handles class names for a classification task.
    
    This layer does not alter the inputs but is used to keep track of the class names in the model, 
    allowing them to be serialized and restored with the model configuration.
    
    Args:
        class_names (list): A list of class names or a string representing a single class name.
        **kwargs: Additional keyword arguments for the Keras Layer.
    """
    def __init__(self, class_names, **kwargs):
        super(ClassNamesLayer, self).__init__(**kwargs)
        self.class_names = list(class_names) if not isinstance(class_names, str) else class_names

    def call(self, inputs, **kwargs):
        """
        Passes the input through without modifying it.
        
        Args:
            inputs (Tensor): The input tensor to the layer.
        
        Returns:
            Tensor: The same input tensor.
        """
        return inputs

    def get_config(self):
        """
        Returns the configuration of the ClassNamesLayer, including the class names.
        
        Returns:
            dict: The configuration dictionary containing the class names.
        """
        config = super().get_config()
        config.update({"class_names": ','.join(self.class_names) if isinstance(self.class_names, list) else self.class_names})
        return config

    @classmethod
    def from_config(cls, config):
        """
        Recreates a ClassNamesLayer from its configuration.
        
        Args:
            config (dict): A configuration dictionary containing the class names.
        
        Returns:
            ClassNamesLayer: A new instance of the ClassNamesLayer.
        """
        config['class_names'] = config['class_names'].split(',') if isinstance(config['class_names'], str) else config['class_names']
        return cls(**config)
