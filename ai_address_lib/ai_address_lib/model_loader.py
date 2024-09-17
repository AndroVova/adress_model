from tensorflow.keras.models import load_model
from .class_names_layer import ClassNamesLayer
from .bert_layer import BertLayer

def load_custom_model(model_path):
    return load_model(
        model_path, custom_objects={'ClassNamesLayer': ClassNamesLayer, 'BertLayer': BertLayer}
    )
