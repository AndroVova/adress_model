from .tokenizer_utils import tokenize_texts
from .class_names_layer import ClassNamesLayer
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)

def get_classes_from_model(model):
    for layer in model.layers:
        if isinstance(layer, ClassNamesLayer):
            return layer.class_names
    return ['CITY', 'COMP', 'COUNTRY', 'PER', 'POS', 'STREET', 'ZIP']

def make_predictions(model, tokenizer, texts, max_len):
    if isinstance(texts, list) or hasattr(texts, '__iter__'):
        texts = list(texts)
        print(f"Converted texts to Python list: {texts}, Type: {type(texts)}")
    elif isinstance(texts, str):
        texts = [texts]
    else:
        raise ValueError("Input texts must be a string or a list of strings.")
    
    input_ids, attention_mask = tokenize_texts(texts, tokenizer, max_len)
    predictions = model.predict([input_ids, attention_mask])
    return predictions

def process_predictions(text, predictions, tokenizer, classes, write_func):
    """
    Общая функция для обработки предсказаний.
    write_func: функция для вывода (например, file.write или print)
    """
    write_func(f"Текст: {text}\n")
    write_func(f"Предсказания:\n")

    tokens = tokenizer.tokenize(text, clean_up_tokenization_spaces=True)
    pred_values = predictions[0].tolist()
    idx = 0
    current_word = ""

    for token in tokens:
        if token.startswith("▁"):
            if current_word:
                if idx < len(pred_values):
                    write_current_word(classes, write_func, pred_values, idx, current_word)
            current_word = token[1:]
        else:
            current_word += token
        
        idx += 1

    if current_word:
        if idx <= len(pred_values):
            write_current_word(classes, write_func, pred_values, idx, current_word)

    write_func("\n")

def write_current_word(classes, write_func, pred_values, idx, current_word):
    max_index = pred_values[idx - 1].index(max(pred_values[idx - 1]))
    class_name = classes[max_index] if max_index < len(classes) else "UNKNOWN"
    write_func(f"{current_word} -> Class: {class_name} - {pred_values[idx - 1]} \n")


def write_predictions_to_file(file, text, predictions, tokenizer, classes):
    process_predictions(text, predictions, tokenizer, classes, file.write)


def print_predictions(text, predictions, tokenizer, classes):
    process_predictions(text, predictions, tokenizer, classes, print)
