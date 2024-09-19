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
    """
    Extracts class names from a model that contains a ClassNamesLayer.
    
    Args:
        model (Model): A Keras model instance.
        
    Returns:
        list: A list of class names found in the ClassNamesLayer of the model. 
              If no such layer is found, a default set of class names is returned.
    """
    for layer in model.layers:
        if isinstance(layer, ClassNamesLayer):
            return layer.class_names
    return ['CITY', 'COMP', 'COUNTRY', 'PER', 'POS', 'STREET', 'ZIP']

def make_predictions(model, tokenizer, texts, max_len):
    """
    Prepares input texts and makes predictions using the provided model.
    
    Args:
        model (Model): The trained Keras model for making predictions.
        tokenizer: The tokenizer used for tokenizing the input texts.
        texts (str or list): Input text or list of texts to make predictions on.
        max_len (int): Maximum sequence length for tokenization.
        
    Returns:
        np.array: Model predictions for the input texts.
        
    Raises:
        ValueError: If the input texts are neither a string nor a list of strings.
    """
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
    Processes model predictions and formats the output using a specified write function.
    
    Args:
        text (str): The input text on which predictions were made.
        predictions (np.array): The model's prediction outputs.
        tokenizer: The tokenizer used for tokenizing the text.
        classes (list): List of class names corresponding to model predictions.
        write_func (function): A function used for writing output, such as `print` or `file.write`.
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
    """
    Writes the prediction for the current word to the output using the provided write function.
    
    Args:
        classes (list): List of class names for classification.
        write_func (function): A function used for writing output.
        pred_values (list): List of prediction values for the current token.
        idx (int): Index of the current token being processed.
        current_word (str): The current word being evaluated.
    """
    max_index = pred_values[idx - 1].index(max(pred_values[idx - 1]))
    class_name = classes[max_index] if max_index < len(classes) else "UNKNOWN"
    write_func(f"{current_word} -> Class: {class_name} - {pred_values[idx - 1]} \n")


def write_predictions_to_file(file, text, predictions, tokenizer, classes):
    """
    Writes predictions for the input text to a file.
    
    Args:
        file (file object): The file where the predictions will be written.
        text (str): The input text on which predictions were made.
        predictions (np.array): The model's prediction outputs.
        tokenizer: The tokenizer used for tokenizing the text.
        classes (list): List of class names corresponding to model predictions.
    """
    process_predictions(text, predictions, tokenizer, classes, file.write)


def print_predictions(text, predictions, tokenizer, classes):
    """
    Prints predictions for the input text to the console.
    
    Args:
        text (str): The input text on which predictions were made.
        predictions (np.array): The model's prediction outputs.
        tokenizer: The tokenizer used for tokenizing the text.
        classes (list): List of class names corresponding to model predictions.
    """
    process_predictions(text, predictions, tokenizer, classes, print)
