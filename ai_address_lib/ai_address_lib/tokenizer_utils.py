from transformers import AlbertTokenizer, logging
logging.set_verbosity_error()
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)

def load_tokenizer():
    """
    Loads and returns the pre-trained ALBERT tokenizer.
    
    Returns:
        AlbertTokenizer: The tokenizer for the 'albert-base-v2' model.
    """
    return AlbertTokenizer.from_pretrained('albert-base-v2')

def tokenize_texts(texts, tokenizer, max_len):
    """
    Tokenizes input texts using the provided tokenizer and prepares them for model input.
    
    Args:
        texts (str or list): A string or list of text sequences to be tokenized.
        tokenizer (AlbertTokenizer): The pre-trained ALBERT tokenizer used for tokenizing the input.
        max_len (int): The maximum sequence length for padding/truncation.
    
    Returns:
        tuple: A tuple containing two elements:
            - input_ids (tf.Tensor): Tensor of token IDs for the input texts.
            - attention_mask (tf.Tensor): Tensor indicating which tokens should be attended to.
    """
    inputs = tokenizer(
        texts, 
        return_tensors='tf',
        padding='max_length',
        truncation=True,
        max_length=max_len
    )
    return inputs['input_ids'], inputs['attention_mask']
