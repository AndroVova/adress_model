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
    return AlbertTokenizer.from_pretrained('albert-base-v2')

def tokenize_texts(texts, tokenizer, max_len):
    inputs = tokenizer(
        texts, 
        return_tensors='tf',
        padding='max_length',
        truncation=True,
        max_length=max_len
    )
    return inputs['input_ids'], inputs['attention_mask']
