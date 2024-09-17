from transformers import BertTokenizer, logging
logging.set_verbosity_error()

def load_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_texts(texts, tokenizer, max_len):
    inputs = tokenizer(
        texts, 
        return_tensors='tf',
        padding='max_length',
        truncation=True,
        max_length=max_len
    )
    return inputs['input_ids'], inputs['attention_mask']
