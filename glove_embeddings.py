import numpy as np
from tensorflow.keras import layers # type: ignore

def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def create_embedding_layer(glove_file_path, vectorize_layer, embedding_dim, max_len):
    embeddings_index = load_glove_embeddings(glove_file_path)
    vocab = vectorize_layer.get_vocabulary()
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for i, word in enumerate(vocab):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = layers.Embedding(
        input_dim=len(vocab),
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=True
    )
    
    return embedding_layer
