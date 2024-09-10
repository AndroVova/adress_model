import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import pickle
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, Input, Model # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tqdm import tqdm

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Используется GPU: {gpus}")
else:
    print("GPU не найден. Используется CPU.")

def load_data_from_csv(file_path):
    texts = []
    labels = []
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            texts.append(row[0])
            labels.append(row[1].split())
    
    return texts, labels

file_path = "resources/test_labeled_data.csv"
texts, labels = load_data_from_csv(file_path)

all_labels = [label for sublist in labels for label in sublist]
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
encoded_labels = [label_encoder.transform(label) for label in labels]
num_classes = len(label_encoder.classes_)
print(label_encoder.classes_)

texts_train, texts_val, labels_train, labels_val = train_test_split(texts, encoded_labels, test_size=0.33, random_state=42)

vectorize_layer = tf.keras.layers.TextVectorization(
    ragged=False,
    output_mode='int',
    output_sequence_length=None,
)
vectorize_layer.adapt(texts_train)

max_len = max(len(text.split()) for text in texts_train)

post_padded_sequences_train = tqdm(vectorize_layer(np.array(texts_train)), desc="Processing Training Sequences")
post_padded_sequences_train = tf.keras.preprocessing.sequence.pad_sequences(post_padded_sequences_train, padding='post', maxlen=max_len)

post_padded_sequences_val = tqdm(vectorize_layer(np.array(texts_val)), desc="Processing Validation Sequences")
post_padded_sequences_val = tf.keras.preprocessing.sequence.pad_sequences(post_padded_sequences_val, padding='post', maxlen=max_len)

padded_labels_train = tf.keras.preprocessing.sequence.pad_sequences(labels_train, maxlen=max_len, padding='post')
padded_labels_train = to_categorical(padded_labels_train, num_classes=num_classes)

padded_labels_val = tf.keras.preprocessing.sequence.pad_sequences(labels_val, maxlen=max_len, padding='post')
padded_labels_val = to_categorical(padded_labels_val, num_classes=num_classes)

glove_file_path = "resources/datasets/glove.6B.200d.txt"

def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

embeddings_index = load_glove_embeddings(glove_file_path)
embedding_dim = 200

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

inputs = Input(shape=(max_len,))
x = embedding_layer(inputs)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(x)
attention = layers.Attention()([x, x])
x = layers.Dense(64, activation='relu')(attention)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(
    post_padded_sequences_train, 
    padded_labels_train, 
    epochs=25, 
    batch_size=64, 
    validation_data=(post_padded_sequences_val, padded_labels_val)
)

tf.keras.models.save_model(model, 'model/my_model_v4.keras')

with open('model/vectorize_layer_v4.pkl', 'wb') as file:
    pickle.dump(vectorize_layer.get_vocabulary(), file)