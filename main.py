import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import pickle
import csv
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, Input # type: ignore
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
            labels.append(list(map(int, row[1].split())))
    
    return texts, labels

file_path = "resources/labeled_data1.csv"
texts, labels = load_data_from_csv(file_path)

texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

vectorize_layer = tf.keras.layers.TextVectorization(
    ragged=False,
    output_mode='int',
    output_sequence_length=None,
)
vectorize_layer.adapt(texts_train)

max_len = max(len(text) for text in texts_train)

post_padded_sequences_train = tqdm(vectorize_layer(np.array(texts_train)), desc="Processing Training Sequences")
post_padded_sequences_train = tf.keras.preprocessing.sequence.pad_sequences(post_padded_sequences_train, padding='post', maxlen=max_len)

post_padded_sequences_val = tqdm(vectorize_layer(np.array(texts_val)), desc="Processing Validation Sequences")
post_padded_sequences_val = tf.keras.preprocessing.sequence.pad_sequences(post_padded_sequences_val, padding='post', maxlen=max_len)

padded_labels_train = tf.keras.preprocessing.sequence.pad_sequences(labels_train, maxlen=max_len, padding='post')
padded_labels_val = tf.keras.preprocessing.sequence.pad_sequences(labels_val, maxlen=max_len, padding='post')

model = models.Sequential([
    layers.Input(shape=(max_len,)),  
    layers.Embedding(input_dim=len(vectorize_layer.get_vocabulary()), output_dim=128),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.5)),
    layers.Bidirectional(layers.LSTM(32, return_sequences=True, dropout=0.5)), 
    layers.Dense(64, activation='relu'), 
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(
    post_padded_sequences_train, 
    padded_labels_train, 
    epochs=5, 
    batch_size=64, 
    validation_data=(post_padded_sequences_val, padded_labels_val)
)

tf.keras.models.save_model(model, 'model/my_model_v3.keras')

with open('model/vectorize_layer_v3.pkl', 'wb') as file:
    pickle.dump(vectorize_layer.get_vocabulary(), file)