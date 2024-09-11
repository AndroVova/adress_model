import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, Input, Model # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from glove_embeddings import create_embedding_layer
from class_names_layer import ClassNamesLayer

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Используется GPU: {gpus}")
    else:
        print("GPU не найден. Используется CPU.")

def load_data_from_csv(file_path):
    texts, labels = [], []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            texts.append(row[0])
            labels.append(row[1].split())
    return texts, labels

def prepare_data(file_path):
    texts, labels = load_data_from_csv(file_path)
    all_labels = [label for sublist in labels for label in sublist]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    encoded_labels = [label_encoder.transform(label) for label in labels]
    num_classes = len(label_encoder.classes_)
    
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts, encoded_labels, test_size=0.33, random_state=42
    )
    return texts_train, texts_val, labels_train, labels_val, label_encoder, num_classes

def create_model(vectorize_layer, embedding_layer, num_classes, label_encoder):
    inputs = Input(shape=(1,), dtype=tf.string)
    x = vectorize_layer(inputs)
    x = embedding_layer(x)
    
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(x)
    
    attention = layers.Attention()([x, x])
    
    x = layers.Dense(64, activation='relu')(attention)
    
    class_names_layer = ClassNamesLayer(class_names=label_encoder.classes_, name='class_names_layer')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(class_names_layer)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    check_gpu()

    file_path = "resources/test_labeled_data.csv"
    texts_train, texts_val, labels_train, labels_val, label_encoder, num_classes = prepare_data(file_path)

    max_len = max(len(text.split()) for text in texts_train)
    vectorize_layer = tf.keras.layers.TextVectorization(
        output_mode='int',
        output_sequence_length=max_len
    )
    vectorize_layer.adapt(texts_train)

    padded_labels_train = tf.keras.preprocessing.sequence.pad_sequences(labels_train, maxlen=max_len, padding='post')
    padded_labels_train = to_categorical(padded_labels_train, num_classes=num_classes)
    padded_labels_val = tf.keras.preprocessing.sequence.pad_sequences(labels_val, maxlen=max_len, padding='post')
    padded_labels_val = to_categorical(padded_labels_val, num_classes=num_classes)

    glove_file_path = "resources/datasets/glove.6B.200d.txt"
    embedding_dim = 200
    embedding_layer = create_embedding_layer(glove_file_path, vectorize_layer, embedding_dim, max_len)

    model = create_model(vectorize_layer, embedding_layer, num_classes, label_encoder)
    model.summary()

    texts_train = tf.convert_to_tensor(texts_train, dtype=tf.string)
    texts_val = tf.convert_to_tensor(texts_val, dtype=tf.string)

    model.fit(
        texts_train,
        padded_labels_train, 
        epochs=10, 
        batch_size=64, 
        validation_data=(texts_val, padded_labels_val)
    )

    tf.keras.models.save_model(model, 'model/my_model_v5.keras')

if __name__ == "__main__":
    main()
