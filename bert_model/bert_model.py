import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, Input, Model # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from class_names_layer import ClassNamesLayer
from bert_layer import BertLayer

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

from transformers import TFAlbertModel, AlbertTokenizer

def create_model_with_bert(max_len, num_classes, class_names):
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
    
    bert_output = BertLayer()([input_ids, attention_mask])
    
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(bert_output)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(x)
    
    x = layers.Dense(64, activation='relu')(x)
    class_names_layer = ClassNamesLayer(class_names=class_names, name='class_names_layer')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(class_names_layer)
    
    model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def main():
    check_gpu()

    file_path = "resources/albert_labeled_data.csv"
    texts_train, texts_val, labels_train, labels_val, label_encoder, num_classes = prepare_data(file_path)

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    max_len = max(len(text.split()) for text in texts_train)
    max_len = min(max_len, 128)

    def tokenize(texts):
        return tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='tf'
        )

    train_encodings = tokenize(texts_train)
    val_encodings = tokenize(texts_val)

    padded_labels_train = tf.keras.preprocessing.sequence.pad_sequences(labels_train, maxlen=max_len, padding='post')
    padded_labels_train = to_categorical(padded_labels_train, num_classes=num_classes)
    padded_labels_val = tf.keras.preprocessing.sequence.pad_sequences(labels_val, maxlen=max_len, padding='post')
    padded_labels_val = to_categorical(padded_labels_val, num_classes=num_classes)

    model = create_model_with_bert(max_len, num_classes, label_encoder.classes_)
    model.summary()

    model.fit(
        [train_encodings['input_ids'], train_encodings['attention_mask']],
        padded_labels_train, 
        epochs=5, 
        batch_size=64, 
        validation_data=([val_encodings['input_ids'], val_encodings['attention_mask']], padded_labels_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]
    )


    tf.keras.models.save_model(model, 'model/my_model_v8.keras')

if __name__ == "__main__":
    main()
