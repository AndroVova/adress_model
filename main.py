import pandas as pd
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Загрузка данных из CSV
unique_streets_df = pd.read_csv('resources/unique_streets.csv')
unique_cities_df = pd.read_csv('resources/unique_cities.csv')
cleaned_geonames_postal_code_df = pd.read_csv('resources/cleaned_geonames_postal_code.csv')

# 2. Функция для генерации немецких адресов
def generate_german_address():
    streets = unique_streets_df['Schwabstraße'].tolist()
    house_number = random.randint(1, 100)
    
    city_data = random.choice(cleaned_geonames_postal_code_df.values)
    postal_code = city_data[1]
    city = city_data[2]
    
    street_address = f"{random.choice(streets)} {house_number}"
    full_address = f"{street_address}, {postal_code} {city}"
    
    return full_address

# 3. Функция для разметки данных
def create_labeled_data_with_tokens(num_samples=10000):
    texts = []
    labels = []
    
    for _ in range(num_samples):
        address = generate_german_address()
        text = f"Here is the address: {address}"
        
        # Токенизация текста
        tokens = text.split()  # Простая токенизация по пробелам
        label = [0] * len(tokens)
        
        # Выделение адреса
        address_tokens = address.split()
        address_start = tokens.index(address_tokens[0])
        address_end = address_start + len(address_tokens)
        
        for i in range(address_start, address_end):
            label[i] = 1
        
        texts.append(text)
        labels.append(label)
    
    return texts, labels

texts, labels = create_labeled_data_with_tokens(num_samples=10000)

print(texts[2])
print(labels[2])

# 4. Векторизация текстов
vectorize_layer = tf.keras.layers.TextVectorization(ragged=False, max_tokens=20000, output_mode='int', output_sequence_length=None)
vectorize_layer.adapt(texts)

# Преобразование меток в тензоры
max_len = max(len(text) for text in texts)
post_padded_sequences = vectorize_layer(np.array(texts))
post_padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(post_padded_sequences, padding='post', maxlen=max_len)

# Преобразование меток в тензоры и паддинг
padded_labels = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=max_len, padding='post')

# 5. Создание модели на базе LSTM
model = models.Sequential([
    layers.Input(shape=(max_len,)),  # Входное слое с определенной длиной последовательности
    layers.Embedding(input_dim=len(vectorize_layer.get_vocabulary()), output_dim=128),  # Векторизация
    layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.5)),  # Первый слой LSTM с Dropout
    layers.Bidirectional(layers.LSTM(32, return_sequences=True, dropout=0.5)),  # Второй слой LSTM с Dropout
    layers.Dense(64, activation='relu'),  # Полносвязный слой
    layers.Dense(1, activation='sigmoid')  # Сигмоидная активация для бинарной классификации
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Обучение модели
model.fit(post_padded_sequences, padded_labels, epochs=5, batch_size=32)

# 7. Проверка на примере
example_text = texts[0]
example_sequence = vectorize_layer([example_text])
predictions = model.predict(example_sequence)

print("Текст:", example_text)
print("Предсказания:", predictions)
