import pandas as pd
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle


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
    
    start_templates = [
        "{address} is the address we are looking for.",
        "{address} is the location to visit.",
        "{address} is where you should go.",
        "{address} is the correct location.",
        "{address} is the destination address.",
        "{address} is the place to find.",
        "{address} is where the event will be held.",
        "{address} is the address you need.",
        "{address} is the location of interest.",
        "{address} is the address mentioned."
    ]
    
    middle_templates = [
        "We have an important location at {address}.",
        "Please send the information to {address} as well.",
        "The correct address is {address}, located near the park.",
        "You can find the address {address} in the document.",
        "Refer to {address} for more details.",
        "Ensure the delivery is made to {address}.",
        "The office is located at {address} on the second floor.",
        "For assistance, visit {address} in the town center.",
        "Your destination, {address}, is on the right.",
        "The new branch is at {address}, as per the latest update."
    ]
    
    end_templates = [
        "The correct address is {address}. Please confirm.",
        "You can reach us at {address}.",
        "The delivery should be made to {address}.",
        "All shipments are to be sent to {address}.",
        "Please address all correspondence to {address}.",
        "Send all inquiries to {address}.",
        "The address you need is {address}.",
        "Make sure to visit {address}.",
        "Our office is located at {address}.",
        "The final destination is {address}."
    ]
    
    all_templates = start_templates + middle_templates + end_templates
    
    for _ in range(num_samples):
        address = generate_german_address()
        template = random.choice(all_templates)
        text = template.format(address=address)
        
        tokens = text.split()
        label = [0] * len(tokens) 
        
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
    layers.Input(shape=(max_len,)),  
    layers.Embedding(input_dim=len(vectorize_layer.get_vocabulary()), output_dim=128),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.5)),
    layers.Bidirectional(layers.LSTM(32, return_sequences=True, dropout=0.5)), 
    layers.Dense(64, activation='relu'), 
    layers.Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Обучение модели
model.fit(post_padded_sequences, padded_labels, epochs=25, batch_size=64)

tf.keras.models.save_model(model, 'model/my_model.keras')

with open('model/vectorize_layer.pkl', 'wb') as file:
    pickle.dump(vectorize_layer.get_vocabulary(), file)
