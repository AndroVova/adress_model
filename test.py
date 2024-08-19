import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model('model/my_model.keras')

# Загрузка словаря токенов
with open('model/vectorize_layer.pkl', 'rb') as file:
    vocabulary = pickle.load(file)

# Восстановление слоя векторизации
vectorize_layer = tf.keras.layers.TextVectorization(
    ragged=False,
    max_tokens=len(vocabulary),
    output_mode='int',
    output_sequence_length=None
)
vectorize_layer.set_vocabulary(vocabulary)

# Примеры тестов
test_cases = [
    "You can find the address at Hauptstraße 52, 56729 Weiler.",
    "Please visit Hauptstraße 52, 56729 Weiler for more information.",
    "The location to check is Hauptstraße 52, 56729 Weiler.",
    "For your convenience, the address is Hauptstraße 52, 56729 Weiler.",
    "Our office is located at Hauptstraße 52, 56729 Weiler.",
    "Send the documents to Hauptstraße 52, 56729 Weiler.",
    "We have moved to Hauptstraße 52, 56729 Weiler.",
    "For further inquiries, visit Hauptstraße 52, 56729 Weiler.",
    "The address you need is Hauptstraße 52, 56729 Weiler.",
    "Find us at Hauptstraße 52, 56729 Weiler.",
    "Here is the address: Hauptstraße 52, 56729 Weiler. Thank you!",
    "The address you are looking for is Hauptstraße 52, 56729 Weiler.",
    "Address: Hauptstraße 52, 56729 Weiler. Please note it down.",
    "Visit Hauptstraße 52, 56729 Weiler to get more details.",
    "We are located at Hauptstraße 52, 56729 Weiler.",
    "The main office is at Hauptstraße 52, 56729 Weiler.",
    "To get directions, refer to Hauptstraße 52, 56729 Weiler.",
    "Please direct all communications to Hauptstraße 52, 56729 Weiler.",
    "Make sure to visit Pobeda street 52, 56729 Weiler for updates."
]
max_len = max(len(text.split()) for text in test_cases)
 
with open('resources/predictions.txt', 'w', encoding='utf-8') as file:
    for text in test_cases:
        example_sequence = vectorize_layer([text])
        predictions = model.predict(example_sequence)
        
        # Получение словаря токенов
        vocab = vectorize_layer.get_vocabulary()
        token_to_word = {i: word for i, word in enumerate(vocab)}
        
        # Формирование строк для записи
        file.write(f"Текст: {text}\n")
        file.write(f"Предсказания:\n")
        
        # Получение токенов и предсказаний
        tokens = example_sequence[0].numpy()
        pred_values = predictions[0].tolist()
        
        for token, pred in zip(tokens, pred_values):
            word = token_to_word.get(token, '[UNK]')
            file.write(f"{word} - {pred}\n")
        
        file.write("\n")
