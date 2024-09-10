import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

model = load_model('model/my_model_v4.keras')

with open('model/vectorize_layer_v4.pkl', 'rb') as file:
    vocabulary = pickle.load(file)

vectorize_layer = tf.keras.layers.TextVectorization(
    ragged=False,
    max_tokens=len(vocabulary),
    output_mode='int',
    output_sequence_length=None
)
vectorize_layer.set_vocabulary(vocabulary)

try:
    classes = model.class_names
except AttributeError:
    print("error getting classes, inserting them manualy")
    classes = ['CITY', 'COMP', 'COUNTRY', 'PER', 'POS', 'STREET', 'ZIP']

test_cases = [
    "Safoshyn Volodymyr\nBlagovisna street 11\n17209 Kyiv\nUkraine",
    "Safoshyn Volodymyr\nBlagovisna street 123\n07209 Odessa\nUkraine", 
    "Safoshyn Volodymyr\nBlagovisna street 1\n 08124 Lviv\n Ukraine", 
    "Safoshyn Volodymyr\nBlagovisna street 1a\n14209 Kharkiv\nUkraine",
    "Safoshyn Volodymyr\nBlagovisna street 11a\n17209 Donetsk\nUkraine", 
    "Prof. Meier Lisa\nLindenallee 10\n50667 Berlin\nDeutschland", 
    "Prof. Meier Lisa\nLindenallee 10\n50667 München", 
    "Innovate Solutions\nDr. Becker Hans\nLindenallee 10\n80331 München\nDeutschland", 
    "Techno AG\nProf. Meier Lisa\nHauptstraße 52\n80331 Hamburg", 
    "Innovate Solutions\nSenior Developer Herr Schmidt Johannes\nLindenallee 10\n10115 Köln\nDeutschland", 
    "Techno AG\nLeiter Vertrieb\nGoethestraße 5\n50667 Berlin", 
    "Dr. Becker Hans\nSenior Developer\nGoethestraße 5\n50667 München\nGermany", 
    "Herr Schmidt Johannes\nSenior Developer\nLindenallee 10\n10115 Köln", 
    "Frau Müller Anna\nHauptstraße 52\n10115 Hamburg\nGermany", 
    "Becker Hans\nHauptstraße 52\n10115 Hamburg"

]
max_len = 33

with open('resources/predictions.txt', 'w', encoding='utf-8') as file:
    for text in test_cases:
        example_sequence = vectorize_layer([text])
        
        example_sequence_padded = pad_sequences(
            example_sequence, 
            padding='post', 
            maxlen=max_len
        )

        example_sequence_padded = tf.convert_to_tensor(example_sequence_padded, dtype=tf.int32)
        
        predictions = model.predict(example_sequence_padded)
        
        vocab = vectorize_layer.get_vocabulary()
        token_to_word = {i: word for i, word in enumerate(vocab)}
        
        file.write(f"Текст: {text}\n")
        file.write(f"Предсказания:\n")
        
        tokens = example_sequence[0].numpy()
        pred_values = predictions[0].tolist()
        words = text.split()
        idx = 0
        for token, pred in zip(tokens, pred_values):
            word = token_to_word.get(token, '[UNK]')
            max_index = pred.index(max(pred))
            if max_index < len(classes):
                class_name = classes[max_index]
            else:
                class_name = "UNKNOWN"
            file.write(f"{words[idx]} ({word}) -> Class: {class_name} - {pred} \n")
            idx += 1
        
        file.write("\n")
