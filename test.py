import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from class_names_layer import ClassNamesLayer

def load_custom_model(model_path):
    return load_model(
        model_path, custom_objects={'ClassNamesLayer': ClassNamesLayer}
    )

def get_vectorize_layer(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.TextVectorization):
            return layer
    return None

def get_classes_from_model(model):
    for layer in model.layers:
        if isinstance(layer, ClassNamesLayer):
            return layer.class_names
    print("Error getting classes, inserting them manually.")
    return ['CITY', 'COMP', 'COUNTRY', 'PER', 'POS', 'STREET', 'ZIP']

def write_predictions_to_file(file, text, predictions, vectorize_layer, classes):
    vocab = vectorize_layer.get_vocabulary()
    token_to_word = {i: word for i, word in enumerate(vocab)}
    
    file.write(f"Текст: {text}\n")
    file.write(f"Предсказания:\n")
    
    example_sequence = vectorize_layer(tf.convert_to_tensor([text], dtype=tf.string))
    tokens = example_sequence[0].numpy()
    pred_values = predictions[0].tolist()
    
    lines = text.split('\n')
    idx = 0
    
    for line in lines:
        words = line.split()
        for word in words:
            if idx < len(tokens):
                token = tokens[idx]
                word_token = token_to_word.get(token, '[UNK]')
                max_index = pred_values[idx].index(max(pred_values[idx]))
                class_name = classes[max_index] if max_index < len(classes) else "UNKNOWN"
                file.write(f"{word} ({word_token}) -> Class: {class_name} - {pred_values[idx]} \n")
                idx += 1
    
    file.write("\n")

def main():
    model_path = 'model/my_model_v5.keras'
    output_file_path = 'resources/predictions.txt'
    test_cases = [
        "Safoshyn Volodymyr\nBlagovisna street 11\n17209 Kyiv\nUkraine",
        "Safoshyn Volodymyr\nBlagovisna street 123\n07209 Odessa\nUkraine",
        "Safoshyn Volodymyr\nBlagovisna street 1\n 08124 Lviv\n Ukraine",
        "Safoshyn Volodymyr\nBlagovisna street 1a\n14209 Kharkiv\nUkraine",
        "Safoshyn Volodymyr\nBlagovisna street 11a\n17209 Donetsk\nUkraine",
        "Prof. Meier Lisa\nLindenallee 10\n50667 Berlin\nDeutschland",
        "Prof. Meier Lisa\nLindenallee 10\n50667 München",
        "Innovate Solutions\nDr. Becker Hans\nLindenallee 10\n80331 München\nDeutschland",
        "Techno Inc.\nProf. Meier Lisa\nHauptstraße 52\n80331 Hamburg",
        "Innovate Solutions\nSenior Developer Herr Schmidt Johannes\nLindenallee 10\n10115 Köln\nDeutschland",
        "Techno Ltd.\nLeiter Vertrieb\nGoethestraße 5\n50667 Berlin",
        "Dr. Becker Hans\nSenior Developer\nGoethestraße 5\n50667 München\nGermany",
        "Herr Schmidt Johannes\nSenior Developer\nLindenallee 10\n10115 Köln",
        "Frau Müller Anna\nHauptstraße 52\n10115 Hamburg\nGermany",
        "Becker Hans\nHauptstraße 52\n10115 Hamburg"
    ]

    model = load_custom_model(model_path)
    vectorize_layer = get_vectorize_layer(model)
    classes = get_classes_from_model(model)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for text in test_cases:
            input_text = tf.convert_to_tensor([text], dtype=tf.string)
            predictions = model.predict(input_text)
            write_predictions_to_file(file, text, predictions, vectorize_layer, classes)

if __name__ == "__main__":
    main()
