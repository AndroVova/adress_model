import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model # type: ignore
from class_names_layer import ClassNamesLayer
from bert_layer import BertLayer
from transformers import BertTokenizer

def load_custom_model(model_path):
    return load_model(
        model_path, custom_objects={'ClassNamesLayer': ClassNamesLayer, 'BertLayer': BertLayer}
    )

def get_classes_from_model(model):
    for layer in model.layers:
        if isinstance(layer, ClassNamesLayer):
            return layer.class_names
    print("Error getting classes, inserting them manually.")
    return ['CITY', 'COMP', 'COUNTRY', 'PER', 'POS', 'STREET', 'ZIP']

def tokenize_texts(texts, tokenizer, max_len):
    inputs = tokenizer(texts, return_tensors='tf', padding='max_length', truncation=True, max_length=max_len)
    return inputs['input_ids'], inputs['attention_mask']

def write_predictions_to_file(file, text, predictions, tokenizer, classes):
    file.write(f"Текст: {text}\n")
    file.write(f"Предсказания:\n")

    tokens = tokenizer.tokenize(text)
    pred_values = predictions[0].tolist()
    idx = 0
    current_word = ""

    for token in tokens:
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                if idx < len(pred_values):
                    max_index = pred_values[idx - 1].index(max(pred_values[idx - 1]))
                    class_name = classes[max_index] if max_index < len(classes) else "UNKNOWN"
                    file.write(f"{current_word} -> Class: {class_name} - {pred_values[idx - 1]} \n")
            
            current_word = token
        
        idx += 1

    if current_word:
        if idx <= len(pred_values):
            max_index = pred_values[idx - 1].index(max(pred_values[idx - 1]))
            class_name = classes[max_index] if max_index < len(classes) else "UNKNOWN"
            file.write(f"{current_word} -> Class: {class_name} - {pred_values[idx - 1]} \n")

    file.write("\n")

    
def tokenize_texts(texts, tokenizer, max_len):
    inputs = tokenizer(
        texts,
        return_tensors='tf',
        padding='max_length',
        truncation=True,
        max_length=max_len
    )
    return inputs['input_ids'], inputs['attention_mask']

def main():
    model_path = 'model/my_model_v6.keras'
    output_file_path = 'resources/bert_predictions.txt'
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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = model.input_shape[0][1]
    classes = get_classes_from_model(model)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for text in test_cases:
            input_ids, attention_mask = tokenize_texts([text], tokenizer, max_len)

            predictions = model.predict([input_ids, attention_mask])

            write_predictions_to_file(file, text, predictions, tokenizer, classes)

if __name__ == "__main__":
    main()
