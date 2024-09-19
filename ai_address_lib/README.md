# Custom ALBERT Text Classification with Keras

This project provides a custom implementation of text classification using ALBERT, LSTM, and custom Keras layers. It includes model loading, tokenization, predictions, and custom layers for handling class names and ALBERT embeddings.

## Files Overview

### `model_loader.py`
This module is responsible for loading a pre-trained Keras model with custom objects.

#### Functions:
- `load_custom_model(model_path)`: Loads the model with custom layers `ClassNamesLayer` and `BertLayer`.

### `tokenizer_utils.py`
This module provides utilities for loading a pre-trained ALBERT tokenizer and tokenizing input texts.

#### Functions:
- `load_tokenizer()`: Loads the ALBERT tokenizer (`albert-base-v2`).
- `tokenize_texts(texts, tokenizer, max_len)`: Tokenizes input texts and returns token IDs and attention masks.

### `predictions.py`
This module handles model predictions and formatting the output. It provides utilities to make predictions and process them for output.

#### Functions:
- `get_classes_from_model(model)`: Retrieves class names from the `ClassNamesLayer`.
- `make_predictions(model, tokenizer, texts, max_len)`: Tokenizes the input texts and generates predictions from the model.
- `process_predictions(text, predictions, tokenizer, classes, write_func)`: Processes the predictions and writes the formatted output using the specified write function.
- `write_current_word(classes, write_func, pred_values, idx, current_word)`: Writes the predicted class for the current word to the output.
- `write_predictions_to_file(file, text, predictions, tokenizer, classes)`: Writes predictions to a file.
- `print_predictions(text, predictions, tokenizer, classes)`: Prints predictions to the console.

### `class_names_layer.py`
This module defines the custom `ClassNamesLayer` used to store class names in the Keras model.

#### Class:
- `ClassNamesLayer`: A custom Keras layer to handle class names. It stores the class names and can be serialized and restored.

### `bert_layer.py`
This module defines the custom `BertLayer` used to integrate the pre-trained ALBERT model into a Keras model.

#### Class:
- `BertLayer`: A custom Keras layer that loads and uses the ALBERT model (`albert-base-v2`) for embeddings.

## Example Usage

```python
from transformers import AlbertTokenizer
from model_loader import load_custom_model
from predictions import get_classes_from_model, make_predictions, print_predictions
from tokenizer_utils import load_tokenizer

# Load the pre-trained model
model = load_custom_model('path_to_model.keras')

# Load the ALBERT tokenizer
tokenizer = load_tokenizer()

# Example texts to classify
texts = ["This is a test sentence for classification"]

# Get class names from the model
classes = get_classes_from_model(model)

# Make predictions
predictions = make_predictions(model, tokenizer, texts, max_len=128)

# Print predictions to console
print_predictions(texts[0], predictions, tokenizer, classes)
```

## Installation

To install the necessary dependencies, use the following command:

```bash
pip install ai_address_lib
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
