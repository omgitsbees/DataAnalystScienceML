import tensorflow_datasets as tfds 
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer 
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from flask import Flask, request, jsonify 

# Step 1: Data Collection
# Load the CNN/Daily Mail dataset
dataset, info = tfds.load('cnn_dailymail', with_info=True, as_supervised=True)
train_data, test_data = dataset['train'], dataset['test']

# Step 2: Preprocessing
def preprocess_text(text):
    # Tokenize text
    tokenizer = Tokenizer()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, padding='post')
    return padded_sequences, tokenizer

# Example preprocessing
sample_text = ["This is an example text for preprocessing."]
padded_sequences, tokenizer = preprocess_text(sample_text)

# Step 3: Model building
# Load pre-trained T5 model and tokenizer
model_name = "t5-small"
model = TFAutomodelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 4: Training
# Define training function (simplified)
def train_model(train_data, model, tokenizer):
    # Tokenize input and output texts
    inputs = tokenizer([x[0].numpy().decode('utf-8') for x in train_data], return_tensors='tf', padding=True, truncation=True)
    targets = tokenizer([x[1].numpy().decode('utf-8') for x in train_data], return_tensors='tf', padding=True, truncation=True)

    # Train the model (simplified)
    model.compile(optimizer='adam', loss=model.compute_loss)
    model.fit(inputs.input_ids, targets.input_ids, epochs=1, batch_size=8)

# Example call (only run with actual training data)
# train_model(train_data, model, tokenizer)

# Step 5: Evaluation
def evaluate_model(test_data, model, tokenizer):
    # Tokenize input texts
    inputs = tokenizer([x[0].numpy().decode('utf-8') for x in test_data], return_tensors='tf', padding=True, truncation=True)

    # Generate summaries
    outputs = model.generate(inputs.input_ids)
    summaries = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return summaries

# Example evaluation call (only run with actual test data)
# summaries = evaluate_model(test_data, model, tokenizer)

# Step 6: Application
app = Flask(__name__)

@app.route('/summarize', methods=['POST'])

def summarize():
    content = request.json
    text = content['text']

    # Preprocess text
    inputs = tokenizer([text], return_tensors='tf', padding=True, truncation=True)

    # GEnerate summary
    summary_ids = model.generate(inputs.input_ids)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens_True)

    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)