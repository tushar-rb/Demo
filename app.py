from flask import Flask, render_template, request
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Initialize Flask application
app = Flask(__name__)

# Load BERT model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-fine-tuned-squad'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    if request.method == 'POST':
        question = request.form['question']
        context = request.form['context']

        # Encode the inputs
        inputs = tokenizer.encode_plus(question, context, return_tensors='pt')

        # Get the model's predictions
        with torch.no_grad():
            outputs = model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Get the most likely start and end token positions
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)

        # Convert token positions to text
        tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = tokenizer.decode(tokens)

        return render_template('index.html', question=question, context=context, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
