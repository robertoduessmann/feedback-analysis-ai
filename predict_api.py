from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from predict import predict_sentiment

# Initialize Flask app
app = Flask(__name__)


@app.route('/api/v1/sentiment/analyze', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input. Please provide a 'text' field."}), 400

    sentiment = predict_sentiment(data['text'])
    return jsonify({"sentiment": sentiment})


# Run the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
