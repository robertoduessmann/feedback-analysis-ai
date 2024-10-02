import sys
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the saved model and tokenizer
tokenizer = BertTokenizer.from_pretrained('./sentiment_model')
model = BertForSequenceClassification.from_pretrained('./sentiment_model')


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return 'positive' if torch.argmax(probs) == 1 else 'negative'


# Example usage
if __name__ == "__main__":
    text = "The movie was fantastic! I loved every part of it."
    sentiment = predict_sentiment(text)
    print(f"Predicted sentiment: {sentiment}")
