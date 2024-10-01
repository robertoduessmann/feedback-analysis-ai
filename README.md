## feedback-analysis-ai
> API that detects the sentiment (positive, negative, neutral) of a feedback

### Donwload deps
```sh
pip install transformers torch scikit-learn pandas flask
```

### Train the Model
```sh
python3 train_sentiment.py
```

### Invoke API
```
curl --location 'http://localhost:8080/api/v1/sentiment/analyze' \
--header 'Content-Type: application/json' \
--data '"This is a fantastic product! I love it."'
```