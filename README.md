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

### Run API
```sh
python3 predict_api.py
```

### Invoke API & test
```
curl --location 'http://localhost:8000/api/v1/sentiment/analyze' \
--header 'Content-Type: application/json' \
--data '{"text": "The movie was amazing, I really enjoyed it!"}'
```