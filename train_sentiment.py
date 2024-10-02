import os
import tarfile
import torch
import requests
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Extract the dataset
url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
with tarfile.open(fileobj=io.BytesIO(requests.get(url).content), mode='r:gz') as tar:
    tar.extractall('.')

# Load IMDb dataset (manually processing positive and negative reviews)
def load_imdb_data(data_dir):
    data = {"text": [], "label": []}
    for label, sentiment in enumerate(["neg", "pos"]):
        folder = os.path.join(data_dir, sentiment)
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename), encoding="utf-8") as f:
                    data["text"].append(f.read())
                    data["label"].append(label)
    return pd.DataFrame(data)


# Load train dataset
train_data = load_imdb_data('./aclImdb/train')

# Filter out only a few records (e.g., 1,000 for quick training)
train_data = train_data.sample(n=1000, random_state=42)

# Preprocess data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_data['text'], train_data['label'], test_size=0.1, random_state=42)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts.to_list(
), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(
    val_texts.to_list(), truncation=True, padding=True, max_length=512)

# Create torch dataset


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)

# Model and Training
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")
