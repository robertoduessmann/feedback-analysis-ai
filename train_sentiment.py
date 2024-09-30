import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import torch

# Load IMDb dataset (or any sentiment dataset you prefer)
dataset = pd.read_csv('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', delimiter='\t')
# Filter out only a few records (e.g., 10,000 for quick training)
dataset = dataset.sample(n=1000)

# Preprocess data
train_texts, val_texts, train_labels, val_labels = train_test_split(dataset['text'], dataset['label'], test_size=0.1)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts.to_list(), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts.to_list(), truncation=True, padding=True, max_length=512)

# Create torch dataset
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)

# Model and Training
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
try:
    model.save_pretrained("/Users/robertoduessmann/work/workspace/feedback-analysis-ai/sentiment_model")
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")
tokenizer.save_pretrained("/Users/robertoduessmann/work/workspace/feedback-analysis-ai/sentiment_model")