from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import pandas as pd

class AmazonReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label)
        
        return item

def train_model():
    # Load preprocessed data
    train_data = pd.read_pickle("./data/processed_train.pkl")
    test_data = pd.read_pickle("./data/processed_test.pkl")

    # Use a subset of the data for faster training
    train_data = train_data.sample(frac=0.03, random_state=42).reset_index(drop=True)
    test_data = test_data.sample(frac=0.03, random_state=42).reset_index(drop=True)
    
    # Convert labels to numeric (0: negative, 1: positive)
    label_map = {"negative": 0, "positive": 1}
    train_data['sentiment'] = train_data['sentiment'].map(label_map)
    test_data['sentiment'] = test_data['sentiment'].map(label_map)
    
    # Tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    # Prepare datasets
    train_dataset = AmazonReviewDataset(train_data['review'], train_data['sentiment'], tokenizer)
    test_dataset = AmazonReviewDataset(test_data['review'], test_data['sentiment'], tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/distilbert_fine_tuned",
        eval_strategy="epoch",
        per_device_train_batch_size=16,
        num_train_epochs=3,
        gradient_accumulation_steps=4,
        fp16=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    trainer.train()
    trainer.save_model("./models/distilbert_fine_tuned")
    print("Model fine-tuned and saved.")

if __name__ == "__main__":
    train_model()
