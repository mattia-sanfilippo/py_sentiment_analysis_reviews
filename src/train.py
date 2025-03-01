from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
import torch
import pandas as pd


class AmazonReviewDataset(Dataset):
    """
    Custom Dataset class for Amazon reviews.

    Args:
        texts (list): List of review texts.
        labels (list): List of corresponding labels.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use for encoding the texts.
        max_length (int, optional): Maximum length of the tokenized sequences. Defaults to 128.
    """  # noqa E501

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing the tokenized text and the corresponding label.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label)

        return item


def train_model():
    """
    Train the sentiment analysis model.

    Loads preprocessed training and test data, initializes the tokenizer and model,
    prepares the datasets, sets up training arguments, and trains the model.
    Saves the trained model and tokenizer to disk.
    """
    # Load preprocessed data
    train_data = pd.read_pickle("./data/processed_train.pkl")
    test_data = pd.read_pickle("./data/processed_test.pkl")

    # Use a subset of the data for faster training
    train_data = train_data.sample(frac=0.1, random_state=42).reset_index(drop=True)
    test_data = test_data.sample(frac=0.1, random_state=42).reset_index(drop=True)

    # Convert labels to numeric (0: negative, 1: positive)
    label_map = {"negative": 0, "positive": 1}
    train_data["sentiment"] = train_data["sentiment"].map(label_map)
    test_data["sentiment"] = test_data["sentiment"].map(label_map)

    # Tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    # Prepare datasets
    train_dataset = AmazonReviewDataset(
        train_data["review"], train_data["sentiment"], tokenizer
    )
    test_dataset = AmazonReviewDataset(
        test_data["review"], test_data["sentiment"], tokenizer
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/distilbert_fine_tuned",
        eval_strategy="epoch",
        per_device_train_batch_size=16,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
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

    model.save_pretrained("./models/distilbert_fine_tuned")
    tokenizer.save_pretrained("./models/distilbert_fine_tuned")

    print("Model and tokenizer saved successfully.")


if __name__ == "__main__":
    train_model()
