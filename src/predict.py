import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report


def load_model(model_path):
    """
    Load the trained model and tokenizer.

    Args:
        model_path (str): Path to the trained model directory.

    Returns:
        tuple: Tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model


def predict(texts, tokenizer, model):
    """
    Predict sentiments for a list of texts.

    Args:
        texts (list): List of texts to analyze.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use for encoding the texts.
        model (transformers.PreTrainedModel): Model to use for sentiment analysis.

    Returns:
        list: List of prediction results.
    """  # noqa E501
    from transformers import pipeline

    sentiment_pipeline = pipeline(
        "text-classification", model=model, tokenizer=tokenizer
    )
    results = sentiment_pipeline(texts)
    return results


def evaluate(test_data_path, model_path):
    """
    Evaluate the model on the test set.

    Args:
        test_data_path (str): Path to the preprocessed test data.
        model_path (str): Path to the trained model directory.
    """
    # Load test data
    test_data = pd.read_pickle(test_data_path)

    # Load the model and tokenizer
    tokenizer, model = load_model(model_path)

    # Predict sentiments
    texts = test_data["review"].tolist()
    true_labels = test_data["sentiment"].tolist()
    label_map = {"LABEL_0": 0, "LABEL_1": 1}

    predictions = predict(texts, tokenizer, model)
    predicted_labels = [label_map[result["label"]] for result in predictions]

    # Evaluate using classification report
    print(
        classification_report(
            true_labels, predicted_labels, target_names=["negative", "positive"]
        )
    )


if __name__ == "__main__":
    model_path = "./models/distilbert_fine_tuned"
    test_data_path = "./data/processed_test.pkl"
    evaluate(test_data_path, model_path)
