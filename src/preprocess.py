import pandas as pd

def load_fasttext_data(file_path):
    """Load FastText sentiment dataset and split labels from reviews."""
    reviews = []
    sentiments = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            label, review = line.split(' ', 1)  # Split into label and review
            sentiments.append(int(label.replace('__label__', '').strip()))
            reviews.append(review.strip())
    
    return pd.DataFrame({'review': reviews, 'sentiment': sentiments})

def preprocess_data(df):
    """Map labels and clean text data."""
    # Map numeric labels to textual sentiments
    df['sentiment'] = df['sentiment'].apply(lambda x: "positive" if x == 2 else "negative")
    
    # Clean text data
    df['review'] = df['review'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    return df

if __name__ == "__main__":
    # Load and preprocess train dataset
    train_data = load_fasttext_data("./data/train.ft.txt")
    train_data = preprocess_data(train_data)
    train_data.to_pickle("./data/processed_train.pkl")
    print("Processed training data saved.")
    
    # Load and preprocess test dataset
    test_data = load_fasttext_data("./data/test.ft.txt")
    test_data = preprocess_data(test_data)
    test_data.to_pickle("./data/processed_test.pkl")
    print("Processed test data saved.")
