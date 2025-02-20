# py_sentiment_analysis_reviews

An NLP model for classifying customer reviews using Python.

## Overview

This project fine-tunes a DistilBERT model to classify customer reviews into positive or negative sentiments. The trained model is deployed as a Flask API to allow users to analyze sentiment in real-time.

## Repository Structure

```graphql
├── data/                          # Processed dataset
│   ├── processed_train.pkl        # Preprocessed training data
│   ├── processed_test.pkl         # Preprocessed test data
├── models/                        # Trained model directory
│   ├── distilbert_fine_tuned/     # Saved fine-tuned model and tokenizer
├── notebooks/                     # Jupyter notebooks for data analysis & evaluation
│   ├── eda.ipynb                  # Exploratory Data Analysis notebook
├── src/                           # Source code
│   ├── preprocess.py              # Data preprocessing script
│   ├── train.py                   # Model training script
│   ├── predict.py                 # Model inference script
│   ├── api.py                     # Flask API implementation
├── requirements.txt               # Dependencies list
├── README.md                      # Project documentation
```

## Installation

### Clone the repository

```bash
git clone https://github.com/mattia-sanfilippo/py_sentiment_analysis_reviews.git
cd py_sentiment_analysis_reviews
```

### Create a virtual environment

```bash
python -m venv py_sentiment_analysis_reviews_env
source py_sentiment_analysis_reviews_env/bin/activate # Mac OS / Linux
.\py_sentiment_analysis_reviews_env\Scripts\activate # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download the dataset

Download the Amazon reviews dataset from [Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews), extract it and save it in the `data/` directory.
You should have the following files:

- `./data/train.ft.txt`
- `./data/test.ft.txt`

## Data preprocessing

Before training the model, preprocess the dataset by running the following command:

```bash
python src/preprocess.py
```

What this script does:

- Cleans the text data by removing special characters, digits, and stopwords.
- Maps the sentiment labels to integers (0 for negative, 1 for positive).
- Saves the preprocessed data as `processed_train.pkl` and `processed_test.pkl` in the `data/` directory.

## Model Training

Train the DistilBERT model by running the following command:

```bash
python src/train.py
```

What this script does:

- Loads the preprocessed dataset
- Fine-tunes the DistilBERT model on the training data using Hugging Face's `Trainer` API
- Saves the trained model in the `models/distilbert_fine_tuned/` directory

## Model Evaluation

To evaluate the model on the test set, run the following command:

```bash
jupyter notebook notebooks/eda.ipynb
```

- Computes Precision, Recall, and F1-score
- Displays the confusion matrix
- Identifies misclassified examples

## Using the API

Once the model is trained, start the Flask API by running the following command:

```bash
python src/api.py
```

The API will be running on `http://127.0.0.1:5000/analyze` by default.

### Send a review for sentiment analysis

To analyze the sentiment of a review, send a POST request to the API endpoint with the following payload:

#### Example request

```json
{
    "review": "The product was great! I loved it."
}
```

#### Example response

```json
{
    "label": "positive",
    "score": 0.97
}
```

## Code conventions

The project follows the [PEP 8 style guide](https://peps.python.org/pep-0008/) for Python code. To check for PEP 8 compliance, run the following command:

```bash
flake8 src/
```

When a commit is made, the pre-commit hook will run the `black` and `flake8` formatters to ensure code consistency.

If you use VSCode, the settings in `.vscode/settings.json` will automatically format the code using `black` and `flake8` when saving a file.

## Known Issues

Current limitations:

- The model struggles with some negations and sarcasm in the reviews.
- Misclassified some reviews with mixed sentiments.

Future improvements:

- Fine-tune the model on a different dataset to improve generalization. Right now, the model is trained using Amazon reviews on a very popular [Kaggle dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews).

## License

This project is licensed under the MIT License.

## Contact

Feel free to contact me through [GitHub](https://github.com/mattia-sanfilippo).
