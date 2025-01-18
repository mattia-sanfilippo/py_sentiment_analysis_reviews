from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load the tokenizer and model
model_path = "./models/distilbert_fine_tuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Initialize the pipeline
sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Define label mapping
label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "positive"
}

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get("review")
    if not text:
        return jsonify({"error": "No review provided"}), 400

    # Perform sentiment analysis
    results = sentiment_pipeline(text)

    # Map labels to human-readable sentiments
    for result in results:
        result["label"] = label_map[result["label"]]

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
