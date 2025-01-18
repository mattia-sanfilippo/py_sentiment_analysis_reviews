from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load fine-tuned model
model = pipeline("text-classification", model="../models/distilbert_fine_tuned")

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    review = data.get("review")
    if not review:
        return jsonify({"error": "No review provided"}), 400
    
    result = model(review)
    return jsonify({"sentiment": result[0]['label'], "score": result[0]['score']})

if __name__ == "__main__":
    app.run(debug=True)
