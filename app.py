from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Allow requests from any origin (for frontend)

# Sample source texts to check against (can be expanded later)
source_texts = [
    "Artificial intelligence is the simulation of human intelligence processes by machines.",
    "Machine learning is a subset of AI that provides systems the ability to automatically learn.",
    "Python is a high-level, interpreted programming language with dynamic semantics."
]

# Function to check similarity between input and source texts
def check_similarity(user_input):
    vectorizer = TfidfVectorizer().fit_transform([user_input] + source_texts)
    similarity_scores = cosine_similarity(vectorizer[0:1], vectorizer[1:])
    max_score = max(similarity_scores[0])
    return round(max_score * 100, 2)

# API route to receive text and return similarity percentage
@app.route("/check", methods=["POST"])
def check_plagiarism():
    data = request.get_json()
    user_text = data.get("text", "")

    if not user_text.strip():
        return jsonify({"error": "No text provided."}), 400

    score = check_similarity(user_text)
    return jsonify({"similarity": score})

# Start the app
if __name__ == "__main__":
    app.run(debug=True)
