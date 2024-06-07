from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load fine-tuned model
chatbot = pipeline("text-generation", model="path/to/save/fine-tuned-model")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("input")
    response = chatbot(user_input, max_length=50)
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
