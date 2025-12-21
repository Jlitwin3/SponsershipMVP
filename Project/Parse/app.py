from flask import Flask, request, jsonify
from chat import chat

app = Flask(__name__)

@app.route("/health")
def health():
    return "ok", 200

@app.route("/api/chat", methods=["POST"])
def chat_route():
    user_message = request.json["message"]
    response = chat(user_message)
    return jsonify({"response": response})
