"""
Simple Flask web interface for the text classifier.
"""

from flask import Flask, render_template, request, jsonify

from src.predict import predict

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    text = request.form.get("text", "").strip()

    if not text:
        return render_template("index.html", error="Please enter a project idea.")

    results = predict(text)
    return render_template("index.html", results=results, input_text=text)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for programmatic access."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    results = predict(text)
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
