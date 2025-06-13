from flask import Flask, request, jsonify
from model.fraud_detector import FraudDetector

app = Flask(__name__)
detector = FraudDetector()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = detector.predict(data['features'])
    return jsonify({'prediction': prediction.tolist()})
