from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# load model + encoder
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

@app.route('/')
def home():
    return "Server running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # convert sex
        sex = 1 if data['sex'] == 'Male' else 0

        features = np.array([[
            data['age'],
            sex,
            data['bp'],
            data['hr'],
            data['temp'],
            data['bmi']
        ]])

        prediction = model.predict(features)
        disease = encoder.inverse_transform(prediction)[0]

        # confidence
        prob = model.predict_proba(features).max()
        confidence = round(prob * 100, 2)

        return jsonify({
            "disease": disease,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)