from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import torch
import os
from model.mlp_model import MLP

# Flask setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '../frontend')

app = Flask(__name__, template_folder=FRONTEND_DIR)
CORS(app)

# Load your PyTorch model
try:
    MODEL_PATH = os.path.join(BASE_DIR, 'model', 'mlp_weight.pth')
    SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

    model = MyModel()  # Make sure constructor matches mlp_model.py
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

# ---------------- Frontend Routes ----------------
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/<page>', methods=['GET'])
def other_pages(page):
    # Serve other HTML pages in frontend folder
    return render_template(f'{page}.html')

# ---------------- API Routes ----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get data from request
        data = request.json.get('data', [])
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Preprocess data
        feature_columns = df.select_dtypes(include=[np.number]).columns
        X = df[feature_columns].fillna(df.mean())

        # Scale features if scaler exists
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        # Convert to torch tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Make predictions
        with torch.no_grad():
            outputs = model(X_tensor)
            # Assuming model outputs logits or probabilities
            if outputs.shape[1] > 1:
                probs = torch.softmax(outputs, dim=1).numpy()
                preds = np.argmax(probs, axis=1)
            else:
                probs = torch.sigmoid(outputs).numpy()
                preds = (probs > 0.5).astype(int).flatten()

        # Format results
        results = []
        for i, pred in enumerate(preds):
            if outputs.shape[1] > 1:
                confidence = float(np.max(probs[i]) * 100)
                if pred == 1 or pred == 2:
                    classification = 'Confirmed Exoplanet' if confidence > 70 else 'Candidate'
                else:
                    classification = 'False Positive'
            else:
                confidence = float(probs[i][0] * 100)
                classification = 'Confirmed Exoplanet' if pred == 1 else 'False Positive'

            results.append({
                'id': i + 1,
                'classification': classification,
                'confidence': confidence
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ---------------- Run Server ----------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
