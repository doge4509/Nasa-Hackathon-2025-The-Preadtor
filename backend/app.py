from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from your_model_file import MLP  # import your class

# Flask setup
app = Flask(__name__)
CORS(app)

# Load model
input_dim = 16  # change to number of features you used
num_classes = 3  # confirmed/candidate/false positive
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLP(input_dim=input_dim, num_classes=num_classes)
model.load_state_dict(torch.load("mlp_weights.pth", map_location=device))
model.to(device)
model.eval()

# Preprocessing (same as training)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# For simplicity, you can save your scaler after training:
# import joblib
# joblib.dump(scaler, 'scaler.pkl')
# Then load it here:
# scaler = joblib.load('scaler.pkl')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Expect JSON: {"data": [ {feature1: val, feature2: val, ...}, ... ]}
        data = request.json['data']
        df = pd.DataFrame(data)

        # Preprocess
        df_scaled = scaler.transform(df)  # if scaler saved
        X_tensor = torch.FloatTensor(df_scaled).to(device)

        # Model prediction
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

        results = []
        for i, (pred, prob) in enumerate(zip(preds.cpu().numpy(), probs.cpu().numpy())):
            results.append({
                "id": i+1,
                "classification": int(pred),  # map 0/1/2 to text if you want
                "confidence": float(max(prob) * 100),
                "features": data[i]
            })

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
