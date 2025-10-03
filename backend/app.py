from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import torch
import os
import warnings
from model.mlp_model import MLP

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Flask setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '../frontend')

app = Flask(__name__, template_folder=FRONTEND_DIR)
CORS(app)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your PyTorch model
try:
    MODEL_PATH = os.path.join(BASE_DIR, 'model', 'mlp_weights.pth')
    SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
    
    print(f"Looking for model at: {MODEL_PATH}")
    print(f"Looking for scaler at: {SCALER_PATH}")
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    # Load scaler first to get feature columns
    scaler = joblib.load(SCALER_PATH)
    
    # Get number of features from scaler
    input_dim = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 20
    
    # FIXED: Changed num_classes from 2 to 3
    model = MLP(input_dim=input_dim, num_classes=3).to(device)  
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f"✅ Model and scaler loaded successfully!")
    print(f"   - Input dimensions: {input_dim}")
    print(f"   - Number of classes: 3")
    print(f"   - Device: {device}")
    print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    model = None
    scaler = None
    input_dim = None

# Class labels mapping
CLASS_LABELS = {
    0: 'False Positive',
    1: 'Candidate',
    2: 'Confirmed Exoplanet'
}

# ---------------- Frontend Routes ----------------
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/<page>', methods=['GET'])
def other_pages(page):
    try:
        return render_template(f'{page}.html')
    except Exception as e:
        return jsonify({'error': f'Page not found: {page}'}), 404

# ---------------- API Routes ----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        # Get data from request
        data = request.json.get('data', [])
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Preprocess data - select only numeric columns
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Ensure we have the right number of features
        if len(feature_columns) != input_dim:
            return jsonify({
                'error': f'Expected {input_dim} features, got {len(feature_columns)}',
                'provided_features': feature_columns
            }), 400
        
        X = df[feature_columns].fillna(df[feature_columns].mean())

        # Scale features
        X_scaled = scaler.transform(X)

        # Convert to torch tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        # Make predictions
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

        # Format results
        results = []
        for i, pred in enumerate(preds):
            confidence = float(np.max(probs[i]) * 100)
            classification = CLASS_LABELS.get(pred, 'Unknown')

            results.append({
                'id': i + 1,
                'prediction': int(pred),
                'classification': classification,
                'confidence': round(confidence, 2),
                'probabilities': {
                    'false_positive': round(float(probs[i][0]) * 100, 2),
                    'candidate': round(float(probs[i][1]) * 100, 2),
                    'confirmed_exoplanet': round(float(probs[i][2]) * 100, 2)
                }
            })

        return jsonify({
            'success': True,
            'results': results,
            'total_predictions': len(results)
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'input_dimensions': input_dim,
        'num_classes': 3,
        'class_labels': CLASS_LABELS,
        'device': str(device)
    })

# Model info endpoint
@app.route('/model-info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'input_dimensions': input_dim,
        'num_classes': 3,
        'class_labels': CLASS_LABELS,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'device': str(device),
        'model_architecture': {
            'type': 'Multi-Layer Perceptron (MLP)',
            'hidden_dim': 512,
            'num_blocks': 4,
            'dropout_rate': 0.3
        }
    })

# ---------------- Run Server ----------------
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Exoplanet Detection Server")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)