from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import io
import os
import torch
import pickle
import json
from model.cnn_mlp_model import CNNPlusMLPReg

# Flask setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '../frontend')

app = Flask(__name__, template_folder=FRONTEND_DIR)
CORS(app)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tabular feature names (10 features)
TAB_FEATS = [
    "koi_depth", "koi_prad", "koi_teq", "koi_insol",
    "koi_steff", "koi_slogg", "koi_srad",
    "kepmag", "period", "duration_hr"
]

# ============== Load Model on Startup ==============

model = None
scaler = None
thresholds = None

try:
    MODEL_PATH = os.path.join(BASE_DIR, 'model', 'cnn_mlp_weights.pth')
    SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
    THRESH_PATH = os.path.join(BASE_DIR, 'model', 'best_thresholds.json')
    
    print(f"Looking for model at: {MODEL_PATH}")
    print(f"Looking for scaler at: {SCALER_PATH}")
    print(f"Looking for thresholds at: {THRESH_PATH}")
    
    # Check files exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
    
    # Load model
    model = CNNPlusMLPReg(in_ch_lc=3, in_dim_tab=len(TAB_FEATS)).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    
    # Load scaler
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load thresholds
    if os.path.exists(THRESH_PATH):
        with open(THRESH_PATH, 'r') as f:
            thresholds = json.load(f)
    else:
        thresholds = {"t1": -0.5, "t2": 0.5}  # defaults
    
    print(f"✅ Model loaded successfully!")
    print(f"   - Device: {DEVICE}")
    print(f"   - Thresholds: t1={thresholds['t1']:.3f}, t2={thresholds['t2']:.3f}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()


# ============== Helper Functions ==============

def standardize_tabular(z, scaler):
    """Standardize tabular features using loaded scaler"""
    vals = []
    for k in TAB_FEATS:
        if k in z.keys():
            try:
                v = float(np.asarray(z[k]).item())
            except:
                v = np.nan
        else:
            v = np.nan
        vals.append(v)
    
    x = np.array(vals, dtype=np.float32)
    mu = scaler['mean'][:len(x)]
    sd = scaler['std'][:len(x)]
    
    # Fill NaN with mean
    bad = ~np.isfinite(x)
    x[bad] = mu[bad]
    
    # Z-score normalization
    sd_safe = np.where((~np.isfinite(sd)) | (sd==0), 1.0, sd)
    x = (x - mu) / sd_safe
    
    return x


def discretize_prediction(y_hat, t1, t2):
    """Convert continuous prediction to class label"""
    if y_hat < t1:
        return -1, "False Positive"
    elif y_hat < t2:
        return 0, "Candidate"
    else:
        return 1, "Confirmed Exoplanet"


# ============== Routes ==============

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/upload_npz', methods=['POST'])
def upload_npz():
    """Accept .npz file upload and extract data"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if not file.filename.endswith('.npz'):
            return jsonify({'error': 'File must be .npz format'}), 400
        
        file_bytes = file.read()
        npz_data = np.load(io.BytesIO(file_bytes), allow_pickle=True)
        
        data_info = extract_npz_data(npz_data)
        
        return jsonify({
            'success': True,
            'message': 'NPZ file uploaded successfully',
            'data_info': data_info
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400


@app.route('/predict', methods=['POST'])
def predict():
    """Run prediction on uploaded npz file"""
    try:
        print("\n=== PREDICT REQUEST RECEIVED ===")
        
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if not file.filename.endswith('.npz'):
            return jsonify({'error': 'File must be .npz format'}), 400
        
        # Load npz file
        file_bytes = file.read()
        z = np.load(io.BytesIO(file_bytes), allow_pickle=True)
        
        # Extract light curve data
        X = z['X'].astype(np.float32)                    # (2, L)
        xi = z['xi'].astype(np.float32)                  # phase points
        mask = z['mask'].astype(np.float32)[None, ...]   # (1, L)
        x_lc = np.concatenate([X, mask], axis=0)[None, ...]  # (1, 3, L)
        
        # Extract and standardize tabular features
        x_tab = standardize_tabular(z, scaler)[None, ...]  # (1, 10)
        
        # Convert to tensors
        x_lc_t = torch.from_numpy(x_lc).to(DEVICE)
        x_tab_t = torch.from_numpy(x_tab).to(DEVICE)
        
        # Run prediction
        with torch.no_grad():
            y_hat = model(x_lc_t, x_tab_t).item()  # continuous score [-1, 1]
        
        # Discretize to class
        t1 = thresholds.get('t1', -0.5)
        t2 = thresholds.get('t2', 0.5)
        label, classification = discretize_prediction(y_hat, t1, t2)
        
        # Calculate confidence (distance from nearest threshold)
        if label == -1:
            confidence = (t1 - y_hat) / (t1 + 1) * 100  # how far below t1
        elif label == 1:
            confidence = (y_hat - t2) / (1 - t2) * 100  # how far above t2
        else:
            dist_to_t1 = abs(y_hat - t1)
            dist_to_t2 = abs(y_hat - t2)
            confidence = (1 - min(dist_to_t1, dist_to_t2) / (t2 - t1)) * 100
        
        confidence = max(0, min(100, confidence))  # clamp to [0, 100]
        
        # Extract metadata
        kepoi_name = str(np.asarray(z['kepoi_name']).item()) if 'kepoi_name' in z else "UNKNOWN"
        kepid = int(np.asarray(z['kepid']).item()) if 'kepid' in z else -1
        
        print(f"Prediction for {kepoi_name}: {classification} (score={y_hat:.3f}, confidence={confidence:.1f}%)")
        
        result = {
            'id': 1,
            'kepoi_name': kepoi_name,
            'kepid': kepid,
            'prediction': int(label),
            'classification': classification,
            'confidence': round(confidence, 2),
            'raw_score': round(y_hat, 4),
            'thresholds': {'t1': t1, 't2': t2}
        }
        
        # ============== PREPARE LIGHT CURVE DATA FOR FRONTEND ==============
        # X has shape (2, L) where first row is typically the flux
        flux_raw = X[0, :]  # or X[1, :] depending on which is the main flux
        
        # Normalize flux for better visualization
        flux_mean = np.mean(flux_raw)
        flux_std = np.std(flux_raw)
        flux_normalized = (flux_raw - flux_mean) / flux_std if flux_std > 0 else flux_raw
        
        # Limit to 1000 points for performance
        max_points = min(1000, len(xi))
        
        light_curve_data = {
            'time': xi[:max_points].tolist(),
            'flux': flux_normalized[:max_points].tolist()
        }
        
        print(f"Sending light curve with {max_points} data points")
        # ==================================================================
        
        return jsonify({
            'success': True,
            'results': [result],
            'total_predictions': 1,
            'light_curve': light_curve_data
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400


def extract_npz_data(z):
    """Extract and validate data from npz file"""
    try:
        required_keys = ['X', 'xi', 'mask']
        missing_keys = [k for k in required_keys if k not in z.keys()]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
        
        X = z['X'].astype(np.float32)
        xi = z['xi'].astype(np.float32)
        mask = z['mask'].astype(np.float32)
        
        kepoi_name = str(np.asarray(z['kepoi_name']).item()) if 'kepoi_name' in z else "UNKNOWN"
        kepid = int(np.asarray(z['kepid']).item()) if 'kepid' in z else -1
        period = float(np.asarray(z['period']).item()) if 'period' in z else None
        duration_hr = float(np.asarray(z['duration_hr']).item()) if 'duration_hr' in z else None
        
        tabular_features = {}
        for feat in TAB_FEATS:
            if feat in z.keys():
                try:
                    tabular_features[feat] = float(np.asarray(z[feat]).item())
                except:
                    tabular_features[feat] = None
            else:
                tabular_features[feat] = None
        
        return {
            'kepoi_name': kepoi_name,
            'kepid': kepid,
            'period': period,
            'duration_hr': duration_hr,
            'light_curve_shape': list(X.shape),
            'phase_points': int(len(xi)),
            'valid_points': int(np.sum(mask > 0.5)),
            'tabular_features': tabular_features
        }
        
    except Exception as e:
        raise ValueError(f"Error extracting npz data: {str(e)}")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Exoplanet Detection Server")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)