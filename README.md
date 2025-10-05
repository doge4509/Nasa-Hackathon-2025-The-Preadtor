# 🌌 The Predator - Exoplanet Detection Webpage

A Deep learning-powered exoplanet detection system using NASA's Kepler mission data.

![Project Banner](https://img.shields.io/badge/NASA-Space%20Apps%202025-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Flask](https://img.shields.io/badge/Flask-3.0.0-lightgrey)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)

## 🎯 Features
- **Light Curve Visualization**: Real-time interactive charts using Chart.js
- **Transit Method Analysis**: Processes Kepler satellite data
- **Three Classification Types**:
  - ✅ Confirmed Exoplanet
  - ❓ Candidate
  - ❌ False Positive

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Edge)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/doge4509/Nasa-Hackathon-2025-The-Preadtor.git
cd Nasa-Hackathon-2025-The-Preadtor
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Model Files

Make sure these files exist in `backend/model/`:
- `cnn_mlp_model.py` - Model architecture
- `cnn_mlp_weights.pth` - Trained model weights
- `scaler.pkl` - Feature scaler
- `best_thresholds.json` - Classification thresholds

## 🎮 Usage

### Step 1: Start the Backend Server

```bash
cd backend
python app.py
```

You should see:
```
==================================================
🚀 Starting Exoplanet Detection Server
==================================================

✅ Model loaded successfully!
   - Device: cpu (or cuda if GPU available)
   - Thresholds: t1=-0.500, t2=0.500
```

The server will run on `http://localhost:5000`

### Step 2: Open the Frontend

Open `frontend/index.html` in your web browser, or navigate to:
```
http://localhost:5000
```

### Step 3: Upload NPZ File

1. Click on the **Upload NPZ File** section
2. Select a `.npz` file containing exoplanet light curve data
3. Click **"Analyze with AI Model"**
4. Wait for the walkie-talkie notification 📻
5. Click the notification or go to **Results** page

### Step 4: View Results

The results page will display:
- **Light Curve Chart**: Interactive visualization of flux over time
- **Classification**: Confirmed/Candidate/False Positive
- **Confidence Score**: Model certainty percentage
- **Target Information**: KOI name and metadata

## 📊 NPZ File Format

The application requires preprocessed light curve data in `.npz` format. This file stores time-series analysis data with multiple NumPy arrays.

### Required Arrays

| Array | Type | Shape | Description |
|-------|------|-------|-------------|
| `X` | float32 | (2, L) | Normalized light curve data. First row: flux values, second row: detrended/filtered flux |
| `xi` | float32 | (L,) | Phase axis (-0.5 to +0.5) or time axis (in days) |
| `mask` | bool | (L,) | Boolean mask indicating valid data points (True) and missing/invalid ones (False) |

### Required Metadata

| Field | Description | Units |
|-------|-------------|-------|
| `kepid` | Kepler Input Catalog (KIC) identifier — unique star ID | - |
| `kepoi_name` | Kepler Object of Interest (KOI) name, e.g., K00771.01 | - |
| `koi_period` | Orbital period of the transiting planet | Days |
| `koi_time0bk` | Epoch of the first transit (BKJD = BJD – 2454833) | BKJD |
| `koi_duration` | Duration of the transit event | Hours |
| `koi_disposition` | Disposition status: CONFIRMED, CANDIDATE, or FALSE POSITIVE | String |

### Optional Tabular Features

| Field | Description | Units |
|-------|-------------|-------|
| `koi_depth` | Transit depth (fractional decrease in brightness) | Fraction or % |
| `koi_prad` | Planet radius | Earth radii (R⊕) |
| `koi_teq` | Planet equilibrium temperature | Kelvin (K) |
| `koi_insol` | Stellar flux received by the planet, relative to Earth | Earth flux units (F⊕) |
| `koi_steff` | Stellar effective temperature | Kelvin (K) |
| `koi_slogg` | Logarithmic surface gravity of the host star | log(cm/s²) |
| `koi_srad` | Stellar radius | Solar radii (R☉) |
| `ra` | Right Ascension of the star | Degrees |
| `dec` | Declination of the star | Degrees |
| `koi_kepmag` | Kepler magnitude (apparent brightness) | Magnitude (mag) |

### Generating Your Own NPZ Files

If you have raw data and want to create compatible `.npz` files:

#### Helper Scripts

- **`examine.py`** – Check the contents and structure of existing `.npz` files
- **`download.py`** – Download raw data based on planet (KOI) identifier
- **`preprocess.py`** – Generate `.npz` files from raw Kepler data

### Test Files

The `test/` directory contains example `.npz` files for testing the website integration. These test files are **not included in the model training process** and serve purely for demonstration purposes.

### Train Data
Visit this https://drive.google.com/drive/u/0/folders/1DcdzmnU_P5FpI_HTP_3eBeVAHlNbtrzU to access the datasets used to train our model.

## 👥 Authors

**The Predators Team**
- William Po-Yen Chou
- Louis Yu-Cheng Chou
- Yu-Cheng Lin
- Rong-Zhu Lin
- Karl Chun-Chih Chou

## 📞 Contact

For questions or issues, please open an issue on GitHub or contact william.chou509@gmail.com