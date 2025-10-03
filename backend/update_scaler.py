import joblib
from sklearn.preprocessing import StandardScaler

# Load old scaler
old_scaler = joblib.load('model/scaler.pkl')

# Create new scaler with same parameters
new_scaler = StandardScaler()
new_scaler.mean_ = old_scaler.mean_
new_scaler.scale_ = old_scaler.scale_
new_scaler.var_ = old_scaler.var_
new_scaler.n_features_in_ = old_scaler.n_features_in_
new_scaler.n_samples_seen_ = old_scaler.n_samples_seen_

# Save with current version
joblib.dump(new_scaler, 'model/scaler.pkl')
print("✅ Scaler updated successfully!")