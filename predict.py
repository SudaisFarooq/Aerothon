import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import pickle
import warnings

# Suppress sklearn standard scaler warnings to output clean JSON to stdout
warnings.filterwarnings("ignore", category=UserWarning)

# Paths containing the trained weights and scaling properties
data_dir = r"C:\Users\Lenovo\.gemini\antigravity\brain\db15276d-9fdf-4590-9995-3ee255368363\scratch\datasets"
model_weights_path = os.path.join(data_dir, "brayton_pinn_model.pt")
scaler_X_path = os.path.join(data_dir, "scaler_X.pkl")
scaler_y_path = os.path.join(data_dir, "scaler_y.pkl")

# Multilayer Perceptron backbone representing the digital twin
class TurbojetPINN(nn.Module):
    def __init__(self, input_dim=13, output_dim=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

def run_prediction(inputs_list):
    # Load scaling parameters to match training bounds
    with open(scaler_X_path, "rb") as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, "rb") as f:
        scaler_y = pickle.load(f)
    
    # Load trained model state
    model = TurbojetPINN()
    model.load_state_dict(torch.load(model_weights_path, weights_only=True))
    model.eval()
    
    # Scale inputs, evaluate network, and map back to physical units
    inputs_arr = np.array([inputs_list], dtype=np.float32)
    inputs_scaled = scaler_X.transform(inputs_arr)
    
    inputs_t = torch.tensor(inputs_scaled, dtype=torch.float32)
    with torch.no_grad():
        preds_scaled_t = model(inputs_t)
        preds_scaled = preds_scaled_t.numpy()
        
    preds_unscaled = scaler_y.inverse_transform(preds_scaled)[0]
    
    # Structure diagnostic predictions
    result = {
        "CompressorHealth": float(preds_unscaled[0]),
        "CombustorHealth": float(preds_unscaled[1]),
        "TurbineHealth": float(preds_unscaled[2]),
        "OverallHealth": float(preds_unscaled[3]),
        "Thrust_N": float(preds_unscaled[4]),
        "TSFC_g_N_s": float(preds_unscaled[5])
    }
    return result

if __name__ == "__main__":
    if len(sys.argv) < 14:
        print(json.dumps({"error": "Missing inputs. Expected 13 features (including Cycle)."}))
        sys.exit(1)
        
    try:
        # Extract features from subprocess CLI arguments
        inputs = [float(sys.argv[i]) for i in range(1, 14)]
        pred = run_prediction(inputs)
        print(json.dumps(pred))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
