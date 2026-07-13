import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Set seeds to ensure training runs are reproducible
torch.manual_seed(42)
np.random.seed(42)

# Point to where the raw telemetry and health ground-truth datasets are stored
data_dir = r"C:\Users\Lenovo\.gemini\antigravity\brain\db15276d-9fdf-4590-9995-3ee255368363\scratch\datasets"
train_feat_path = os.path.join(data_dir, "train.csv")
test_feat_path = os.path.join(data_dir, "test.csv")
gt_path = os.path.join(data_dir, "ground_truth.csv")

print("Loading dataset files...")
train_feat = pd.read_csv(train_feat_path)
test_feat = pd.read_csv(test_feat_path)
gt = pd.read_csv(gt_path)

# Merge the flight sensor readings with the hidden health state labels
train_df = pd.merge(train_feat, gt, on=["EngineID", "Cycle"], how="inner")
test_df = pd.merge(test_feat, gt, on=["EngineID", "Cycle"], how="inner")

# Telemetry inputs and target health/performance indicators
feature_cols = [
    'Cycle', 'Altitude_m', 'Mach', 'Tamb_K', 'Pamb_Pa', 'RPM_rev_min', 
    'FuelFlow_kg_s', 'P2_Pa', 'T2_K', 'P3_Pa', 'T3_K', 'P4_Pa', 'T4_K'
]
target_cols = [
    'CompressorHealth', 'CombustorHealth', 'TurbineHealth', 
    'OverallHealth', 'Thrust_N', 'TSFC_g_N_s'
]

# Track index mapping to extract specific sensor variables inside the physics loss
fuel_flow_idx = feature_cols.index('FuelFlow_kg_s')
Tamb_idx = feature_cols.index('Tamb_K')
Pamb_idx = feature_cols.index('Pamb_Pa')
T2_idx = feature_cols.index('T2_K')
P2_idx = feature_cols.index('P2_Pa')
T3_idx = feature_cols.index('T3_K')
P3_idx = feature_cols.index('P3_Pa')
T4_idx = feature_cols.index('T4_K')
P4_idx = feature_cols.index('P4_Pa')

# Standardize features to improve gradient flow and prevent scale imbalances
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(train_df[feature_cols])
X_test_scaled = scaler_X.transform(test_df[feature_cols])

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(train_df[target_cols])
y_test_scaled = scaler_y.transform(test_df[target_cols])

# Store scaling parameters as tensors for dynamic unscaling inside the custom loss module
y_mean = torch.tensor(scaler_y.mean_, dtype=torch.float32)
y_scale = torch.tensor(scaler_y.scale_, dtype=torch.float32)
X_mean = torch.tensor(scaler_X.mean_, dtype=torch.float32)
X_scale = torch.tensor(scaler_X.scale_, dtype=torch.float32)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)

# Neural network backbone mapping 13 inputs (sensors + cycle) to 6 outputs
class TurbojetPINN(nn.Module):
    def __init__(self, input_dim=13, output_dim=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),  # Tanh is selected to ensure continuous second-order derivatives
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

# Custom loss module enforcing calorically semi-perfect gas thermodynamics
class BraytonPINNLoss(nn.Module):
    def __init__(self, y_mean, y_scale, X_mean, X_scale,
                 fuel_flow_idx, Tamb_idx, Pamb_idx, T2_idx, P2_idx, T3_idx, P3_idx, T4_idx, P4_idx):
        super().__init__()
        # Buffer registration keeps weights and statistics on the same device (CPU/GPU)
        self.register_buffer('y_mean', y_mean)
        self.register_buffer('y_scale', y_scale)
        self.register_buffer('X_mean', X_mean)
        self.register_buffer('X_scale', X_scale)
        
        self.fuel_flow_idx = fuel_flow_idx
        self.Tamb_idx = Tamb_idx
        self.Pamb_idx = Pamb_idx
        self.T2_idx = T2_idx
        self.P2_idx = P2_idx
        self.T3_idx = T3_idx
        self.P3_idx = P3_idx
        self.T4_idx = T4_idx
        self.P4_idx = P4_idx
        self.mse = nn.MSELoss()

    def forward(self, pred_scaled, target_scaled, x_scaled, 
                lambda_health=10.0, lambda_tsfc=100.0, lambda_brayton=1.0):
        # 1. Evaluate empirical data loss (MSE) on normalized parameters
        loss_data = self.mse(pred_scaled, target_scaled)
        
        # 2. Reconstruct physical units to evaluate thermodynamic loss terms
        pred_unscaled = pred_scaled * self.y_scale + self.y_mean
        comp_h = pred_unscaled[:, 0]
        comb_h = pred_unscaled[:, 1]
        turb_h = pred_unscaled[:, 2]
        over_h = pred_unscaled[:, 3]
        thrust = pred_unscaled[:, 4]
        tsfc   = pred_unscaled[:, 5]
        
        x_unscaled = x_scaled * self.X_scale + self.X_mean
        fuel_flow = x_unscaled[:, self.fuel_flow_idx]
        Tamb = x_unscaled[:, self.Tamb_idx]
        Pamb = x_unscaled[:, self.Pamb_idx]
        T2 = x_unscaled[:, self.T2_idx]
        P2 = x_unscaled[:, self.P2_idx]
        T3 = x_unscaled[:, self.T3_idx]
        P3 = x_unscaled[:, self.P3_idx]
        T4 = x_unscaled[:, self.T4_idx]
        P4 = x_unscaled[:, self.P4_idx]
        
        # --- 3. ALGEBRAIC & EMPIRICAL CONSTRAINTS ---
        # The overall health index is defined as a weighted average of component health
        expected_overall_h = 0.4 * comp_h + 0.3 * comb_h + 0.3 * turb_h
        loss_physics_health = torch.mean((over_h - expected_overall_h) ** 2)
        
        # TSFC is physically defined as fuel flow (converted to g/s) divided by net thrust (N)
        expected_tsfc = (fuel_flow * 1000.0) / (thrust + 1e-8)
        loss_physics_tsfc = torch.mean((tsfc - expected_tsfc) ** 2)
        
        # --- 4. BRAYTON CYCLE SEMI-PERFECT THERMODYNAMIC CONSTRAINTS ---
        # Empirically evaluate specific heat (cp) as a function of local temperature.
        # This accounts for gas ionization/vibrational energy changes at high temperatures (T > 1000 K).
        def get_k(T):
            cp = 1005.0 + 0.05 * T + 1.5e-4 * (T ** 2) - 8.0e-8 * (T ** 3)
            return 287.0 / cp  # Exponent k = R/cp representing (gamma-1)/gamma

        # A. Compressor Isentropic efficiency constraint:
        # Calculates ideal exit temperature and checks predicted compressor degradation.
        T_comp_avg = (Tamb + T2) / 2.0
        kc = get_k(T_comp_avg)
        pressure_ratio_c = torch.clamp(P2 / (Pamb + 1e-8), min=0.1)
        T2_isen = Tamb * torch.pow(pressure_ratio_c, kc)
        expected_comp_h = torch.clamp((T2_isen - Tamb) / (T2 - Tamb + 1e-8), 0.5, 1.0)
        loss_brayton_comp = torch.mean((comp_h - expected_comp_h) ** 2)
        
        # B. Turbine Isentropic efficiency constraint:
        # Evaluates the expansion work extracted by the turbine stage.
        T_turb_avg = (T3 + T4) / 2.0
        kt = get_k(T_turb_avg)
        pressure_ratio_t = torch.clamp(P4 / (P3 + 1e-8), min=0.01, max=1.0)
        T4_isen = T3 * torch.pow(pressure_ratio_t, kt)
        expected_turb_h = torch.clamp((T3 - T4) / (T3 - T4_isen + 1e-8), 0.3, 1.0)
        loss_brayton_turb = torch.mean((turb_h - expected_turb_h) ** 2)

        # C. Combustor pressure drop (mean ratio ≈ 0.949, representing friction/mixing loss)
        loss_brayton_comb_p = torch.mean(((P3 - 0.949 * P2) / (P2 + 1e-8)) ** 2)
        
        total_brayton_loss = loss_brayton_comp + loss_brayton_comb_p + loss_brayton_turb
        
        # Combine data loss with weighted physics regularization terms
        total_loss = (loss_data + 
                       lambda_health * loss_physics_health + 
                       lambda_tsfc * loss_physics_tsfc + 
                       lambda_brayton * total_brayton_loss)
                       
        return total_loss, loss_data, loss_physics_health, loss_physics_tsfc, total_brayton_loss

# Initialize network, optimizer, and cosine scheduler to fine-tune learning rate
model = TurbojetPINN()
criterion = BraytonPINNLoss(y_mean, y_scale, X_mean, X_scale,
                            fuel_flow_idx, Tamb_idx, Pamb_idx, T2_idx, P2_idx, T3_idx, P3_idx, T4_idx, P4_idx)
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800, eta_min=5e-4)

# Execute the main training loop
epochs = 800
print("Training the semi-perfect Brayton-informed PINN model with Cycle...")
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    outputs_scaled = model(X_train_t)
    loss, loss_data, loss_health, loss_tsfc, loss_brayton = criterion(outputs_scaled, y_train_t, X_train_t)
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if epoch % 100 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Total Loss: {loss.item():.4f} | Data Loss: {loss_data.item():.4f} | Health Phys: {loss_health.item():.5f} | TSFC Phys: {loss_tsfc.item():.5f} | Brayton Phys: {loss_brayton.item():.5f}")

# Final validation on unseen test partition
model.eval()
with torch.no_grad():
    test_preds_scaled = model(X_test_t)
    test_preds_unscaled = (test_preds_scaled * y_scale + y_mean).numpy()

y_test_unscaled = test_df[target_cols].values
print("\n" + "="*50)
print("TEST PERFORMANCE WITH SEMI-PERFECT BRAYTON REGULARIZATION (CYCLE INCLUDED)")
print("="*50)
for i, col in enumerate(target_cols):
    col_true = y_test_unscaled[:, i]
    col_pred = test_preds_unscaled[:, i]
    rmse = np.sqrt(mean_squared_error(col_true, col_pred))
    r2 = r2_score(col_true, col_pred)
    print(f"{col:<20} | RMSE: {rmse:>10.4f} | R2 Score: {r2:>8.4f}")
print("="*50)

# Serialize the trained model weights and scaling parameters
torch.save(model.state_dict(), os.path.join(data_dir, "brayton_pinn_model.pt"))
print("Brayton PINN Model weights saved successfully!")

with open(os.path.join(data_dir, "scaler_X.pkl"), "wb") as f:
    pickle.dump(scaler_X, f)
with open(os.path.join(data_dir, "scaler_y.pkl"), "wb") as f:
    pickle.dump(scaler_y, f)
print("Scalers saved successfully!")
