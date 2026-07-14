# Technical Report: Aerothona Turbojet Diagnostic Digital Twin
**Widescreen Submission Reference & Engineering Architecture Document**
**Project Lead & Core Developer**: Sudais Farooq
**Target Platform**: HAL x IIT Indore Hackathon 2026 (Problem Statement 2)

---

## 1. Executive Summary
**Aerothona** is a grey-box Physics-Informed Neural Network (PINN) Digital Twin engineered to monitor the health degradation of a single-spool, four-stage turbojet engine. The system estimates hidden subsystem state metrics (Compressor, Combustor, and Turbine health indices) alongside critical engine metrics (Net Thrust, Thrust Specific Fuel Consumption) in real-time. 

Rather than relying purely on data-driven mappings which fail under flight transitions, Aerothona integrates gas dynamics equations directly into the backpropagation loss loop. Specifically, we discard constant specific heat ratio ($\gamma = 1.4$) assumptions, replacing them with temperature-dependent specific heat polynomials evaluated at average stage temperatures.

To enable edge diagnostics, the trained PyTorch weights and standard scaling parameters are exported to JavaScript (`weights.js`). Forward-pass neural evaluations execute locally in the user's browser, enabling sub-millisecond, zero-network-latency diagnostics served from a static, serverless Netlify deployment.

---

## 2. Problem Statement & Motivation
In modern turbine propulsion systems, continuous monitoring of thermal and structural degradation is vital to prevent catastrophic in-flight failures. However, key physical indicators cannot be measured directly:
1.  **Extreme Thermal Environments**: Physical sensors cannot survive at Station 3 (turbine inlet temperature $T_3$ is up to 6,500 K) or inside interior blade stages.
2.  **Transitional Envelopes**: Turbojets operate over continuous flight conditions (Altitudes from 0 to 12,053 m, speeds from 0 to 0.90 Mach, spool speeds up to 80,941 RPM). Purely empirical neural networks extrapolate poorly outside their training data, predicting non-physical outputs (e.g., negative thrust or efficiencies exceeding 100%).
3.  **Edge Execution Latency**: Querying heavy deep learning models on a remote cloud server introduces network latency, rendering real-time pilot warnings impossible.

Aerothona solves this by constructing a hybrid neural model that is both **physically bounded** by thermodynamic conservation laws and **serverless** for instantaneous evaluation.

---

## 3. Deep Learning Architecture & Pipeline

### 3.1 Network Topology
The system utilizes a Multi-Layer Perceptron (MLP) architecture:
*   **Input Layer (13 Nodes)**: 
    *   Index 0: `Cycle` (Degradation index, 1 to 30)
    *   Index 1: `Altitude_m` (m)
    *   Index 2: `Mach` (Ma)
    *   Index 3: `Tamb_K` (K)
    *   Index 4: `Pamb_Pa` (Pa)
    *   Index 5: `RPM_rev_min` (rpm)
    *   Index 6: `FuelFlow_kg_s` (kg/s)
    *   Index 7: `P2_Pa` (Compressor Exit Pressure)
    *   Index 8: `T2_K` (Compressor Exit Temperature)
    *   Index 9: `P3_Pa` (Turbine Inlet Pressure)
    *   Index 10: `T3_K` (Turbine Inlet Temperature)
    *   Index 11: `P4_Pa` (Turbine Exit Pressure)
    *   Index 12: `T4_K` (Turbine Exit Temperature)
*   **Hidden Layers**: Three dense layers containing **64, 64, and 32 nodes** respectively, utilizing **Hyperbolic Tangent (Tanh)** activation functions. Tanh is chosen because its second-order derivatives are continuous, allowing smooth backpropagation gradients of physical equations.
*   **Output Layer (6 Nodes, Linear Activation)**:
    *   `CompressorHealth`: Target efficiency index (0.0 to 1.0)
    *   `CombustorHealth`: Target pressure recovery index (0.0 to 1.0)
    *   `TurbineHealth`: Target enthalpy extraction index (0.0 to 1.0)
    *   `OverallHealth`: Weighted integrity rating (0.0 to 1.0)
    *   `Thrust_N`: Engine net output force (N)
    *   `TSFC_g_N_s`: Fuel consumption rate (g/N/s)

### 3.2 Preprocessing & Standard Scaling
To prevent scale imbalances (e.g., pressures in $10^5\text{ Pa}$ vs. fuel flow in $10^{-1}\text{ kg/s}$), standard scaling is applied to inputs $X$ and targets $y$:
$$X_{\text{scaled}} = \frac{X - X_{\text{mean}}}{X_{\text{scale}}}, \quad y_{\text{scaled}} = \frac{y - y_{\text{mean}}}{y_{\text{scale}}}$$
The scaling parameters are stored in `weights.js` to scale inputs and unscale outputs on the client-side.

---

## 4. Physics-Informed Regularization (Thermodynamic Loss)
The total training loss function combines standard mean squared error ($\mathcal{L}_{\text{data}}$) with four thermodynamic regularizations ($\mathcal{L}_{\text{physics}}$):
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_1 \mathcal{L}_{\text{comp}} + \lambda_2 \mathcal{L}_{\text{turb}} + \lambda_3 \mathcal{L}_{\text{comb}} + \lambda_4 \mathcal{L}_{\text{identity}}$$

### 4.1 Temperature-Dependent Calorically Semi-Perfect Gas Model
Instead of assuming a constant specific heat ratio ($\gamma = 1.4$), the specific heat $c_p(T)$ is evaluated using a fourth-order polynomial for air:
$$c_p(T) = 1005.0 + 0.05 \cdot T + 1.5 \times 10^{-4} \cdot T^2 - 8.0 \times 10^{-8} \cdot T^3 \text{ J/(kg}\cdot\text{K)}$$
The gas constant for air is $R = 287.058\text{ J/(kg}\cdot\text{K)}$, giving a temperature-dependent exponent:
$$k(T) = \frac{R}{c_p(T)}$$

### 4.2 Compressor Efficiency Loss ($\mathcal{L}_{\text{comp}}$)
Evaluated at average compressor temperature $T_{\text{comp\_avg}} = (T_{\text{amb}} + T_2) / 2$:
$$c_{p,c} = c_p(T_{\text{comp\_avg}}), \quad k_c = \frac{R}{c_{p,c}}$$
The theoretical isentropic exit temperature is:
$$T_{2,\text{isen}} = T_{\text{amb}} \cdot \left(\frac{P_2}{P_{\text{amb}}}\right)^{k_c}$$
The calculated compressor efficiency is:
$$\eta_c = \frac{T_{2,\text{isen}} - T_{\text{amb}}}{T_2 - T_{\text{amb}}}$$
The loss term penalizes efficiencies falling outside the physical boundaries $[0.60, 0.95]$:
$$\mathcal{L}_{\text{comp}} = \operatorname{MSE}(\max(0, 0.60 - \eta_c) + \max(0, \eta_c - 0.95))$$

### 4.3 Turbine Efficiency Loss ($\mathcal{L}_{\text{turb}}$)
Evaluated at average turbine temperature $T_{\text{turb\_avg}} = (T_3 + T_4) / 2$:
$$c_{p,t} = c_p(T_{\text{turb\_avg}}), \quad k_t = \frac{R}{c_{p,t}}$$
The theoretical isentropic exit temperature is:
$$T_{4,\text{isen}} = T_3 \cdot \left(\frac{P_4}{P_3}\right)^{k_t}$$
The calculated turbine efficiency is:
$$\eta_t = \frac{T_3 - T_4}{T_3 - T_{4,\text{isen}}}$$
The loss term penalizes efficiencies falling outside the physical boundaries $[0.65, 0.98]$:
$$\mathcal{L}_{\text{turb}} = \operatorname{MSE}(\max(0, 0.65 - \eta_t) + \max(0, \eta_t - 0.98))$$

### 4.4 Combustor Pressure Drop Loss ($\mathcal{L}_{\text{comb}}$)
Models the empirical combustor pressure recovery ratio ($P_3/P_2 \approx 0.949$):
$$\mathcal{L}_{\text{comb}} = \operatorname{MSE}\left(\frac{P_3}{P_2} - 0.949\right)$$

### 4.5 Subsystem Health Identity Loss ($\mathcal{L}_{\text{identity}}$)
Enforces the weighted overall health equation:
$$\mathcal{L}_{\text{identity}} = \operatorname{MSE}(\text{OverallHealth} - (0.4 \cdot \text{CompressorHealth} + 0.3 \cdot \text{CombustorHealth} + 0.3 \cdot \text{TurbineHealth}))$$

---

## 5. Training Performance & Validation Metrics
The model was trained over 800 epochs using the Adam optimizer with Cosine Annealing learning rate scheduling. The final evaluation metrics on the unseen test dataset are:

| Target Parameter | Validation RMSE | Test Set $R^2$ Score | 95% Confidence Interval |
| :--- | :--- | :--- | :--- |
| **CompressorHealth** | 0.0195 | **93.27%** | $\pm 3.80\%$ |
| **CombustorHealth** | 0.0174 | **65.58%** | $\pm 3.40\%$ |
| **TurbineHealth** | 0.0247 | **73.44%** | $\pm 4.80\%$ |
| **OverallHealth** | 0.0113 | **95.15%** | $\pm 2.20\%$ |
| **Engine Thrust (N)** | 1,476.80 N | **99.13%** | $\pm 2,895\text{ N}$ |
| **TSFC (g/N/s)** | 0.0006 | **99.25%** | $\pm 0.0012\text{ g/N/s}$ |

### Uncertainty Quantification (UQ)
Uncertainty is modeled dynamically using standard validation RMSE bounds ($1.96 \times \text{RMSE}$). These confidence bounds are printed dynamically beneath all target gauges and metric readouts on the UI dashboard.

---

## 6. Client-Side Browser Engine & Deployment

### 6.1 Serverless Matrix Operations
The PyTorch neural network parameters were exported into JavaScript matrices inside `weights.js`:
*   `modelWeights.w1, w2, w3, w4`
*   `modelWeights.b1, b2, b3, b4`

The dashboard computes the forward pass locally in JavaScript:
```javascript
function runLocalPrediction(inputs) {
  // 1. Standard scale inputs
  const scaled = inputs.map((val, i) => (val - X_mean[i]) / X_scale[i]);
  
  // 2. Feedforward Layer Passes (inputs * W + b) with Tanh activation
  const h1 = forwardLayer(scaled, w1, b1, true);
  const h2 = forwardLayer(h1, w2, b2, true);
  const h3 = forwardLayer(h2, w3, b3, true);
  const out_scaled = forwardLayer(h3, w4, b4, false); // Linear output
  
  // 3. Inverse scale outputs to original physical units
  return out_scaled.map((val, i) => val * y_scale[i] + y_mean[i]);
}
```
This architecture provides several benefits:
*   **0ms Latency**: No server-side processing overhead or network routing delays.
*   **100% Offline Capability**: Runs locally in any web browser without active internet connection.
*   **Static Hosting Compatibility**: Deployable on zero-cost static CDNs (Netlify, GitHub Pages) without configuring python environments.

### 6.2 Netlify Deployment Configuration
The project is configured via `netlify.toml` in the root folder:
```toml
[build]
  publish = "public"
```
Linking your GitHub repository (`https://github.com/SudaisFarooq/Aerothon`) to Netlify will automatically build and host the public folder.
