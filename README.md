# Aerothona: Turbojet Diagnostic Digital Twin (HAL PS-2)

**Aerothona** is a grey-box Physics-Informed Neural Network (PINN) Digital Twin engineered to monitor the structural and efficiency degradation of a single-spool, four-stage turbojet engine. 

This repository contains the complete implementation, interactive dashboard, and final documentation for the HAL PS-2 Hackathon.

---

## 🚀 Live Digital Twin Dashboard
The interactive dashboard is deployed serverless on Netlify and runs entirely client-side with 0ms network latency:
🔗 **[Launch Aerothona Dashboard](https://aerothona-twin.netlify.app)** *(or open `public/index.html` locally)*

---

## 📁 Submission Deliverables

The workspace has been cleaned to contain only the official deliverables:

1.  **Technical Report**:
    *   **[`technical_report.pdf`](technical_report.pdf)**: Professional LaTeX-styled technical paper detailing the methodology, feature engineering, thermodynamic specific heat polynomials, physics losses, and validation results.
2.  **Source Code**:
    *   **[`turbojet_pinn_digital_twin.ipynb`](turbojet_pinn_digital_twin.ipynb)**: A single, fully self-contained Jupyter Notebook containing:
        *   Data loading & preprocessing using relative paths.
        *   PyTorch PINN model definition.
        *   Differentiable specific-heat ($c_p(T)$) thermodynamic loss constraints.
        *   Training loop (800 epochs, Adam, Cosine Annealing scheduler).
        *   Validation and $R^2$ / RMSE metrics evaluation.
        *   Automated export cell generating the JavaScript weights database (`weights.js`).
3.  **Digital Twin Dashboard**:
    *   **[`public/index.html`](public/index.html)**: Interactive HUD dashboard. Displays operating conditions, component health dials, predicted thrust, TSFC, real-time thermodynamic cycle checks, and prediction confidence intervals.
    *   **[`public/weights.js`](public/weights.js)**: Serialized weights, biases, and normalization scaling factors of the trained model.
    *   **[`public/airplane_turbine_bg.png`](public/airplane_turbine_bg.png)**: Photorealistic dashboard background.
    *   **[`netlify.toml`](netlify.toml)**: Netlify deployment configuration.
4.  **Presentation Slides**:
    *   **[`6a45efcfcf7a7_IIT_Indore-HAL_Hackathon_Template.pptx`](6a45efcfcf7a7_IIT_Indore-HAL_Hackathon_Template.pptx)**: Final presentation slides covering the engineering rationale, surrogate modeling strategy, health estimation methodology, and key results.
5.  **Datasets**:
    *   **[`datasets/`](datasets/)**: Local copies of the telemetry CSVs (`train.csv`, `test.csv`, `ground_truth.csv`) and checkpoints.

---

## 📈 Model Performance & Evaluation

The model was trained over 800 epochs on flight telemetry data. Validation results on unseen test profiles are:

| Output Parameter | Physical Definition | Validation RMSE | Test $R^2$ Score | 95% Confidence Interval |
| :--- | :--- | :--- | :--- | :--- |
| **CompressorHealth** | Compressor isentropic efficiency degradation | 0.0195 | **93.27%** | +/- 3.80% |
| **CombustorHealth** | Pressure drop degradation ratio | 0.0174 | **65.58%** | +/- 3.40% |
| **TurbineHealth** | Enthalpy extraction degradation | 0.0247 | **73.44%** | +/- 4.80% |
| **OverallHealth** | Weighted structural health index | 0.0113 | **95.15%** | +/- 2.20% |
| **Engine Thrust** | Net output force | 1,476.80 N | **99.13%** | +/- 2,895 N |
| **TSFC** | Thrust Specific Fuel Consumption | 0.0006 g/N/s | **99.25%** | +/- 0.0012 g/N/s |

---

## 🧠 Physics-Informed Constraints
To ensure thermodynamic consistency across unseen flight envelopes, the loss function penalizes non-physical predictions:
*   **Temperature-Dependent Specific Heat**: Discards constant $c_p = 1.4$ assumptions, utilizing:
    $$c_p(T) = 1005.0 + 0.05 \cdot T + 1.5 \times 10^{-4} \cdot T^2 - 8.0 \times 10^{-8} \cdot T^3 \text{ J/(kg}\cdot\text{K)}$$
*   **Isentropic Compression/Expansion Bounds**: Restricts compressor efficiency ($0.60 \le \eta_c \le 0.95$) and turbine efficiency ($0.65 \le \eta_t \le 0.98$) using ReLU-based penalty terms.
*   **Combustor Pressure Drop**: Penalizes deviations from the combustor recovery ratio ($P_3/P_2 \approx 0.949$).
*   **Overall Health Identity**: Enforces structural consistency:
    $$\text{OverallHealth} = 0.4 \cdot \text{CompressorHealth} + 0.3 \cdot \text{CombustorHealth} + 0.3 \cdot \text{TurbineHealth}$$
