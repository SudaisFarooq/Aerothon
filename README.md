# Project Walkthrough - Aerothona: Turbojet Diagnostic Digital Twin

We have successfully engineered and deployed **Aerothona**, a Physics-Informed Neural Network (PINN) Digital Twin for real-time turbojet health monitoring and performance prediction. 

This walkthrough documents the final system architecture, deliverables, evaluation metrics, and deployment instructions.

---

## 1. Accomplished Tasks

*   **13-Input PINN Core**: Added `Cycle` (1 to 30) directly as the 13th input feature (index 0) to capture temporal degradation trends.
*   **Calorically Semi-Perfect Gas Model**: Replaced textbook constant specific heat assumptions ($\gamma = 1.4$) with temperature-dependent specific heat polynomials evaluated at average stage temperatures:
    $$c_p(T) = 1005.0 + 0.05 \cdot T + 1.5 \times 10^{-4} \cdot T^2 - 8.0 \times 10^{-8} \cdot T^3 \text{ J/(kg}\cdot\text{K)}$$
*   **Browser-Based Neural Network Engine**: Exported model weights, biases, and standard scaling coordinates directly into JavaScript (**`public/weights.js`**). The neural network executes feedforward evaluations locally in the client browser, bypassing server round-trip latency and removing backend hosting dependencies.
*   **One-Click Netlify Support**: Created a `netlify.toml` file directing Netlify to serve the static `public/` directory, allowing zero-config deployment.
*   **Telemetry Range Calibration**: Adjusted input ranges (30k–85k RPM, combustor temperatures up to 6.5k K) to match the micro-turbojet dataset parameters.
*   **Float-Entry Fixes**: Configured input tags with `step="any"` and bypassed slider snap roundings by reading directly from input text boxes.
*   **Notebook Generation**: Authored a complete Jupyter Notebook [turbojet_pinn_digital_twin.ipynb](file:///c:/Users/Lenovo/OneDrive%20-%20iitr.ac.in/Documents/retry/turbojet_pinn_digital_twin.ipynb) documenting the entire pipeline.

---

## 2. Test Set Evaluation Metrics (Cycle Included)

The model converged over 800 epochs with Adam optimization and a Cosine Annealing learning rate schedule:

| Output Parameter | Target Description | Validation RMSE | Test set $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **CompressorHealth** | Compressor efficiency degradation | 0.0195 | **93.27%** |
| **CombustorHealth** | Pressure drop degradation ratio | 0.0174 | **65.58%** |
| **TurbineHealth** | Enthalpy extraction degradation | 0.0247 | **73.44%** |
| **OverallHealth** | Weighted structural health index | 0.0113 | **95.15%** |
| **Thrust_N** | Net engine force | 1,476.80 N | **99.13%** |
| **TSFC_g_N_s** | Fuel efficiency metric | 0.0006 g/N/s | **99.25%** |

---

## 3. Uncertainty Quantification (95% Confidence Intervals)

Uncertainty is evaluated using standard validation RMSE bounds ($1.96 \times \text{RMSE}$):
*   **Thrust**: $\pm 2,895\text{ N}$
*   **TSFC**: $\pm 0.0012\text{ g/N/s}$
*   **Compressor Health**: $\pm 3.8\%$
*   **Combustor Health**: $\pm 3.4\%$
*   **Turbine Health**: $\pm 4.8\%$
*   **Overall Health**: $\pm 2.2\%$

---

## 4. Final File Deliverables
*   **`netlify.toml`**: Netlify configuration file.
*   **`train_brayton_pinn.py`**: Model training script.
*   **`predict.py`**: Local CLI inference utility.
*   **`server.js`**: Node.js Express server script.
*   **`turbojet_pinn_digital_twin.ipynb`**: Complete Jupyter Notebook.
*   **`public/index.html`**: Watermarked HUD-styled diagnostics page.
*   **`public/weights.js`**: Exported model parameters.
*   **`public/airplane_turbine_bg.png`**: Photorealistic background asset.
