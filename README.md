# 🏭 CNC Milling Machine Predictive Maintenance: A Physics-Informed Digital Twin



[Image of predictive maintenance process flow diagram]


## 📌 Executive Summary
Unplanned machine downtime is one of the largest cost drivers in modern manufacturing. Traditional maintenance strategies usually fall into two categories: **Reactive** (run-to-failure, which causes catastrophic damage) or **Preventive** (time-based replacement, which wastes perfectly good components). 

This project develops a **Predictive Maintenance (PdM) Pipeline** and a **Live Digital Twin Dashboard** for a CNC Milling Machine using the AI4I 2020 dataset. By combining raw sensor telemetry with **Physics-Informed Feature Engineering**, this system not only predicts *when* a machine will fail (Prognostics) but also diagnoses *which* component will fail (Diagnostics).

---

## ⚙️ Physics-Informed Feature Engineering (Domain Knowledge)
Raw sensor data (like RPM or Torque) alone often lacks the context of the physical stress the machine is experiencing. To give the Machine Learning models a deeper "mechanical understanding," I engineered the following features based on machine kinematics and thermodynamics:

### 1. Actual Mechanical Power (Watts)
A motor spinning fast with no load is fine; a motor spinning slowly under immense torque is struggling. The true measure of spindle motor effort is Mechanical Power.
* **Formula:** $P = \tau \times \omega$
* **Applied Equation:** $Power = Torque \times \left( Rotational\_Speed \times \frac{2\pi}{60} \right)$
* **Engineering Rationale:** Sudden spikes in power consumption indicate that the cutting tool is struggling against the material, which is a strong precursor to **Overstrain Failure (OSF)**.

### 2. Tool Stress Factor (Interaction Metric)
A brand-new cutting tool can easily withstand high torque. However, applying high torque to a severely worn tool guarantees breakage. 
* **Formula:** $Stress\_Factor = Torque \times Tool\_Wear$
* **Engineering Rationale:** This non-linear interaction feature helps the model understand the compounded risk. High stress factors correlate heavily with **Tool Wear Failures (TWF)**.

### 3. Thermal Inefficiency ($\Delta T$)
* **Formula:** $\Delta T = Process\_Temperature - Air\_Temperature$
* **Engineering Rationale:** A widening gap between ambient and process temperatures indicates that the machine's cooling/lubrication system is failing to dissipate friction heat, directly predicting **Heat Dissipation Failures (HDF)**.

---

## 🧠 Machine Learning Architecture

The system utilizes a **Dual-Model XGBoost Architecture** to handle both diagnostic and prognostic requirements:

### Phase 1: Overcoming the "Accuracy Trap" & Multicollinearity
* **Correlation Analysis:** A correlation heatmap revealed severe multicollinearity between raw sensors and engineered features (e.g., Torque vs. RPM has a correlation of -0.88). To optimize the model, highly correlated redundant features were dropped, retaining only the most informative physics-based features.
* **Imbalanced Data Strategy:** CNC failures are rare anomalies. A naive model could easily achieve 96% accuracy simply by predicting "Normal" for every cycle. 
* **Solution:** I utilized `scale_pos_weight` in XGBoost and shifted the optimization metric from Accuracy to **Recall**. By setting a custom **30% risk threshold**, the system prioritizes catching "hidden disasters" early, successfully identifying **82% of actual critical failures** before they occur.

### Phase 2: The Dual-Models
1. **Fault Classifier (Diagnostic):** A Multi-Class XGBoost Classifier that identifies the exact failure mode (TWF, HDF, PWF, OSF) to automate spare part procurement.
2. **RUL Regressor (Prognostic):** An XGBoost Regressor trained to predict the **Remaining Useful Life (RUL)**. It calculates exactly how many machining cycles are left before the degradation path intersects the critical failure limit.



---

## 🖥️ Live Digital Twin Dashboard
To bridge the gap between Data Science and the factory floor, I deployed the models into a real-time **Digital Twin UI** using Gradio.

**Features of the Dashboard:**
* **Live Telemetry Input:** Operators (or automated PLCs) feed RPM, Torque, and Tool Wear data.
* **Dynamic Degradation Tracking:** Real-time plotting of the machine's drift towards the critical failure limit.
* **RUL Forecasting:** A live countdown of remaining safe production cycles.
* **AI Action Plan:** Automated, color-coded alerts instructing operators whether to continue production, schedule maintenance, or hit the Emergency Stop.

---

## 📂 Repository Structure

```text
predictive-maintenance-cnc/
│
├── data/
│   └── ai4i2020.csv               # Raw Kaggle dataset
│
├── notebooks/
│   └── 01_predictive_maintenance_modeling.ipynb  # EDA, Feature Eng, & Model Training
│
├── models/
│   └── milling_ai_bundle_2026.pkl # Exported Dual-XGBoost models & encoders
│
├── app.py                         # Gradio Live Dashboard Deployment Script
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation


🚀 How to Run Locally
Clone the repository:

Bash
git clone [https://github.com/yourusername/predictive-maintenance-cnc.git](https://github.com/yourusername/predictive-maintenance-cnc.git)
cd predictive-maintenance-cnc

Install dependencies:

Bash
pip install -r requirements.txt


Launch the Digital Twin Dashboard:

Bash
python app.py