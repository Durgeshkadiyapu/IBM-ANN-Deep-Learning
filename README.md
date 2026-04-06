# 📞 Telecom Customer Churn Prediction — ANN with Keras Tuner

A deep learning project that predicts customer churn for a telecom company using an **Artificial Neural Network (ANN)** with automated hyperparameter optimization via **Keras Tuner**. Built on IBM's Telco Customer Churn dataset.

---

## 📌 Project Overview

Customer churn is a critical business problem — retaining existing customers is far cheaper than acquiring new ones. This project builds an end-to-end churn prediction pipeline: from raw data ingestion and feature engineering, through ANN architecture design, to automated hyperparameter tuning and a deployable prediction function.

---

## 🚀 Key Highlights

| Metric | Value |
|--------|-------|
| Dataset | IBM Telco Customer Churn Dataset (7,043 customers) |
| Task | Binary Classification (Churn: Yes / No) |
| Architecture | Multi-layer ANN (TensorFlow / Keras) |
| Feature Selection | Random Forest + RFE (top 5 features) |
| Hyperparameter Tuning | **Keras Tuner** (RandomSearch) |
| Evaluation | Accuracy, Loss (training & validation curves) |
| Deployment | Reusable `predict_churn()` function with serialized model |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Deep Learning | TensorFlow, Keras (`models`, `layers`) |
| Hyperparameter Tuning | **Keras Tuner** (RandomSearch) |
| Feature Selection | Scikit-learn (RandomForestClassifier + RFE) |
| Preprocessing | StandardScaler, Label Encoding (`pd.factorize`) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Serialization | Pickle (model, scaler, selected features) |
| Environment | Google Colab / Jupyter Notebook |

---

## 📂 Project Structure

```
IBM_ANN/
│
├── IBM_ANN.ipynb               # Main notebook — full pipeline
├── best_churn_model.h5         # Saved best ANN model
├── scaler.pkl                  # Saved StandardScaler
├── selected_features.pkl       # Saved top 5 RFE-selected features
└── README.md                   # Project documentation
```

---

## ⚙️ How It Works

### 1. Data Loading
- Loaded IBM Telco Customer Churn dataset directly from GitHub
- Dataset contains 7,043 customers with 21 features including tenure, charges, contract type, internet service, and more

### 2. Data Preprocessing
- Encoded target variable: `Churn` → {Yes: 1, No: 0}
- Encoded `gender` → {Female: 0, Male: 1}
- Converted `TotalCharges` to numeric (coerced errors)
- Label-encoded all remaining categorical columns using `pd.factorize()`
- Removed rows with null values (`TotalCharges` had 11 missing entries)

### 3. Exploratory Data Analysis
- Correlation heatmap of all numerical features
- Identified relationships between monthly charges, tenure, total charges, and churn

### 4. Feature Selection (RFE + Random Forest)
- Applied **Recursive Feature Elimination (RFE)** with `RandomForestClassifier` (150 estimators)
- Selected **top 5 most predictive features** out of 19
- Plotted feature importance rankings for all features

### 5. Train/Test Split & Scaling
- 80/20 train-test split (random_state=42 for reproducibility)
- Applied **StandardScaler** on training data; transformed test data with the same scaler
- Serialized scaler to `scaler.pkl` for deployment

### 6. Initial ANN Model
- Architecture: `Dense(64, relu)` → `Dense(32, relu)` → `Dense(16, relu)` → `Dense(8, relu)` → `Dense(1, sigmoid)`
- Compiled with Adam optimizer, binary cross-entropy loss
- Trained for 50 epochs with batch size 32, 20% validation split
- Plotted training vs validation loss and accuracy curves

### 7. Hyperparameter Tuning (Keras Tuner)
- Used **Keras Tuner RandomSearch** to automatically find the best architecture:
  - Layer 1 units: {32, 64, 128}
  - Optional second layer: {True, False} with units {16, 32}
- EarlyStopping callback (patience=5 on val_loss)
- Objective: maximize `val_accuracy` over 5 trials

### 8. Best Model Training & Evaluation
- Rebuilt and retrained the best architecture found by Keras Tuner for 50 epochs
- Evaluated on held-out test set: Loss and Accuracy reported
- Saved best model to `best_churn_model.h5`

### 9. Deployment Function
- `predict_churn(*args)` — loads saved model, scaler, and features; accepts raw customer feature values and returns:
  - Churn Probability (0.0 to 1.0)
  - Will Customer Churn: "Yes" (probability > 0.50) or "No"

---

## 📊 Pipeline Summary

```
Raw Data (IBM GitHub) → Encoding + Cleaning → Correlation EDA
                     → RFE Feature Selection (Top 5) → Train/Test Split
                     → StandardScaler → Initial ANN Training
                     → Keras Tuner Hyperparameter Search → Best Model Training
                     → Model Evaluation → Serialization → predict_churn()
```

---

## 🔮 Sample Prediction

```python
predict_churn(34, 1, 1, 56.96, 1900)
# Output:
# {"Churn Probability": 0.23, "Will Customer Churn": "No"}
```

---

## 🔧 How to Run

```bash
# Install dependencies
pip install tensorflow keras-tuner scikit-learn pandas numpy matplotlib seaborn

# Launch the notebook
jupyter notebook IBM_ANN.ipynb
```

> **Note:** The dataset loads directly from IBM's GitHub repository — no local file needed.

---

## 🧠 What I Learned

- End-to-end ANN pipeline for binary classification on real business data
- Automated hyperparameter tuning using Keras Tuner (RandomSearch)
- Recursive Feature Elimination (RFE) with Random Forest for feature selection
- EarlyStopping callbacks to prevent overfitting during tuning
- Serializing and loading model components (model, scaler, features) for deployment
- Building a reusable prediction function for production-style inference

---

## 👤 Author

**Kadiyapu Durgesh Kumar**  
B.Tech CSE (AI & ML Minor) | VIT-AP University  
AI & Data Science Trainee | DRISHTI CPS, IIT Indore

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/durgesh-kumar-37a46a31b)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Durgeshkadiyapu)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
