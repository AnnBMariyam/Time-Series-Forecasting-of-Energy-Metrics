# ⚡ Time-Series Energy Forecasting with Deep Learning (RNNs)

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-red?style=flat-square&logo=tensorflow)
![RNN](https://img.shields.io/badge/Models-LSTM%20%7C%20GRU%20%7C%20Conv1D%20%7C%20Seq2Seq-purple?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

> **How accurately can deep learning predict electricity transformer temperature — and does model complexity always win?**
> This project benchmarks 6 forecasting models on 17,000+ hourly readings, showing that **data preparation beats model complexity** every time.

---

## 📌 Project Overview

This project applies time-series analysis and deep learning to forecast oil temperature in electricity transformers using the **ETTh1 benchmark dataset** — a real-world multivariate energy dataset with trend, seasonality, and non-stationarity challenges.

Six models were built and compared — from a Naïve Baseline to a Seq2Seq Encoder-Decoder — with a key focus on proper temporal validation, stationarity handling, and multi-step forecasting. The results challenge a common assumption: **more complex models don't automatically win**.

---

## 📂 Repository Structure

```
Time_series_forecasting_with_RNNs/
│
├── Time_series_forecasting_with_RNNs.ipynb   # Full notebook: EDA, modeling, evaluation
├── dataset.png                                # Dataset overview visualization
├── Trend.png                                  # Trend & seasonality decomposition
├── First_20_days_of_OT.png                    # Target variable: first 20 days of oil temp
├── Predicted_vs_true_values.png               # Actual vs predicted comparison
├── Final_comparison_of_models.png             # Side-by-side model performance chart
├── training_and_validation_loss_lstm.png      # LSTM training/validation loss curves
└── README.md
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Name** | ETTh1 (Electricity Transformer Temperature) |
| **Source** | Public benchmark dataset for time-series forecasting |
| **Frequency** | Hourly |
| **Size** | ~17,000+ rows |
| **Target Variable** | Oil Temperature (OT) |
| **Other Features** | Multiple load and environmental sensor readings |
| **Characteristics** | Trend, seasonality, non-stationarity |

---

## 🔧 Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deep Learning | TensorFlow, Keras |
| Environment | Jupyter Notebook |

---

## ⚙️ Methodology

### 1. Exploratory Data Analysis
- Visualized long-term trend and seasonal patterns in oil temperature
- Checked stationarity of the target series (ADF test)
- Analyzed multivariate correlations between sensor readings

### 2. Feature Engineering
- Extracted time-based features (hour, day, month)
- Applied **cyclical encoding** (sine/cosine transformations) to preserve periodicity
- Applied **differencing** to address non-stationarity — this was a critical step that significantly improved deep learning model performance

### 3. Data Splitting
- Used **time-aware train/validation/test split** — no random shuffling
- Preserved temporal order to prevent data leakage and ensure realistic evaluation

---

## 🤖 Models Implemented

| Model | Type | Purpose |
|---|---|---|
| **Naïve Baseline** | Statistical | Reference benchmark |
| **LSTM** | Recurrent Neural Network | Sequential pattern learning |
| **GRU** | Recurrent Neural Network | Efficient sequence modeling |
| **Conv1D** | Convolutional Network | Local temporal pattern detection |
| **Dilated Conv1D** | Convolutional Network | Extended receptive field forecasting |
| **Seq2Seq Encoder–Decoder** | RNN | Multi-step horizon forecasting |

All models evaluated using **Mean Absolute Error (MAE)**.

---

## 📈 Key Results & Insights

### 🏆 What Worked
- After applying differencing to correct non-stationarity, **GRU and Conv1D showed significant accuracy improvements**
- **Seq2Seq Encoder-Decoder** successfully captured long-horizon patterns for multi-step forecasting
- The Naïve Baseline established a strong floor — models that couldn't beat it were quickly eliminated

### ⚠️ Critical Finding
> **Non-stationarity was the biggest performance bottleneck — not model architecture.**
> Deep learning models underperformed until differencing was applied. This demonstrates that data preparation and feature engineering matter more than choosing a complex model.

### 💡 Key Takeaway for Real-World Forecasting
In energy analytics and operational monitoring, blindly using complex models without addressing stationarity, data leakage, or proper temporal splits produces unreliable forecasts. This project shows the full disciplined workflow required to make deep learning work on real time-series data.

---

## 📸 Visualizations

| File | What It Shows |
|---|---|
| `dataset.png` | Dataset structure and feature overview |
| `Trend.png` | Trend & seasonality decomposition of oil temperature |
| `First_20_days_of_OT.png` | Target variable behavior over first 20 days |
| `Predicted_vs_true_values.png` | Best model predictions vs actual values |
| `Final_comparison_of_models.png` | MAE comparison across all 6 models |
| `training_and_validation_loss_lstm.png` | LSTM learning curve — training vs validation loss |

---

## 🌍 Real-World Applications

The techniques used in this project are directly applicable to:
- **Energy demand forecasting** — predict load and consumption patterns
- **Predictive maintenance** — detect anomalies in transformer temperature before failure
- **Operational monitoring** — real-time forecasting for grid management systems
- **Any multivariate time-series domain** — finance, supply chain, weather

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/AnnBMariyam/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn tensorflow jupyter

# 3. Run the notebook
jupyter notebook Time_series_forecasting_with_RNNs.ipynb
```

---

## 👩‍💻 Author

**Ann B Mariyam**
MS Data Analytics — University of Illinois Springfield
[LinkedIn](https://www.linkedin.com/in/ann-b-mariyam-a238a7275/) | [GitHub](https://github.com/AnnBMariyam) | annbijumariyam02@gmail.com
