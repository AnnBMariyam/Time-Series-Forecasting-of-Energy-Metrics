# Time series forecasting with RNNs
## Project Overview

This project focuses on time-series forecasting of energy-related metrics using the ETTh1 dataset, which contains multivariate electricity transformer temperature measurements collected at regular intervals.

The objective is to analyze temporal patterns (trend, seasonality, autocorrelation) and build forecasting models to predict future values of the target variable (oil temperature). Both baseline and deep learning approaches are evaluated to understand their effectiveness in real-world forecasting scenarios.

## Dataset Description

Dataset: ETTh1 (Electricity Transformer Temperature)

Source: Public benchmark dataset for time-series forecasting

Frequency: Hourly

Rows: ~17,000+

## Features:

Oil temperature (target variable)

Multiple load and environmental sensor readings

Type: Multivariate time series

The dataset exhibits trend, seasonality, and non-stationarity, making it suitable for advanced forecasting techniques.

## Tools & Technologies

Python

Pandas, NumPy

Matplotlib, Seaborn

TensorFlow / Keras

Deep Learning (RNNs)

## Methodology
1. Exploratory Data Analysis

Visualized trends and seasonality

Checked stationarity of the target series

Analyzed correlations between variables

2. Feature Engineering

Time-based features

Cyclical encoding using sine and cosine transformations

Differencing to improve stationarity

3. Data Splitting

Time-aware split into:

Training

Validation

Test sets
(No random shuffling to preserve temporal order)

## Models Implemented

Naïve Baseline Model

LSTM (Long Short-Term Memory)

GRU (Gated Recurrent Unit)

Conv1D

Dilated Conv1D

Seq2Seq Encoder–Decoder (Multi-step forecasting)

Models were evaluated using Mean Absolute Error (MAE).

## Key Results & Insights

Baseline models provided a strong reference and highlighted the importance of proper evaluation.

Non-stationarity negatively impacted deep learning performance until differencing was applied.

After stationarity adjustments:

GRU and Conv1D models significantly improved forecasting accuracy.

Seq2Seq models successfully captured long-horizon patterns.

Demonstrated that data preparation and baselines are as important as model complexity.

## Visualizations

The project includes:

Time-series plots

Decomposition (trend & seasonality)

Actual vs predicted comparisons

Multi-step forecast visualizations

## Conclusion

This project demonstrates the application of time-series analysis and deep learning models to a real-world forecasting problem. It highlights the importance of:

Baseline comparisons

Proper temporal validation

Feature engineering for stationarity

Model interpretability alongside performance

The techniques used here are directly applicable to energy analytics, demand forecasting, and operational monitoring systems.
