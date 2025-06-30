# Stock Price Prediction Using Recurrent Neural Networks

This project focuses on **forecasting stock prices** for top tech companies — *Amazon (AMZN), Google (GOOGL), IBM,* and *Microsoft (MSFT)* — using historical time series data and a suite of RNN-based models, including **Simple RNN**, **LSTM**, and **GRU** architectures.

---

## Objective

Design, train, and evaluate time series prediction models to estimate future closing prices using:
- A **Simple RNN** as baseline
- A **GRU** (Gated Recurrent Unit) as an advanced temporal model

All models are tuned with key hyperparameters such as:
- `units`, `activation`, `dropout_rate`, `batch_size`, and `epochs`

---

## Dataset Summary

- **Companies:** AMZN, GOOGL, IBM, MSFT  
- **Period:** Jan 2006 – Jan 2018  
- **Features Used:** `Open`, `High`, `Low`, `Close`, `Volume`  
- **Frequency:** Daily  
- **Preprocessing:**
  - Missing value handling
  - Feature scaling using `MinMaxScaler`
  - Windowed sequence creation
  - Train-test split: 80/20 split preserving time order

---

## Model Architectures & Tuning

### Model 1: Simple RNN
- **Best Configuration:**  
  - Units: 128  
  - Activation: ReLU  
  - Dropout: 0.3  
  - Batch Size: 32  
  - Epochs: EarlyStopped at 11  

- **Performance on Test Set:**  
  - MSE: `0.000393`  
  - MAE: `0.01634`  
  - RMSE: `0.01982`  
  - R² Score: `0.9616`

---

### Model 2: GRU (Advanced RNN)
- **Best Configuration:**  
  - Units: 128  
  - Activation: tanh  
  - Dropout: 0.3  
  - Batch Size: 32  
  - Epochs: Full 15  

- **Performance on Test Set:**  
  - MSE: `0.000515`  
  - MAE: `0.01809`  
  - RMSE: `0.02270`  
  - R² Score: `0.9780`

---

## Key Insights

- **GRU architecture** showed impressive variance capture with the highest R².
- **Simple RNN** delivered lower MAE and faster convergence, making it computationally efficient.
- **Hyperparameter tuning** (especially dropout and batch size) played a crucial role in preventing overfitting and improving generalization.
- **EarlyStopping** callback helped in retaining the best weight state across both models.

---

## Visuals

The repo includes:
- Predicted vs Actual line plots for both training and test data
- Validation curves showing training vs validation loss

---

## Tech Stack

- Python 3.x
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- TensorFlow + Keras

---

## Next Steps

- Add **Bidirectional RNNs**

---

## Author

Created by [@sharma-kashish][@karthikjuluru] - feel free to contact!

---
