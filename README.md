# 🧠 AI Stock & Crypto Prediction Model

An advanced AI-powered system that predicts **stock** and **cryptocurrency** prices using **deep learning (PyTorch & sklearn)**, **technical indicators**, and **market sentiment (Fear & Greed Index)**.  

This project combines **quantitative analysis**, **machine learning**, and **sentiment data** to forecast market trends and visualize results.

---

## 🚀 Key Features

- 📊 **Automatic stock & crypto data download** via [Yahoo Finance](https://finance.yahoo.com)
- 💹 **Fear & Greed Index sentiment data** integration from [alternative.me](https://alternative.me/crypto/fear-and-greed-index/)
- 🧮 Built-in **technical indicators**:
  - Moving Averages (MA50, MA200)  
  - Exponential Moving Averages (EMA12, EMA26)  
  - RSI (Relative Strength Index)  
  - MACD (Moving Average Convergence Divergence)  
  - Momentum, ROC, and Volatility  
- 🧠 Deep learning models (LSTM / Residual LSTM)
- ⚙️ Supports:
  - **Single-input model (price only)**  
  - **Multi-input model (price + sentiment + indicators)**  
- 📈 Auto-generated **charts & predictions**

---

## 🧩 Tech Stack

| Category | Technology |
|-----------|-------------|
| **Language** | Python 3.10+ |
| **Machine Learning** | PyTorch |
| **Data Processing** | pandas, numpy |
| **APIs** | yfinance, alternative.me |
| **Visualization** | matplotlib |
| **Preprocessing** | scikit-learn |
| **Optimization** | numba |

---

## 🗂️ Project Structure


```
├── main.py                         # Main AI prediction script
├── 2025527_stock_prices.xlsx       # Cached stock/crypto data (auto-created)
├── 2025527_fear_and_greed.xlsx     # Cached Fear & Greed Index data (auto-created)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── outputs/                        # Generated charts & prediction plots
```


---

## ⚡ Installation

```bash

# Clone the repository
git clone https://github.com/kidyan0000/quantitative_analysis.git

cd quantitative_analysis

python quantitative_analysis_pytorch.py
# By default, the model predicts Bitcoin (BTC-USD) prices from 2000–2025 using an LSTM model. 
# To switch to a different asset, open the script and edit:
# ticker = "AAPL"   # Example: Apple stock

```

## 🔧 Configuration

```python
ticker = "BTC-USD"      # Target stock or crypto symbol
start_date = "2000-01-01"
end_date = "2025-05-27"
To select model type:

```python
# Single-input model (price only)
y_train_pred, y_train, y_test_pred, y_test = ML_1I1O(data, scaler)

# Multi-input model (price + sentiment + RSI + MACD)
y_train_pred, y_train, y_test_pred, y_test = ML_MI1O(data, fear_and_greed_data, scaler)

```
## ⚠️ Disclaimer
This project is for educational and research purposes only.
It does not constitute financial advice.
Use at your own risk — markets are unpredictable.

## 🤝 Contributing
Contributions, feedback, and feature requests are welcome!
Please open an issue or submit a pull request if you'd like to collaborate.

## 📜 License
This project is licensed under the MIT License.

## 🌐 Connect
GitHub: @kidyan0000

LinkedIn: https://www.linkedin.com/in/sikang-yan/

Website: https://www.yansikang.com
