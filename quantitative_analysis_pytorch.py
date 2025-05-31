import pandas as pd
import numpy as np
import yfinance as yf
import csv
import os
import glob
from datetime import datetime
from curl_cffi import requests
import time
import matplotlib.pyplot as plt
import sys

from numba.cuda.simulator.cudadrv.nvvm import is_available
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, root_mean_squared_error

import torch
import torch.nn as nn

def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

# Choose the download data
@run_once
def download_data(ticker, start_date, end_date):
    # Initialize the dataframe for close 
    close_df = pd.DataFrame()
    ## Create an empty DataFrame to store the close prices
    session = requests.Session(impersonate="chrome")
    close_df = yf.download(ticker, start=start_date, end=end_date, session=session)['Close']
    # close_df = yf.download(ticker, start=start_date, end=end_date)['Close']
    time.sleep(2)
    return close_df

def evaluate_data(close_df, ticker):
    data = pd.DataFrame()
    # Create the trade date
    data['Date'] = close_df['Date']

    # Remove missing values
    data['Close'] = close_df[ticker].dropna()

    # Calculate daily returns
    data['Return'] = data['Close'].pct_change()

    # Create additional features
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data['Volatility'] = data['Return'].rolling(window=50).std()

    # Momentum last 20 days
    momentum_20 = data["Close"] - data["Close"].shift(20)
    # MACD (Moving Average Convergence Divergence)
    # Positive MACD → upward momentum
    # Negative MACD → downward momentum
    MACD = data["EMA_12"] - data["EMA_26"]
    # Rate of Change (ROC)
    ROC_20 = ((data["Close"] - data["Close"].shift(20)) / data["Close"].shift(20)) * 100
    
    # Relative Strength Index (RSI) 
    # RSI > 70: Overbought (may indicate reversal) 
    # RSI < 30: Oversold
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + rs))

    # Linear Regression Slope (last 30 days)
    X = np.arange(30).reshape(-1, 1)
    y = data["Close"].tail(30).values.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0][0]
    return data, momentum_20.iloc[-1], MACD.iloc[-1], ROC_20.iloc[-1], RSI.iloc[-1], slope

def plot_data_ave(data, fig_name, isSave):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Close')
    plt.plot(data['Date'], data['MA_50'], label='50-day MA')
    plt.plot(data['Date'], data['MA_200'], label='200-day MA')
    plt.title('Moving Averages '+fig_name)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Save the png 
    if isSave is True:
        fig_file = fig_name+'_ave.png'
        fig_path = os.path.join(sys.path[0], fig_file)
        plt.savefig(fig_path, format='png', dpi=1200)
    # Show the png
    plt.show()

def plot_data_vol(data, fig_name, isSave):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Volatility'])
    plt.title('Volatility '+fig_name)
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    
    # Save the png 
    if isSave is True:
        fig_file = fig_name+'_vol.png'
        fig_path = os.path.join(sys.path[0], fig_file)
        plt.savefig(fig_path, format='png', dpi=1200)
    # Show the png
    plt.show()

def plot_data_pred(data, y_test, y_test_pred, test_rmse, fig_name, isSave):
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4,1)

    ax1 = fig.add_subplot(gs[:3, 0])
    ax1.plot(data['Date'][-len(y_test):], y_test, label='Price')
    ax1.plot(data['Date'][-len(y_test_pred):], y_test_pred, label='Predicted Price')
    ax1.legend()
    plt.title('Price Prediction '+fig_name)
    plt.xlabel('Date')
    plt.ylabel('Price')

    ax2 = fig.add_subplot(gs[3, 0])
    ax2.axhline(test_rmse, color='blue', linestyle='--', label='RMSE')
    ax2.plot(data['Date'][-len(y_test):], abs(y_test_pred-y_test), color='red', label='Price Error (abs)')
    ax2.legend()
    plt.title('Price Prediction Error '+fig_name)
    plt.xlabel('Date')
    plt.ylabel('Price Error')

    plt.tight_layout()
    
    # Save the png 
    if isSave is True:
        fig_file = fig_name+'_pred.png'
        fig_path = os.path.join(sys.path[0], fig_file)
        plt.savefig(fig_path, format='png', dpi=1200)
    # Show the png
    plt.show()

def prepare_data(data, scaler):
    # Normalize the datas
    data = scaler.fit_transform(data[['Close']])
    seq_length = 30
    data_sets = []
    
    for i in range(len(data)-seq_length):
        data_sets.append(data[i:i+seq_length])
    train_size = int(len(data_sets) * 0.8)
    
    data_sets = np.array(data_sets)
    # Split the data into training and testing sets 
    X_train_data = data_sets[:train_size, :-1, :]
    y_train_data = data_sets[:train_size, -1, :]
    X_test_data  = data_sets[train_size:, :-1, :]
    y_test_data  = data_sets[train_size:, -1, :]

    return X_train_data, y_train_data, X_test_data, y_test_data


class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out

def main():
    global device 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    ticker = "BTC-USD"
    # ticker = "JPM"

    start_date = '2000-01-01'
    end_date = '2025-05-27'
    data_file = '2025527_stock_prices.xlsx'
    '''
    start_date = '2000-01-01'
    end_date = datetime.today()
    data_file = str(end_date.year)+str(end_date.month)+str(end_date.day)+'_stock_prices.xlsx'
    '''
    
    # Check if the dataframe already exists
    data_path = os.path.join(sys.path[0], data_file)
    if not os.path.isfile(data_path):
        # Export data to excel
        download_data(ticker, start_date, end_date).to_excel(data_path)
        close_df = pd.read_excel(data_path)
    else:
        close_df = pd.read_excel(data_path)

    # Evaluate the datas from dataframe
    data, momentum_20, MACD, ROC_20, RSI, slope = evaluate_data(close_df, ticker)
    print(f"Quantitative values: \nMomentum={momentum_20}, \nMACD={MACD}, \nROC_20={ROC_20}, \nRSI={RSI}, \nslope={slope}")
    # Plot the train data and test data
    plot_data_ave(data, ticker, True)
    plot_data_vol(data, ticker, True)

    # Prepare the training datas and test datas
    scaler = StandardScaler()

    X_train_data, y_train_data, X_test_data, y_test_data = prepare_data(data, scaler)
    X_train = torch.from_numpy(X_train_data).type(torch.Tensor).to(device)
    y_train = torch.from_numpy(y_train_data).type(torch.Tensor).to(device)
    X_test = torch.from_numpy(X_test_data).type(torch.Tensor).to(device)
    y_test = torch.from_numpy(y_test_data).type(torch.Tensor).to(device)

    model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)
    criterion = nn.MSELoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    num_epochs = 200

    for i in range(num_epochs):
        y_train_pred = model(X_train)
        loss = criterion(y_train_pred, y_train)
        if i%25 == 0:
            print('epoch: ', i, 'loss: ', loss.item())
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    y_test_pred = model(X_test)
    
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
    y_train = scaler.inverse_transform(y_train.detach().cpu().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
    y_test = scaler.inverse_transform(y_test.detach().cpu().numpy())

    train_rmse = root_mean_squared_error(y_train[:, 0], y_train_pred[:, 0])
    test_rmse = root_mean_squared_error(y_test[:, 0], y_test_pred[:, 0])

    print('train RMSE: ', train_rmse)
    print('test RMSE: ', test_rmse)

    plot_data_pred(data, y_test, y_test_pred, test_rmse, ticker, True)

if __name__ == "__main__":
    main()