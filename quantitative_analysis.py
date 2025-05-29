import pandas as pd
import numpy as np
import yfinance as yf
import csv
import os
import glob
from datetime import datetime
# from curl_cffi import requests
import time
import matplotlib.pyplot as plt
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn

def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

'''
def save_xls(list_dfs, xls_path):
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer, "sheets%s" % n)
'''

# Choose the download data
@run_once
def download_data(tickers, start_date, end_date):
    # Initialize the dataframe for close 
    close_df = pd.DataFrame()
    ## Create an empty DataFrame to store the close prices
    for ticker in tickers:
        # session = requests.Session(impersonate="chrome")
        # close_df[ticker] = yf.download(ticker, start=start_date, end=end_date, session=session)
        close_df[ticker] = yf.download(ticker, start=start_date, end=end_date)['Close']
        time.sleep(2)
    return close_df

def preprocess_data(data):
    # Split the data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    return train_data.dropna(), test_data

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

# Evaluate the accuracy of the model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return accuracy, precision, recall, f1

# Generate the prediction of the model
def generate_predictions(model, X):
    return model.predict(X)

# Generate the trade signal
def generate_signals(predictions, data):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0
    signals.loc[signals.index[1:], 'Signal'] = np.where(predictions[1:] > predictions[:-1], 1, -1)
    return signals

# Backtest the selected strategy
def backtest_strategy(signals, data):
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['Position'] = signals['Signal']
    portfolio = positions.multiply(data['Return'], axis=0)
    cumulative_returns = (1 + portfolio).cumprod()
    return cumulative_returns

# Evaluate the trade strategy
def evaluate_strategy(returns):
    total_return = returns.iloc[-1] - 1
    annualized_return = (returns.iloc[-1] ** (252 / len(returns))) - 1
    sharpe_ratio = (returns.diff().mean() / returns.diff().std()) * np.sqrt(252)
    max_drawdown = (returns / returns.cummax() - 1).min()
    return total_return, annualized_return, sharpe_ratio, max_drawdown

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # One input → one output

    def forward(self, x):
        # Applies sigmoid() to the output so it becomes a probability (0 to 1).
        return torch.sigmoid(self.linear(x))  # Apply sigmoid

def main():
    ## Define the list of tickers
    # tickers = ['SPY','BND','GLD','QQQ','VTI', 'JPM']
    tickers = ["BTC-USD"]

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
        download_data(tickers, start_date, end_date).to_excel(data_path)
        close_df = pd.read_excel(data_path)
    else:
        close_df = pd.read_excel(data_path)

    # Get the training data and test data
    train_data_dict = {} # train data set
    test_data_dict = {}  # test data set
    for ticker in tickers:
        # evaluate data
        data, momentum_20, MACD, ROC_20, RSI, slope = evaluate_data(close_df, ticker)
        print(f"Quantitative values: \nMomentum={momentum_20}, \nMACD={MACD}, \nROC_20={ROC_20}, \nRSI={RSI}, \nslope={slope}")
        # Plot the train data and test data
        plot_data_ave(data, ticker, True)
        plot_data_vol(data, ticker, True)
        
        train_data_dict[ticker], test_data_dict[ticker] = preprocess_data(data)

    # Prepare the train and test datas
    X_train = train_data_dict['BTC-USD'][['MA_50', 'MA_200', 'Volatility']].values
    y_train = np.sign(train_data_dict['BTC-USD']['Return'].values)
    X_test = test_data_dict['BTC-USD'][['MA_50', 'MA_200', 'Volatility']].values
    y_test = np.sign(test_data_dict['BTC-USD']['Return'].values)

    # Logistic Regression
    # logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
    # logreg.fit(X_train, y_train)
    X = torch.tensor(X_train, dtype=torch.float32)[:,[0]]
    y = torch.tensor((y_train+1)/2, dtype=torch.float32).unsqueeze(1)
    model = LogisticRegressionModel()
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        y_pred = model(X)                # Forward pass
        loss = criterion(y_pred, y)      # Compute loss
        optimizer.zero_grad()            # Reset gradients
        loss.backward()                  # Compute new gradients
        optimizer.step()                 # Update weights
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    with torch.no_grad():
        X_sorted, indices = torch.sort(X, dim=0)
        y_prob = model(X_sorted)
    plt.figure()
    plt.scatter(X.numpy(), y.numpy(), label="Data")
    plt.plot(X_sorted.numpy(), y_prob.numpy(), color='red', label='Logistic Curve')
    plt.show()

    '''
    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(logreg, X_test, y_test)
    print(f"Logistic Regression: \nAccuracy={accuracy}, \nPrecision={precision}, \nRecall={recall}, \nF1-score={f1}")

    # Generate predictions
    logreg_predictions = generate_predictions(logreg, X_test)

    # Generate trade signals
    logreg_signals = generate_signals(logreg_predictions, test_data_dict['BTC-USD'])

    # Compute the backtest strategy of the models
    logreg_returns = backtest_strategy(logreg_signals, test_data_dict['BTC-USD'])

    # Evaluate the trade strategy
    logreg_total_return, logreg_annualized_return, logreg_sharpe_ratio, logreg_max_drawdown = evaluate_strategy(logreg_returns)

    print("Logistic Regression:")
    print(f"Total Return: {logreg_total_return.iloc[0]}")
    print(f"Annualized Return: {logreg_annualized_return.iloc[0]}")
    print(f"Sharpe Ratio: {logreg_sharpe_ratio.iloc[0]}")
    print(f"Maximum Drawdown: {logreg_max_drawdown.iloc[0]}")
    '''

    '''
    # Random Forest Classifier
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(rf, X_test, y_test)
    print(f"\nRandom Forest: \nAccuracy={accuracy}, \nPrecision={precision}, \nRecall={recall}, \nF1-score={f1}\n\n")

    # Generate predictions
    rf_predictions = generate_predictions(rf, X_test)

    # Generate trade signals
    rf_signals = generate_signals(rf_predictions, test_data_dict['BTC-USD'])

    # Compute the backtest strategy of the models
    rf_returns = backtest_strategy(rf_signals, test_data_dict['BTC-USD'])

    # Evaluate the trade strategy
    rf_total_return, rf_annualized_return, rf_sharpe_ratio, rf_max_drawdown = evaluate_strategy(rf_returns)

    print("\nRandom Forest:")
    print(f"Total Return: {rf_total_return.iloc[0]}")
    print(f"Annualized Return: {rf_annualized_return.iloc[0]}")
    print(f"Sharpe Ratio: {rf_sharpe_ratio.iloc[0]}")
    print(f"Maximum Drawdown: {rf_max_drawdown.iloc[0]}")
    '''

if __name__ == "__main__":
    main()