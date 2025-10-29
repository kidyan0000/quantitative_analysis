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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
def download_close_data(ticker, start_date, end_date):
    # Initialize the dataframe for close 
    close_df = pd.DataFrame()
    ## Create an empty DataFrame to store the close prices
    session = requests.Session(impersonate="chrome")
    close_df = yf.download(ticker, start=start_date, end=end_date, session=session)['Close']
    # close_df = yf.download(ticker, start=start_date, end=end_date)['Close']
    time.sleep(2)
    return close_df
@run_once
def download_fear_and_greed_data(start_date, end_date):
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    response = requests.get(url)
    data = response.json()

    data_fear_and_greed = pd.DataFrame(data["data"])
    data_fear_and_greed["Date"] = pd.to_datetime(data_fear_and_greed["timestamp"], unit='s')
    data_fear_and_greed["value"] = data_fear_and_greed["value"].astype(int)
    # Sort oldest to newest
    data_fear_and_greed = data_fear_and_greed.sort_values("Date").reset_index(drop=True)

    # Filter the DataFrame
    data_fear_and_greed = data_fear_and_greed[
        (data_fear_and_greed["Date"] >= pd.to_datetime(start_date)) &
        (data_fear_and_greed["Date"] <= end_date)
        ]
    return data_fear_and_greed

def evaluate_data(close_df, ticker, start_date):
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
    data['momentum_20'] = momentum_20
    # MACD (Moving Average Convergence Divergence)
    # Positive MACD → upward momentum
    # Negative MACD → downward momentum
    MACD = data["EMA_12"] - data["EMA_26"]
    data['MACD'] = MACD
    # Rate of Change (ROC)
    ROC_20 = ((data["Close"] - data["Close"].shift(20)) / data["Close"].shift(20)) * 100
    data['ROC_20'] = ROC_20

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
    data['RSI'] = RSI

    # Linear Regression Slope (last 30 days)
    X = np.arange(30).reshape(-1, 1)
    y = data["Close"].tail(30).values.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0][0]
    data['slope'] = slope
    return data

def evaluate_fear_and_greed_data(fear_and_greed_df, start_date):
    fear_and_greed_data = pd.DataFrame()
    # Create the trade date
    fear_and_greed_data['Date'] = fear_and_greed_df['Date']

    # Remove missing values
    fear_and_greed_data['Index'] = fear_and_greed_df['value'].dropna()

    # Create additional features
    fear_and_greed_data['MA_50'] = fear_and_greed_data['Index'].rolling(window=50).mean()
    fear_and_greed_data['MA_200'] = fear_and_greed_data['Index'].rolling(window=200).mean()
    fear_and_greed_data["EMA_12"] = fear_and_greed_data["Index"].ewm(span=12, adjust=False).mean()
    fear_and_greed_data["EMA_26"] = fear_and_greed_data["Index"].ewm(span=26, adjust=False).mean()
    fear_and_greed_data['Volatility'] = fear_and_greed_data['Index'].rolling(window=50).std()
    
    return fear_and_greed_data

def plot_data_price(data, ticker, isSave):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Close')
    plt.plot(data['Date'], data['MA_50'], label='50-day MA')
    plt.plot(data['Date'], data['MA_200'], label='200-day MA')
    plt.title('Moving Averages '+ticker)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Save the png 
    if isSave is True:
        fig_file = ticker+'_ave.png'
        fig_path = os.path.join(sys.path[0], fig_file)
        plt.savefig(fig_path, format='png', dpi=1200)
    # Show the png
    plt.show()

def plot_data_pred(data, y_test, y_test_pred, test_rmse, ticker, isSave):
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4,1)

    ax1 = fig.add_subplot(gs[:3, 0])
    ax1.plot(data['Date'][-len(y_test):], data['Close'][-len(y_test):], label='Price Original')
    ax1.plot(data['Date'][-len(y_test):], data['MA_50'][-len(y_test):], label='Price 50-day MA')
    ax1.plot(data['Date'][-len(y_test):], data['MA_200'][-len(y_test):], label='Price 200-day MA')
    ax1.plot(data['Date'][-len(y_test_pred):], y_test_pred, label='Predicted Price')
    ax1.legend()
    plt.title('Price Prediction '+ticker)
    plt.xlabel('Date')
    plt.ylabel('Price')

    ax2 = fig.add_subplot(gs[3, 0])
    ax2.axhline(test_rmse, color='blue', linestyle='--', label='RMSE')
    ax2.plot(data['Date'][-len(y_test):], abs(y_test_pred-y_test), color='red', label='Price Error (abs)')
    ax2.legend()
    plt.title('Price Prediction Error '+ticker)
    plt.xlabel('Date')
    plt.ylabel('Price Error')

    plt.tight_layout()
    
    # Save the png 
    if isSave is True:
        fig_file = ticker+'_pred.png'
        fig_path = os.path.join(sys.path[0], fig_file)
        plt.savefig(fig_path, format='png', dpi=1200)
    # Show the png
    plt.show()

def plot_data(data, key, ticker, isSave):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data[key])
    plt.title(key+' '+ticker)
    plt.xlabel('Date')
    plt.ylabel(key)
    
    # Save the png 
    if isSave is True:
        fig_file = ticker+'_'+key+'.png'
        fig_path = os.path.join(sys.path[0], fig_file)
        plt.savefig(fig_path, format='png', dpi=1200)
    # Show the png
    plt.show()

def ML_1I1O(data, scaler):
    # Normalize the datas
    price_scaled = scaler['Price'].fit_transform(data[['Close']].dropna())
    seq_length = 30
    data_sets_price = []
    
    for i in range(len(price_scaled)-seq_length):
        data_sets_price.append(price_scaled[i:i+seq_length])
    train_size = int(len(data_sets_price) * 0.8)
    
    data_sets_price = np.array(data_sets_price)
    # Split the data into training and testing sets 
    X_train_data = data_sets_price[:train_size, :-1, :]
    y_train_data = data_sets_price[:train_size, -1, :]
    X_test_data  = data_sets_price[train_size:train_size+seq_length, :-1, :]
    y_test_data  = data_sets_price[train_size:train_size+seq_length, -1, :]

    X_train = torch.from_numpy(X_train_data).type(torch.Tensor).to(device)
    y_train = torch.from_numpy(y_train_data).type(torch.Tensor).to(device)
    X_test = torch.from_numpy(X_test_data).type(torch.Tensor).to(device)
    y_test = torch.from_numpy(y_test_data).type(torch.Tensor).to(device)

    model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)
    criterion = nn.MSELoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    num_epochs = 200

    for epoch in range(num_epochs):
        y_train_pred = model(X_train)
        loss = criterion(y_train_pred, y_train)
        if epoch%25 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    y_test_pred = model(X_test)
    return y_train_pred, y_train, y_test_pred, y_test

def ML_MI1O(data, fear_and_greed_data, scaler):
    def hybrid_loss(y_pred, y_true, lagged_fear, lagged_rsi, lagged_macd):
        # y_pred, y_true: [batch_size, 10]
        # lagged inputs: [batch_size, 1]

        # Main MSE loss across all 10 days
        mse = nn.functional.mse_loss(y_pred, y_true)

        # Directional Loss
        # Compute direction using day 10 minus day 1
        dir_true = torch.sign(y_true[:, -1] - y_true[:, 0])
        dir_pred = torch.sign(y_pred[:, -1] - y_pred[:, 0])
        wrong_dir = (dir_true != dir_pred).float()
        
        # Fear-based direction penalty
        fear_weight = lagged_fear.view(-1) / 100.0  # Normalize
        dir_loss = torch.mean(wrong_dir * fear_weight)

        # RSI Penalty
        # Assume RSI is constant across the 10 days
        high_rsi = (lagged_rsi.view(-1) > 0.7).float()
        low_rsi = (lagged_rsi.view(-1) < 0.3).float()
        
        price_diff = y_pred[:, -1] - y_pred[:, 0]  # price direction
        price_increase = (price_diff > 0).float()
        price_decrease = (price_diff < 0).float()

        penalty_high = high_rsi * price_increase
        penalty_low  = low_rsi  * price_decrease
        rsi_loss = (penalty_high + penalty_low).mean()

        # MACD Penalty
        macd_pos = (lagged_macd.view(-1) > 0).float()
        macd_neg = (lagged_macd.view(-1) < 0).float()
        
        penalty_macd = macd_pos * price_decrease + macd_neg * price_increase
        macd_loss = penalty_macd.mean()

        # Trend Alignment Penalty
        trend_true = torch.sign(y_true[:, 1:] - y_true[:, :-1])
        trend_pred = torch.sign(y_pred[:, 1:] - y_pred[:, :-1])
        trend_mismatch = (trend_true != trend_pred).float()
        trend_loss = trend_mismatch.mean()

        # Under-Prediction Penalty
        under_prediction = (y_pred < y_true).float()
        under_penalty = ((y_true - y_pred) * under_prediction).mean()

        # Total loss
        total_loss = (
            mse
            + 0.1 * dir_loss
            + 0.05 * rsi_loss
            + 0.05 * macd_loss
            + 0.1 * trend_loss
            + 0.1 * under_penalty
        )
        return total_loss
    
    def criterion(y_pred, y_true, fear_input, rsi_input, macd_input):
        return hybrid_loss(y_pred, y_true, fear_input, rsi_input, macd_input)

    # Clean the datas
    data["RSI"] = data["RSI"].shift(3) # Lag the RSI by 3 
    data["MACD"] = data["MACD"].shift(3) # Lag the MACD by 3 
    fear_and_greed_data["Index"] = fear_and_greed_data["Index"].shift(3) # Lag the Fear Index Input by 3
    
    # filter_date = max(data['Date'].loc[0], fear_and_greed_data['Date'].loc[0])
    filter_date = '2018-04-17' # there are some missing datas before 04.17
    data = data[data['Date'] >= filter_date].reset_index(drop=True)
    fear_and_greed_data = fear_and_greed_data[fear_and_greed_data['Date'] >= filter_date].reset_index(drop=True)

    # Scaler datas 
    price_scaled = scaler['Price'].fit_transform(data[['Close']].dropna())
    rsi_scaled = scaler['RSI'].fit_transform(data[['RSI']].dropna())
    macd_scaled = scaler['MACD'].fit_transform(data[['MACD']].dropna())
    fear_index_scaled = scaler['Fear and Greed'].fit_transform(fear_and_greed_data[['Index']].dropna())

    data_sets = {
        'Price': [],
        'Fear and Greed': [],
        'RSI': [],
        'MACD': [],
        'Predicted Price': []
    }
    
    seq_length = 60
    output_horizon = 5
    for i in range(len(price_scaled) - seq_length - output_horizon + 1):
        data_sets['Price'].append(price_scaled[i:i+seq_length])   # MA_50 sequence
        data_sets['RSI'].append(rsi_scaled[i + seq_length - 1])  # RSI at last step
        data_sets['MACD'].append(macd_scaled[i + seq_length - 1])  
        data_sets['Fear and Greed'].append(fear_index_scaled[i+seq_length-1])  # Volatility at last step

        data_sets['Predicted Price'].append(price_scaled[i+seq_length:i+seq_length+output_horizon]) # Y data
    
    train_size = int(len(data_sets['Price']) * 0.8)

    for key in data_sets:
        data_sets[key] = np.array(data_sets[key])

    Train = {
        'Price': data_sets['Price'][:train_size],
        'Fear and Greed': data_sets['Fear and Greed'][:train_size, -1].reshape(-1, 1),
        'RSI': data_sets['RSI'][:train_size].reshape(-1, 1),
        'MACD': data_sets['MACD'][:train_size].reshape(-1, 1),
    }

    Test = {
        'Price': data_sets['Price'][train_size:],
        'Fear and Greed': data_sets['Fear and Greed'][train_size:, -1].reshape(-1, 1),
        'RSI': data_sets['RSI'][train_size:].reshape(-1, 1),
        'MACD': data_sets['MACD'][train_size:].reshape(-1, 1),
    }
   
    # Convert to Torch tensors
    for key in Train:
        Train[key] = torch.from_numpy(Train[key]).float().to(device)
    for key in Test:
        Test[key] = torch.from_numpy(Test[key]).float().to(device)

    y_train = torch.from_numpy(data_sets['Predicted Price'][:train_size]).type(torch.Tensor).squeeze(-1).to(device)
    y_test = torch.from_numpy(data_sets['Predicted Price'][train_size:]).type(torch.Tensor).squeeze(-1).to(device)
     
    model = ResidualLSTM(price_input_dim=1, aux_input_dim=3, hidden_dim=64, num_layers=3, output_length=output_horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 500

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        y_train_pred = model(Train['Price'], torch.cat([Train['Fear and Greed'], Train['RSI'], Train['MACD']],1))
        # loss = criterion(y_train_pred, y_train)
        loss = criterion(y_train_pred, y_train, Train['Fear and Greed'], Train['RSI'], Train['MACD'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    model.eval()

    y_test_pred = model(Test['Price'], torch.cat([Test['Fear and Greed'], Test['RSI'], Test['MACD']],1))
    return y_train_pred, y_train, y_test_pred, y_test

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

class MultiInputPredictionModel(nn.Module):
    def __init__(self, price_input_dim, aux_input_dim, hidden_dim, num_layers, output_dim):
        super(MultiInputPredictionModel, self).__init__()

        # LSTM for price sequence
        self.lstm_price = nn.LSTM(price_input_dim, hidden_dim, num_layers, batch_first=True)

        # Linear layer for auxiliary inputs (RSI and Fear Index)
        self.fc_aux     = nn.Linear(aux_input_dim, hidden_dim)

        # Output layer after combining both paths
        self.fc_out     = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, price_input, aux_input):
        # Initialize LSTM hidden states
        h0 = torch.zeros(self.lstm_price.num_layers, price_input.size(0), self.lstm_price.hidden_size, device=price_input.device)
        c0 = torch.zeros(self.lstm_price.num_layers, price_input.size(0), self.lstm_price.hidden_size, device=price_input.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm_price(price_input, (h0, c0))
        price_feat = lstm_out[:, -1, :]  # Last time step

        # Process RSI and Fear index
        aux_feat = self.fc_aux(aux_input)

        # Combine and output
        combined = torch.cat((price_feat, aux_feat), dim=1)
        out = self.fc_out(combined)
        return out
    
class ResidualLSTM(nn.Module):
    def __init__(self, price_input_dim, aux_input_dim, hidden_dim, output_length, num_layers, dropout_prob=0.3):
        super(ResidualLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers  # Keep it simple

        self.lstm = nn.LSTM(input_size=price_input_dim,
                            hidden_size=hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=dropout_prob)

        # Auxiliary indicators (e.g., RSI, MACD, Fear Index)
        self.fc_aux = nn.Sequential(
            nn.Linear(aux_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # Output layer to produce 10-day forecast
        self.combined_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_length)  # <- output_length = 10
        )

    def forward(self, price_input, aux_input):
        batch_size = price_input.size(0)
        device = price_input.device

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

        lstm_out, _ = self.lstm(price_input, (h0, c0))
        price_feat = lstm_out[:, -1, :]  # Feature from last time step

        aux_feat = self.fc_aux(aux_input)

        combined = torch.cat((price_feat, aux_feat), dim=1)
        predicted_delta_sequence = self.combined_fc(combined)  # shape: (batch, 10)

        last_price = price_input[:, -1, 0:1]  # Use last price as reference
        
        return predicted_delta_sequence + last_price  # Predict future prices

def main():
    global device 
    print(torch.cuda.is_available())
    time.sleep(2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    ticker = "BTC-USD"
    # ticker = "JPM"

    start_date = '2000-01-01'
    end_date = '2025-05-27'
    data_file = '2025527_stock_prices.xlsx'
    data_fear_and_greed_file = '2025527_fear_and_greed.xlsx'
    '''
    start_date = '2000-01-01'
    end_date = datetime.today()
    data_file = str(end_date.year)+str(end_date.month)+str(end_date.day)+'_stock_prices.xlsx'
    '''


    # Check if the dataframe already exists
    data_path = os.path.join(sys.path[0], data_file)
    if not os.path.isfile(data_path):
        # Export data to excel
        download_close_data(ticker, start_date, end_date).to_excel(data_path)
        close_df = pd.read_excel(data_path)
    else:
        close_df = pd.read_excel(data_path)

    data_fear_and_greed_path = os.path.join(sys.path[0], data_fear_and_greed_file)    
    if not os.path.isfile(data_fear_and_greed_path):
        # Export data to excel
        fear_and_greed_df = download_fear_and_greed_data(start_date, end_date).to_excel(data_fear_and_greed_path)
        fear_and_greed_df = pd.read_excel(data_fear_and_greed_path)
    else:
        fear_and_greed_df = pd.read_excel(data_fear_and_greed_path)

    # Evaluate the datas from dataframe
    data = evaluate_data(close_df, ticker, start_date)
    print(
        f"Quantitative values: \nMomentum={data['momentum_20'].iloc[-1]}, \nMACD={data['MACD'].iloc[-1]}, \nROC_20={data['ROC_20'].iloc[-1]}, \nRSI={data['RSI'].iloc[-1]}, \nslope={data['slope'].iloc[-1]}"
        )
    # Plot the train data and test data
    plot_data_price(data, ticker, False)

    fear_and_greed_data = evaluate_fear_and_greed_data(fear_and_greed_df, start_date)
    plot_data(fear_and_greed_data, 'Index', ticker, False)
    plot_data(data, 'RSI', ticker, False)
    plot_data(data, 'MACD', ticker, False)

    # Prepare the training datas and test datas
    scaler_price = StandardScaler()
    scaler_vol   = StandardScaler()
    scaler_fear_and_greed  = MinMaxScaler(feature_range=(0, 1))
    scaler_RSI = MinMaxScaler(feature_range=(0, 1))
    scaler_MACD = StandardScaler()
    scaler = {
        'Price':scaler_price, 
        'Volatility': scaler_vol,
        'Fear and Greed': scaler_fear_and_greed,
        'RSI': scaler_RSI,
        'MACD': scaler_MACD
        }

    y_train_pred, y_train, y_test_pred, y_test = ML_1I1O(data=data, scaler=scaler) # Price as one single input
    # y_train_pred, y_train, y_test_pred, y_test = ML_MI1O(data=data, fear_and_greed_data=fear_and_greed_data, scaler=scaler) # Price, Fear Index and RSI as three input
    
    y_train_pred = scaler['Price'].inverse_transform(y_train_pred.detach().cpu().numpy())
    y_train = scaler['Price'].inverse_transform(y_train.detach().cpu().numpy())
    y_test_pred = scaler['Price'].inverse_transform(y_test_pred.detach().cpu().numpy())
    y_test = scaler['Price'].inverse_transform(y_test.detach().cpu().numpy())

    train_rmse = root_mean_squared_error(y_train[:, 0], y_train_pred[:, 0])
    test_rmse = root_mean_squared_error(y_test[:, 0], y_test_pred[:, 0])

    print('train RMSE: ', train_rmse)
    print('test RMSE: ', test_rmse)

    plot_data_pred(data, y_test, y_test_pred, test_rmse, ticker, True)

if __name__ == "__main__":
    main()