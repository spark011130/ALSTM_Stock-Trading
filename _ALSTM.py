import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

### STRATEGY 

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def calculate_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_stochastic_oscillator(data, k_window=14, d_window=3):
    lowest_low = data['Low'].rolling(window=k_window).min()
    highest_high = data['High'].rolling(window=k_window).max()
    
    k_percent = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

### ALSTM MODEL

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, lstm_output):
        attn_scores = torch.tanh(self.attn_weights(lstm_output))  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # Normalize across time dimension
        context = torch.sum(attn_weights * lstm_output, dim=1)  # Weighted sum
        return context, attn_weights

class ALSTM(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.0):
        super(ALSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        context, attn_weights = self.attention(lstm_out)  # Apply attention
        output = self.fc(context)  # Final prediction
        return output, attn_weights

### DATA PROCESSING

# Load stock data
def load_data(filepath, isScaled=True):
    df = pd.read_csv(filepath)
        
    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")
    
    df['RSI'] = calculate_rsi(df)
    df['MACD'], df['Signal'] = calculate_macd(df)
    df['Middle_Band'], df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(df)
    df['ATR'] = calculate_atr(df)
    df['%K'], df['%D'] = calculate_stochastic_oscillator(df)
    df.dropna(inplace=True)
    if isScaled:
        train_size = int(len(df) * 0.8)
        scaler = MinMaxScaler()
        df.iloc[:train_size, 1:] = scaler.fit_transform(df.iloc[:train_size, 1:])  # date 제외 regularization
        df.iloc[train_size+90:, 1:] = scaler.transform(df.iloc[train_size+90:, 1:])  # date 제외 regularization
    return df, scaler

class StockDataset(Dataset):
    def __init__(self, data, seq_length=10):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.seq_length, :]  # 입력: 주가 + 지표 + 거래량
        y = self.data[idx + self.seq_length, -1]  # 출력: 다음날 주가
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Train the model
def train_model(model, train_loader, epochs=100, lr=0.001):
    model.to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):        
        total_loss=0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred, _ = model(X_batch)
            y_pred = y_pred.squeeze()

            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")
    
    torch.save(model.state_dict(), "outputs/alstm_model.pth")
    print("Model saved.")

# Evaluate model and calculate performance metrics
def evaluate_model(model, test_loader, scaler):
    model.to(device)
    model.eval()
    predictions, actuals = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred, _ = model(X_batch)
            y_pred = y_pred.squeeze().cpu().numpy()
            predictions.extend(y_pred)
            actuals.extend(y_batch.numpy())

    # Convert to numpy arrays
    actuals = np.array(actuals).reshape(-1, 1)
    predictions = np.array(predictions).reshape(-1, 1)

    # Inverse transform only 'Close' price
    min_close = scaler.data_min_[3]  # 'Close' price min
    max_close = scaler.data_max_[3]  # 'Close' price max
    actuals_original = actuals * (max_close - min_close) + min_close
    predictions_original = predictions * (max_close - min_close) + min_close

    y_test = actuals_original.flatten()
    y_pred = predictions_original.flatten()

    # Calculate returns for IC
    returns_actual = np.diff(y_test) / y_test[:-1]
    returns_pred = np.diff(y_pred) / y_pred[:-1]

    # Performance Metrics
    mse = np.mean((y_test - y_pred)**2)
    mae = np.mean(np.abs(y_test - y_pred))
    ic, _ = spearmanr(returns_pred, returns_actual)  # IC는 수익률 비교

    # ICIR (Information Coefficient Information Ratio)
    ic_series = pd.Series(returns_actual).rolling(window=10).apply(
        lambda x: spearmanr(x, pd.Series(returns_pred).iloc[x.index])[0], raw=False
    )
    icir = ic_series.mean() / ic_series.std() if ic_series.std() != 0 else np.nan

    print(f'MSE: {mse:.6f}, MAE: {mae:.6f}, IC: {ic:.6f}, ICIR: {icir:.6f}')

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual (Original Scale)')
    plt.plot(y_pred, label='Predicted (Original Scale)')
    plt.legend()
    plt.title(f'MSE: {mse:.6f}, MAE: {mae:.6f}, IC: {ic:.6f}, ICIR: {icir:.6f} Stock Prediction')
    plt.savefig("outputs/ALSTM_prediction.png", dpi=300)
    plt.show()
    

if __name__ == "__main__":
    df, scaler = load_data(filepath="inputs/bitcoin_prices.csv", isScaled=True)
    print(df.head())
    data_np = df.iloc[:, 1:].values
    split = int(len(data_np) * 0.8)
    train_data, test_data = data_np[:split], data_np[split+90:]

    seq_length = 10
    train_dataset = StockDataset(train_data, seq_length)
    test_dataset = StockDataset(test_data, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    ## OVERLAPPING 발생 문제제

    input_dim = data_np.shape[1]
    model = ALSTM(input_dim=input_dim).to(device)
    
    train_model(model, train_loader)
    
    model.load_state_dict(torch.load("outputs/alstm_model.pth"))
    evaluate_model(model, test_loader, scaler)