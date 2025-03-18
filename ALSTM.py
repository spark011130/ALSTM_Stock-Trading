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
np.random.seed(0)

### INDICATORS 
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
        df.iloc[:train_size, 1:] = scaler.fit_transform(df.iloc[:train_size, 1:])  # regularization except date
        df.iloc[train_size+90:, 1:] = scaler.transform(df.iloc[train_size+90:, 1:])  # regularization except date
    return df, scaler

class StockDataset(Dataset):
    def __init__(self, data, seq_length=10):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.seq_length, :]  # Input: Today-seq_length ~ today stock price + strategies + volume
        y = self.data[idx + self.seq_length, -1]  # Output: Next day stock price
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
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {(epoch+1)}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")
    
    torch.save(model.state_dict(), "outputs/alstm_model.pth")
    print("Model saved.")

def evaluate_trading_strategy(y_test, y_pred, threshold=0.03, risk_free_rate=0.02, N=365):
    """
    Evaluate Trend Following & Event-Driven Trading strategies + Key Metrics Calculation
    
    :param y_test: Actual price array
    :param y_pred: Predicted price array
    :param threshold: Threshold for event-driven trading (default 5%)
    :param risk_free_rate: Risk-free interest rate (used for Sharpe Ratio, default 2%)
    :param N: Number of trading days, 365 for crypto (Bitcoin), 252 for stock markets
    
    :return: Dictionary containing Cumulative Return, AR (%), AV (%), SR, and MDD (%)
    """
    # Calculate daily returns
    returns = np.diff(y_test) / y_test[:-1]  # Actual returns
    pred_returns = np.diff(y_pred) / y_pred[:-1]  # Predicted returns

    df = pd.DataFrame(returns)
    summary = df.describe()
    print(summary)
    
    # Trend-following strategy: If y_pred > y_pred, go long (buy)
    trend_strategy = np.where(pred_returns > 0, 1, 0)  # Buy (1), no Short.
    print(f"trend strategy is traded {np.sum(trend_strategy)} times in {len(y_pred)} days.")
    trend_returns = trend_strategy * returns  # Returns based on the trading strategy
    
    # Event-driven strategy: If y_pred exceeds the threshold, take the buy position.
    event_strategy = np.where(pred_returns > threshold, 1, 0)
    print(f"event strategy is traded {np.sum(event_strategy)} times in {len(y_pred)} days.")
    event_returns = event_strategy * returns  # Returns based on the event-driven strategy

    # Calculate cumulative returns (Keep as arrays)
    cumulative_return_trend = np.cumprod(1 + trend_returns) - 1  # Trend-following strategy
    cumulative_return_event = np.cumprod(1 + event_returns) - 1  # Event-driven strategy

    # Annualized Return (AR) - Convert to percentage
    ar_trend = float(np.mean(trend_returns) * N * 100)  # Convert to %
    ar_event = float(np.mean(event_returns) * N * 100)  # Convert to %

    # Annualized Volatility (AV) - Convert to percentage
    av_trend = float(np.std(trend_returns) * np.sqrt(N) * 100)  # Convert to %
    av_event = float(np.std(event_returns) * np.sqrt(N) * 100)  # Convert to %

    # Sharpe Ratio (SR)
    sr_trend = float((ar_trend - risk_free_rate * 100) / av_trend) if av_trend != 0 else np.nan
    sr_event = float((ar_event - risk_free_rate * 100) / av_event) if av_event != 0 else np.nan

    # Maximum Drawdown (MDD) - Convert to percentage
    def calculate_mdd(cumulative_returns):
        if len(cumulative_returns) == 0:  # Check if array is empty
            return 0.0
        peak = np.maximum.accumulate(cumulative_returns)  # Track the highest value reached
        drawdown = (cumulative_returns - peak) / peak  # Drawdown calculation
        return abs(float(np.min(drawdown) * 100))  # Convert to %

    mdd_trend = calculate_mdd(cumulative_return_trend)
    mdd_event = calculate_mdd(cumulative_return_event)

    # Output results
    results = {
        "Trend Following": {
            "Cumulative Return": float(cumulative_return_trend[-1] * 100),  # Convert to %
            "AR (%)": ar_trend,
            "AV (%)": av_trend,
            "SR": sr_trend,
            "MDD (%)": mdd_trend
        },
        "Event-Driven Trading": {
            "Cumulative Return": float(cumulative_return_event[-1] * 100),  # Convert to %
            "AR (%)": ar_event,
            "AV (%)": av_event,
            "SR": sr_event,
            "MDD (%)": mdd_event
        }
    }
    
    return results

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
    ic, _ = spearmanr(returns_pred, returns_actual)

    # ICIR (Information Coefficient Information Ratio)
    ic_series = pd.Series(returns_actual).rolling(window=10).apply(
        lambda x: spearmanr(x, pd.Series(returns_pred).iloc[x.index])[0], raw=False
    )
    icir = ic_series.mean() / ic_series.std() if ic_series.std() != 0 else np.nan

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual (Original Scale)')
    plt.plot(y_pred, label='Predicted (Original Scale)')
    plt.legend()
    plt.title('LSTM Stock Prediction')
    plt.savefig("outputs/LSTM_prediction.png", dpi=300)
    plt.show()

    metrices = [mse, mae, ic, icir]
    return y_pred, y_test, metrices

def print_and_save_evaluation_results(results, metrics, filename="outputs/ALSTM_trading_strategy_evaluation.csv"):
    """
    Formats, prints, and saves trading strategy evaluation results, including additional evaluation metrics.
    
    :param results: Dictionary containing strategy metrics
    :param metrics: List containing additional evaluation metrics [MSE, MAE, IC, ICIR]
    :param filename: Name of the file to save results (default: outputs/ALSTM_trading_strategy_evaluation.csv)
    """
    # Convert results dictionary to DataFrame
    df = pd.DataFrame(results).T  # Transpose for a structured tabular format

    # Round numerical values for better readability
    df = df.round(2)

    # Append additional evaluation metrics to the DataFrame
    additional_metrics = pd.DataFrame({
        "MSE": [metrics[0]],
        "MAE": [metrics[1]],
        "IC": [metrics[2]],
        "ICIR": [metrics[3]]
    }, index=["Evaluation Metrics"])

    # Concatenate strategy results and evaluation metrics
    df = pd.concat([df, additional_metrics])

    # Print formatted DataFrame
    print("\n📊 Trading Strategy Evaluation Results 📊\n")
    print(df.to_string())
    print("\n🔹 Cumulative Return: Total return (%)")
    print("🔹 AR (Annualized Return): Yearly return (%)")
    print("🔹 AV (Annualized Volatility): Yearly volatility (%)")
    print("🔹 SR (Sharpe Ratio): Risk-adjusted return (Higher is better)")
    print("🔹 MDD (Maximum Drawdown): Maximum loss from peak (%)")
    print("\n📌 Additional Evaluation Metrics:")
    print("🔹 MSE (Mean Squared Error): Measures prediction error (Lower is better)")
    print("🔹 MAE (Mean Absolute Error): Measures average absolute error (Lower is better)")
    print("🔹 IC (Information Coefficient): Measures predictive power (Higher is better)")
    print("🔹 ICIR (Information Coefficient Information Ratio): Measures consistency of IC")

    # Save results to a CSV file
    df.to_csv(filename, index=True)
    print(f"\n✅ Results saved successfully to '{filename}'")

    return df

if __name__ == "__main__":
    df, scaler = load_data(filepath="inputs/bitcoin_prices.csv", isScaled=True)
    print(df.head())
    data_np = df.iloc[:, 1:].values
    split = int(len(data_np) * 0.8)
    # Adding 90, to prevent overlapping
    train_data, test_data = data_np[:split], data_np[split+90:]

    seq_length = 10
    train_dataset = StockDataset(train_data, seq_length)
    test_dataset = StockDataset(test_data, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_dim = data_np.shape[1]
    model = ALSTM(input_dim=input_dim).to(device)
    
    train_model(model, train_loader)
    
    model.load_state_dict(torch.load("outputs/alstm_model.pth"))
    y_pred, y_test, metrices = evaluate_model(model, test_loader, scaler)

    results = evaluate_trading_strategy(y_test, y_pred)
    print_and_save_evaluation_results(results, metrices)