import torch
import torch.nn as nn
import torch.optim as optim

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
    def __init__(self, input_dim=12, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(ALSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        context, attn_weights = self.attention(lstm_out)  # Apply attention
        output = self.fc(context)  # Final prediction
        return output, attn_weights

# Example usage:
if __name__ == "__main__":
    batch_size = 16
    seq_length = 30  # Number of past days used for prediction
    input_dim = 12
    
    # Example input data (random for demonstration)
    sample_input = torch.rand(batch_size, seq_length, input_dim)
    model = ALSTM(input_dim=input_dim)
    
    # Forward pass
    output, attn_weights = model(sample_input)
    print("Output shape:", output.shape)  # Expected: (batch_size, output_dim)
    print("Attention weights shape:", attn_weights.shape)  # Expected: (batch_size, seq_len, 1)