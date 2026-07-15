import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df = pd.read_csv("output.csv")
def create_sequences_from_flows(df, seq_len=10, stride=5):
    sequences = []
    for src_ip in df['Source'].unique():
        src_data = df[df['Source'] == src_ip].sort_values('Time' if 'Time' in df.columns else df.columns[0])
        if len(src_data) >= seq_len:
            for i in range(0, len(src_data) - seq_len, stride):
                window = src_data.iloc[i:i+seq_len]
                seq_features = []
                for j in range(seq_len):
                    packet = window.iloc[j]
                    features = [
                        1.0,  
                        float(packet['Length']) / 10000.0,  
                        1.0 if packet['Protocol'] == 'TCP' else 0.5, 
                        1.0 
                    ]
                    seq_features.append(features)
                sequences.append(np.array(seq_features))
    return np.array(sequences)

print("Creating temporal sequences from IoT traffic data...")
X_sequences = create_sequences_from_flows(df, seq_len=10, stride=5)
print(f"Created {len(X_sequences)} sequences with shape {X_sequences[0].shape}")

scaler = MinMaxScaler(feature_range=(0, 1))
n_samples, seq_len, n_features = X_sequences.shape
X_reshaped = X_sequences.reshape(-1, n_features)
X_normalized = scaler.fit_transform(X_reshaped)
X_sequences = X_normalized.reshape(n_samples, seq_len, n_features)


X_train_real, X_test_real = train_test_split(X_sequences, test_size=0.2, random_state=42)
print(f"Train shape: {X_train_real.shape}")
print(f"Test shape: {X_test_real.shape}")

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len, l2_reg=0.01, dropout=0.2):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        self.encoder_lstm1 = nn.LSTM(input_dim, 16, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.encoder_lstm2 = nn.LSTM(16, 8, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.encoder_lstm3 = nn.LSTM(8, 4, batch_first=True)

        self.repeat_vector = nn.Linear(4, seq_len * 4)
        self.decoder_lstm1 = nn.LSTM(4, 8, batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        self.decoder_lstm2 = nn.LSTM(8, 16, batch_first=True)
        self.dropout4 = nn.Dropout(dropout)
        self.decoder_lstm3 = nn.LSTM(16, input_dim, batch_first=True)

    def forward(self, x):

        out, _ = self.encoder_lstm1(x)
        out = self.dropout1(out)
        out, _ = self.encoder_lstm2(out)
        out = self.dropout2(out)
        _, (h_n, _) = self.encoder_lstm3(out)
        bottleneck = h_n[-1] 


        repeated = self.repeat_vector(bottleneck).view(-1, self.seq_len, 4)
        out, _ = self.decoder_lstm1(repeated)
        out = self.dropout3(out)
        out, _ = self.decoder_lstm2(out)
        out = self.dropout4(out)
        out, _ = self.decoder_lstm3(out)

        return out, bottleneck

# ===== Training Function =====
def train_lstm_autoencoder(X_train, X_test, epochs=50, batch_size=32, l2_reg=0.01, dropout=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = X_train.shape[1]
    input_dim = X_train.shape[2]

    model = LSTMAutoencoder(input_dim, seq_len, l2_reg=l2_reg, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=l2_reg)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs, bottleneck = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            outputs, bottleneck = model(test_tensor.to(device))
            val_loss = criterion(outputs, test_tensor.to(device)).item()
            val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Plot training loss
    plt.figure(figsize=(10,6))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss (MSE)")
    plt.title("LSTM Autoencoder Training Loss (Real IoT Data)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    model.eval()
    with torch.no_grad():
        _, X_encoded = model(test_tensor.to(device))
    X_encoded = X_encoded.cpu().numpy()

    print(f"✓ Encoded data shape: {X_encoded.shape}")
    return model, X_encoded
print("\n" + "="*60)
print("Training LSTM Autoencoder on Real IoT Data")
print("="*60 + "\n")

model_lstm, X_encoded_lstm = train_lstm_autoencoder(
    X_train_real,
    X_test_real,
    epochs=30,
    batch_size=16,
    l2_reg=0.01,
    dropout=0.2
)

print("\n✓ LSTM Autoencoder training complete!")
print(f"✓ Encoded representation shape: {X_encoded_lstm.shape}")

# ===== Anomaly Detection on Encoded Data =====
from sklearn.preprocessing import StandardScaler

X_encoded_scaled = StandardScaler().fit_transform(X_encoded_lstm)

# Calculate reconstruction error on test set
model_lstm.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_tensor = torch.tensor(X_test_real, dtype=torch.float32)

with torch.no_grad():
    reconstructed, _ = model_lstm(test_tensor.to(device))
    reconstruction_error = torch.nn.functional.mse_loss(
        reconstructed, test_tensor.to(device), reduction='none'
    ).mean(dim=[1, 2]).cpu().numpy()

# Visualize anomalies
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
threshold = np.percentile(reconstruction_error, 90)
plt.hist(reconstruction_error, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'90th Percentile: {threshold:.4f}')
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Reconstruction Errors")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
anomaly_labels = (reconstruction_error > threshold).astype(int)
plt.scatter(range(len(reconstruction_error)), reconstruction_error, c=anomaly_labels, cmap='RdYlGn_r', alpha=0.6)
plt.axhline(threshold, color='red', linestyle='--', linewidth=2)
plt.xlabel("Sample Index")
plt.ylabel("Reconstruction Error")
plt.title(f"Anomaly Detection ({anomaly_labels.sum()} anomalies detected)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n✓ Detected {anomaly_labels.sum()} anomalous sequences out of {len(anomaly_labels)} test samples")