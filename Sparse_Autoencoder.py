import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


df = pd.read_csv("output.csv")
unique_ips = pd.unique(df[['Source', 'Destination']].values.ravel('K'))
ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}
num_nodes = len(unique_ips)

node_features = np.zeros((num_nodes, 4))
for ip, idx in ip_to_idx.items():
    df_src = df[df['Source'] == ip]
    packets_sent = len(df_src)
    avg_len_sent = df_src['Length'].mean() if packets_sent > 0 else 0
    unique_dests = df_src['Destination'].nunique()
    tcp_ratio = (df_src['Protocol'] == 'TCP').mean() if packets_sent > 0 else 0
    node_features[idx] = [packets_sent, avg_len_sent, unique_dests, tcp_ratio]

print(f"✓ Created node features with shape: {node_features.shape}")

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(node_features)

X_train_1, X_test_1 = train_test_split(X_scaled, test_size=0.2, random_state=42)
print(f"✓ Train shape: {X_train_1.shape}")
print(f"✓ Test shape: {X_test_1.shape}")

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, l1_reg=0.01):
        super(SparseAutoencoder, self).__init__()
        self.l1_reg = l1_reg

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SparseAutoencoder(input_dim=X_train_1.shape[1], l1_reg=0.01).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
train_tensor = torch.tensor(X_train_1, dtype=torch.float32)
test_tensor = torch.tensor(X_test_1, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=16, shuffle=True)

epochs = 50
train_losses, val_losses = [], []
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch, in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs, encoded = model(batch)
        mse_loss = criterion(outputs, batch)
        l1_loss = model.l1_reg * torch.mean(torch.abs(encoded))
        loss = mse_loss + l1_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    model.eval()
    with torch.no_grad():
        outputs, encoded = model(test_tensor.to(device))
        mse_val = criterion(outputs, test_tensor.to(device)).item()
        l1_val = model.l1_reg * torch.mean(torch.abs(encoded)).item()
        val_loss = mse_val + l1_val
        val_losses.append(val_loss)

    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_sparse_ae.pth')

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

# ===== Plot Training Loss =====
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), train_losses, marker='o', label='Train Loss', color='blue', linewidth=2)
plt.plot(range(1, epochs+1), val_losses, marker='s', label='Validation Loss', color='red', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Sparse Autoencoder Loss (MSE + L1)")
plt.title("Sparse Autoencoder Training on Real IoT Data")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.yscale('log')  
plt.tight_layout()
plt.show()

# ===== Load Best Model and Encode Test Data =====
model.load_state_dict(torch.load('best_sparse_ae.pth'))
model.eval()

with torch.no_grad():
    outputs, X_encoded_1 = model(test_tensor.to(device))
    reconstruction_error = torch.nn.functional.mse_loss(
        outputs, test_tensor.to(device), reduction='none'
    ).mean(dim=1).cpu().numpy()

X_encoded_1 = X_encoded_1.cpu().numpy()

print(f"\n✓ Encoded data shape: {X_encoded_1.shape}")
print(f"✓ Reconstruction error - Mean: {reconstruction_error.mean():.6f}, Std: {reconstruction_error.std():.6f}")

# ===== Visualize Reconstruction Error =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of reconstruction errors
axes[0].hist(reconstruction_error, bins=25, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel("Reconstruction Error")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of Reconstruction Errors (Sparse AE)")
axes[0].grid(True, alpha=0.3)

# Scatter plot with anomaly threshold
threshold = np.percentile(reconstruction_error, 90)
anomalies = reconstruction_error > threshold
axes[1].scatter(range(len(reconstruction_error)), reconstruction_error, 
               c=anomalies, cmap='RdYlGn_r', alpha=0.6, s=50)
axes[1].axhline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'90th Percentile: {threshold:.4f}')
axes[1].set_xlabel("Sample Index")
axes[1].set_ylabel("Reconstruction Error")
axes[1].set_title(f"Anomaly Detection ({anomalies.sum()} anomalies detected)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===== Visualize Encoded Representation =====
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_encoded_1[:, 0], X_encoded_1[:, 1], c=reconstruction_error, 
                    cmap='viridis', alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
plt.xlabel("Encoded Dimension 1")
plt.ylabel("Encoded Dimension 2")
plt.title("Sparse Autoencoder Encoded Representation (First 2 Dimensions)")
cbar = plt.colorbar(scatter, label="Reconstruction Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===== Summary Statistics =====
print("\n" + "="*60)
print("SPARSE AUTOENCODER SUMMARY")
print("="*60)
print(f"Total nodes: {len(X_encoded_1)}")
print(f"Anomalous nodes detected: {anomalies.sum()}")
print(f"Normal nodes: {(~anomalies).sum()}")
print(f"Anomaly percentage: {100 * anomalies.sum() / len(anomalies):.2f}%")