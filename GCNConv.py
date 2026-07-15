import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import numpy as np

class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GCNDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(GCNDecoder, self).__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        return z

class GCN_Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=8):
        super(GCN_Autoencoder, self).__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim)
        self.decoder = GCNDecoder(hidden_dim, input_dim)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, z

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN_Autoencoder(input_dim=4, hidden_dim=8).to(device)
data = data.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
num_epochs = 100
losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    out, z = model(data.x, data.edge_index)
    loss = criterion(out, data.x) 
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# ===== Visualization: Training Loss =====
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), losses, marker='o', color='blue', linewidth=2, markersize=4)
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss (MSE)")
plt.title("GCN Autoencoder Training Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ===== Evaluate on Test Data =====
model.eval()
with torch.no_grad():
    reconstructed, encoded = model(data.x, data.edge_index)
    reconstruction_error = torch.nn.functional.mse_loss(
        reconstructed, data.x, reduction='none'
    ).mean(dim=1).cpu().numpy()

# ===== Visualization: Reconstruction Error Distribution =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of reconstruction errors
axes[0].hist(reconstruction_error, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel("Reconstruction Error")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of Node Reconstruction Errors")
axes[0].grid(True, alpha=0.3)

# Error threshold for anomaly detection
threshold = np.percentile(reconstruction_error, 90)
anomalies = reconstruction_error > threshold
axes[1].scatter(range(len(reconstruction_error)), reconstruction_error, 
               c=anomalies, cmap='RdYlGn_r', alpha=0.6, s=50)
axes[1].axhline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'90th Percentile: {threshold:.4f}')
axes[1].set_xlabel("Node Index")
axes[1].set_ylabel("Reconstruction Error")
axes[1].set_title(f"Anomaly Detection ({anomalies.sum()} anomalies detected)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===== Visualization: Encoded Representation =====
encoded_np = encoded.cpu().numpy()
plt.figure(figsize=(10, 6))
scatter = plt.scatter(encoded_np[:, 0], encoded_np[:, 1], c=reconstruction_error, 
                    cmap='viridis', alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
plt.xlabel(f"Encoded Dimension 1")
plt.ylabel(f"Encoded Dimension 2")
plt.title("GCN Encoded Representation (First 2 Dimensions)")
cbar = plt.colorbar(scatter, label="Reconstruction Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n✓ Total nodes: {data.num_nodes}")
print(f"✓ Anomalous nodes detected: {anomalies.sum()}")
print(f"✓ Reconstruction error - Mean: {reconstruction_error.mean():.6f}, Std: {reconstruction_error.std():.6f}")