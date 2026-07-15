import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler  # ✓ FIXED: MinMaxScaler instead of StandardScaler
from sklearn.model_selection import train_test_split
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

X_train, X_temp = train_test_split(X_scaled, test_size=0.4, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

print(f"✓ Train shape: {X_train.shape}")
print(f"✓ Val shape:   {X_val.shape}")
print(f"✓ Test shape:  {X_test.shape}")


class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DeepAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
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
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


def train_autoencoder_with_gmm(X_train, X_val, X_test, epochs=100, batch_size=16, l2_reg=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    model = DeepAutoencoder(input_dim).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=l2_reg)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    train_losses, val_losses, test_losses = [], [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        model.eval()
        with torch.no_grad():
            val_out, _ = model(val_tensor.to(device))
            val_loss = criterion(val_out, val_tensor.to(device)).item()
            
            test_out, _ = model(test_tensor.to(device))
            test_loss = criterion(test_out, test_tensor.to(device)).item()
        
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_autoencoder_gmm.pth')
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | Test: {test_loss:.6f}")

    model.load_state_dict(torch.load('best_autoencoder_gmm.pth'))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', color='blue', label='Train Loss', linewidth=2, markersize=4)
    plt.plot(range(1, len(val_losses)+1), val_losses, marker='s', color='red', label='Validation Loss', linewidth=2, markersize=4)
    plt.plot(range(1, len(test_losses)+1), test_losses, marker='^', color='green', label='Test Loss', linewidth=2, markersize=4, linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss (MSE)")
    plt.title("Deep Autoencoder Training Loss (FIXED: MinMaxScaler + Sigmoid)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.yscale('log')  # Log scale to see progress
    plt.tight_layout()
    plt.show()

    model.eval()
    with torch.no_grad():
        _, X_train_enc = model(train_tensor.to(device))
        _, X_val_enc = model(val_tensor.to(device))
        _, X_test_enc = model(test_tensor.to(device))

    print(f"\n✓ Encoded data shapes:")
    print(f"  Train: {X_train_enc.cpu().numpy().shape}")
    print(f"  Val:   {X_val_enc.cpu().numpy().shape}")
    print(f"  Test:  {X_test_enc.cpu().numpy().shape}")
    
    return model, (X_train_enc.cpu().numpy(), X_val_enc.cpu().numpy(), X_test_enc.cpu().numpy())


print("\n" + "="*60)
print("Training Deep Autoencoder on Real IoT Data (FIXED)")
print("="*60 + "\n")

model, (X_train_enc, X_val_enc, X_test_enc) = train_autoencoder_with_gmm(
    X_train, X_val, X_test,
    epochs=100, 
    batch_size=16, 
    l2_reg=0.001
)

print("\nFitting Gaussian Mixture Model on validation set...")
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_val_enc)
val_assignments = gmm.predict(X_val_enc)
test_assignments = gmm.predict(X_test_enc)

print(f"✓ Val Component Assignments: {np.unique(val_assignments, return_counts=True)}")
print(f"✓ Test Component Assignments: {np.unique(test_assignments, return_counts=True)}")
print(f"✓ GMM Log-Likelihood (Val): {gmm.score(X_val_enc):.4f}")

# ===== Visualize Results =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Component Distribution (Validation)
axes[0, 0].hist(val_assignments, bins=3, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel("Component ID")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_title("GMM Component Distribution (Validation)")
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Encoded Data Scatter (first 2 dimensions)
scatter = axes[0, 1].scatter(X_val_enc[:, 0], X_val_enc[:, 1], 
                             c=val_assignments, cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[0, 1].set_xlabel("Encoded Dim 1")
axes[0, 1].set_ylabel("Encoded Dim 2")
axes[0, 1].set_title("Encoded Data with GMM Components (Validation)")
plt.colorbar(scatter, ax=axes[0, 1], label="Component")
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Reconstruction Error by Component (Test)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_tensor = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    outputs, _ = model(test_tensor.to(device))
    reconstruction_error = torch.nn.functional.mse_loss(
        outputs, test_tensor.to(device), reduction='none'
    ).mean(dim=1).cpu().numpy()

axes[1, 0].scatter(range(len(test_assignments)), reconstruction_error, 
                  c=test_assignments, cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
axes[1, 0].set_xlabel("Sample Index")
axes[1, 0].set_ylabel("Reconstruction Error")
axes[1, 0].set_title("Reconstruction Error by Sample (Test)")
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: BIC & AIC Scores
n_components_range = range(2, 8)
bic_scores = [GaussianMixture(n_components=k, random_state=42).fit(X_val_enc).bic(X_val_enc) 
              for k in n_components_range]
aic_scores = [GaussianMixture(n_components=k, random_state=42).fit(X_val_enc).aic(X_val_enc) 
              for k in n_components_range]

axes[1, 1].plot(n_components_range, bic_scores, marker='o', label='BIC', linewidth=2, color='blue')
axes[1, 1].plot(n_components_range, aic_scores, marker='s', label='AIC', linewidth=2, color='red')
axes[1, 1].set_xlabel("Number of Components")
axes[1, 1].set_ylabel("Score")
axes[1, 1].set_title("Model Selection: BIC vs AIC")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===== Step 7: Summary Statistics =====
print("\n" + "="*60)
print("SUMMARY STATISTICS (TEST SET)")
print("="*60)
print(f"Total test samples: {len(X_test_enc)}")
for i in range(3):
    count = (test_assignments == i).sum()
    pct = 100 * count / len(test_assignments)
    print(f"  Component {i}: {count} samples ({pct:.1f}%)")

print(f"\nReconstruction Error Statistics (Test):")
print(f"  Mean: {reconstruction_error.mean():.6f}")
print(f"  Std:  {reconstruction_error.std():.6f}")
print(f"  Min:  {reconstruction_error.min():.6f}")
print(f"  Max:  {reconstruction_error.max():.6f}")
print(f"  Median: {np.median(reconstruction_error):.6f}")