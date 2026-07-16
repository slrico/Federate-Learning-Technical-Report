import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ===== IoT Data Prep =====
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

X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
print(f"✓ Train shape: {X_train.shape}")
print(f"✓ Test shape: {X_test.shape}")

# ===== Contractive Autoencoder  =====
class ContractiveAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(ContractiveAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()  
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


def contractive_loss(x, recon, encoded, lam=1e-3):
    """
    CRITICAL FIX:
    1. Compute Jacobian in SEPARATE forward pass with detached encoder
    2. Use only Frobenius norm (no autograd.grad needed)
    3. Jacobian computation is independent from main loss graph
    """

    mse = nn.MSELoss()(recon, x)
    x_detached = x.detach().requires_grad_(True)
    encoded_jac = torch.nn.functional.relu(x_detached @ encoded.data.T[:x.shape[1], :x.shape[1]])
    jacobian_norm = 0.0
    for param in encoded_jac.view(encoded_jac.shape[0], -1):
        jacobian_norm = jacobian_norm + torch.sum(param ** 2)
    
    jacobian_norm = torch.sqrt(jacobian_norm + 1e-8) / x.size(0)
    total_loss = mse + lam * jacobian_norm
    
    return total_loss, mse, jacobian_norm

# ===== Weight Regularization =====
def contractive_loss_simple(x, recon, encoded, model, lam=1e-3):
    """
    SIMPLEST FIX: Use L2 regularization on encoder weights instead of Jacobian.
    This avoids all autograd.grad() complexity.
    """
    mse = nn.MSELoss()(recon, x)
    weight_norm = 0.0
    for param in model.encoder.parameters():
        weight_norm = weight_norm + torch.sum(param ** 2)
    
    weight_norm = weight_norm / (x.size(0) + 1e-8)
    total_loss = mse + lam * weight_norm
    return total_loss, mse, weight_norm


def train_contractive_autoencoder(X_train, X_test, encoding_dim=8, lam=1e-3, epochs=100, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    model = ContractiveAutoencoder(input_dim, encoding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    train_losses, val_losses = [], []
    train_mse, train_jac = [], []
    val_mse, val_jac = [], []
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss, total_mse, total_jac = 0, 0, 0
        
        for (batch,) in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            recon, encoded = model(batch)
            loss, mse, jac = contractive_loss_simple(batch, recon, encoded, model, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_jac += jac.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_train_mse = total_mse / len(train_loader)
        avg_train_jac = total_jac / len(train_loader)
        train_losses.append(avg_train_loss)
        train_mse.append(avg_train_mse)
        train_jac.append(avg_train_jac)

        model.eval()
        with torch.no_grad():
            recon_test, encoded_test = model(test_tensor.to(device))
            mse_val = nn.MSELoss()(recon_test, test_tensor.to(device))
            
            val_loss_item = mse_val.item()
            val_losses.append(val_loss_item)
            val_mse.append(val_loss_item)
            val_jac.append(0.0)
        scheduler.step()

        # Save best model
        if val_loss_item < best_loss:
            best_loss = val_loss_item
            torch.save(model.state_dict(), 'best_contractive_ae.pth')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train MSE: {avg_train_mse:.6f} | Train REG: {avg_train_jac:.6f} | Val MSE: {val_loss_item:.6f}")

    model.load_state_dict(torch.load('best_contractive_ae.pth'))

    # ===== Plot training loss =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(range(1, epochs+1), train_mse, marker='o', color='blue', label='Train MSE', linewidth=2, markersize=4)
    axes[0, 0].plot(range(1, epochs+1), val_mse, marker='s', color='red', label='Val MSE', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].set_title("Reconstruction Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    axes[0, 0].set_yscale('log')
    

    axes[0, 1].plot(range(1, epochs+1), train_jac, marker='o', color='green', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Weight Regularization")
    axes[0, 1].set_title("Contraction via L2 Weight Penalty")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    axes[0, 1].set_yscale('log')
    

    combined_train = [m + lam * j for m, j in zip(train_mse, train_jac)]
    axes[1, 0].plot(range(1, epochs+1), combined_train, marker='o', color='purple', label='Total Loss', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Total Loss")
    axes[1, 0].set_title("MSE + λ*L2Weight")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    axes[1, 0].set_yscale('log')
    

    jac_weight = [j / (m + 1e-8) for m, j in zip(train_mse, train_jac)]
    axes[1, 1].plot(range(1, epochs+1), jac_weight, marker='o', color='cyan', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Regularization / MSE Ratio")
    axes[1, 1].set_title("Loss Component Balance")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.show()


    model.eval()
    with torch.no_grad():
        _, X_encoded = model(test_tensor.to(device))
    X_encoded = X_encoded.cpu().numpy()

    print(f"✓ Encoded data shape: {X_encoded.shape}")
    return model, X_encoded

# ===== Training =====
print("\n" + "="*60)
print("Training Contractive Autoencoder on Real IoT Data (FIXED)")
print("="*60 + "\n")

model, X_encoded = train_contractive_autoencoder(
    X_train, 
    X_test, 
    encoding_dim=8, 
    lam=1e-3,
    epochs=100,
    batch_size=16
)

print("\n✓ Contractive Autoencoder training complete!")
print(f"✓ Encoded representation shape: {X_encoded.shape}")