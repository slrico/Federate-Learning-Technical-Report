import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ===== Data Prep =====
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

# ===== Adversarial Autoencoder =====
class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, output_dim):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid() 
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.net(z)


def train_adversarial_autoencoder(X_train, X_test, epochs=50, batch_size=16, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    encoder = Encoder(input_dim).to(device)
    decoder = Decoder(input_dim).to(device)
    discriminator = Discriminator().to(device)

    ae_optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

    criterion_recon = nn.MSELoss()
    criterion_adv = nn.BCELoss()

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    ae_losses, d_losses, adv_losses, val_losses = [], [], [], []
    best_ae_loss = float('inf')

    for epoch in range(epochs):
        total_ae_loss, total_d_loss, total_adv_loss = 0, 0, 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            z_real = torch.randn(batch.size(0), 8).to(device)  # prior samples
            z_fake = encoder(batch).detach()

            d_real = discriminator(z_real)
            d_fake = discriminator(z_fake)

            d_loss_real = criterion_adv(d_real, torch.ones_like(d_real))
            d_loss_fake = criterion_adv(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_loss_real + d_loss_fake) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            z = encoder(batch)
            recon = decoder(z)
            ae_loss = criterion_recon(recon, batch)

            d_pred = discriminator(z)
            adv_loss = criterion_adv(d_pred, torch.ones_like(d_pred))

            total_loss = ae_loss + 0.001 * adv_loss

            ae_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            ae_optimizer.step()

            total_ae_loss += ae_loss.item()
            total_d_loss += d_loss.item()
            total_adv_loss += adv_loss.item()

        avg_ae_loss = total_ae_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)
        avg_adv_loss = total_adv_loss / len(train_loader)
        
        ae_losses.append(avg_ae_loss)
        d_losses.append(avg_d_loss)
        adv_losses.append(avg_adv_loss)

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            z_test = encoder(test_tensor.to(device))
            recon_test = decoder(z_test)
            val_loss = criterion_recon(recon_test, test_tensor.to(device)).item()
            val_losses.append(val_loss)

        encoder.train()
        decoder.train()
        if avg_ae_loss < best_ae_loss:
            best_ae_loss = avg_ae_loss
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'discriminator': discriminator.state_dict()
            }, 'best_aae_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | AE Loss: {avg_ae_loss:.6f} | D Loss: {avg_d_loss:.6f} | Adv Loss: {avg_adv_loss:.6f} | Val Loss: {val_loss:.6f}")

    # ===== Plot Losses =====
    plt.figure(figsize=(12, 6))
    plt.plot(ae_losses, label="Autoencoder Loss", color="blue", linewidth=2)
    plt.plot(d_losses, label="Discriminator Loss", color="red", linewidth=2)
    plt.plot(adv_losses, label="Adversarial Loss", color="green", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", color="orange", linewidth=2, linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Adversarial Autoencoder Training on Real IoT Data")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Load best model
    checkpoint = torch.load('best_aae_model.pth')
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.eval()
    with torch.no_grad():
        X_encoded = encoder(test_tensor.to(device)).cpu().numpy()

    print(f"\n✓ Encoded data shape: {X_encoded.shape}")
    return encoder, decoder, discriminator, X_encoded


print("\n" + "="*60)
print("Training Adversarial Autoencoder on Real IoT Data")
print("="*60 + "\n")

encoder, decoder, discriminator, X_encoded = train_adversarial_autoencoder(
    X_train, X_test, 
    epochs=150, 
    batch_size=16, 
    lr=1e-3
)

print("\n✓ Adversarial Autoencoder training complete!")
print(f"✓ Encoded representation shape: {X_encoded.shape}")