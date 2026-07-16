import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns


############### Data Prep ##################################
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
X_all = scaler.fit_transform(node_features)
X_train_ae, X_temp = train_test_split(X_all, test_size=0.4, random_state=42)
X_val_ae, X_test_ae = train_test_split(X_temp, test_size=0.5, random_state=42)

print(f"✓ Train AE shape: {X_train_ae.shape}")
print(f"✓ Val AE shape:   {X_val_ae.shape}")
print(f"✓ Test AE shape:  {X_test_ae.shape}")

############# The Model ##############################
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def train_autoencoder(X_train, X_val, X_test, epochs=100, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    model = DeepAutoencoder(input_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
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
            torch.save(model.state_dict(), 'best_autoencoder.pth')
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | Test: {test_loss:.6f}")

    model.load_state_dict(torch.load('best_autoencoder.pth'))


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', color='red', linewidth=2)
    plt.plot(range(1, len(test_losses)+1), test_losses, label='Test Loss', color='green', linewidth=2, linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss (MSE)")
    plt.title("Autoencoder Training (3-Way Split)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


    model.eval()
    with torch.no_grad():
        _, X_train_enc = model(train_tensor.to(device))
        _, X_val_enc = model(val_tensor.to(device))
        _, X_test_enc = model(test_tensor.to(device))
    
    return (X_train_enc.cpu().numpy(), 
            X_val_enc.cpu().numpy(), 
            X_test_enc.cpu().numpy())

print("\n" + "="*60)
print("Training Autoencoder (PROPER 3-WAY SPLIT)")
print("="*60 + "\n")

X_train_enc, X_val_enc, X_test_enc = train_autoencoder(X_train_ae, X_val_ae, X_test_ae, epochs=100, batch_size=16)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_tensor = torch.tensor(X_val_ae, dtype=torch.float32)

model.eval()
with torch.no_grad():
    val_out, _ = model(val_tensor.to(device))
    val_recon_error = torch.nn.functional.mse_loss(
        val_out, val_tensor.to(device), reduction='none'
    ).mean(dim=1).cpu().numpy()


threshold = np.percentile(val_recon_error, 90)
print(f"\n✓ Anomaly Threshold (from validation set): {threshold:.6f}")


test_tensor = torch.tensor(X_test_ae, dtype=torch.float32)
with torch.no_grad():
    test_out, _ = model(test_tensor.to(device))
    test_recon_error = torch.nn.functional.mse_loss(
        test_out, test_tensor.to(device), reduction='none'
    ).mean(dim=1).cpu().numpy()

y_test_labels = (test_recon_error > threshold).astype(int)

print(f"✓ Test Set Labels:")
print(f"  Normal samples: {(y_test_labels == 0).sum()}")
print(f"  Anomalous samples: {(y_test_labels == 1).sum()}")
print("\n" + "="*60)
print("Training DBN Classifier (PROPER SPLIT)")
print("="*60 + "\n")

train_tensor = torch.tensor(X_train_ae, dtype=torch.float32)
with torch.no_grad():
    train_out, _ = model(train_tensor.to(device))
    train_recon_error = torch.nn.functional.mse_loss(
        train_out, train_tensor.to(device), reduction='none'
    ).mean(dim=1).cpu().numpy()

y_train_labels = (train_recon_error > threshold).astype(int)

print(f"Train Set Labels:")
print(f"  Normal: {(y_train_labels == 0).sum()} | Anomalous: {(y_train_labels == 1).sum()}")


scaler_enc = StandardScaler()
X_train_enc_scaled = scaler_enc.fit_transform(X_train_enc)
X_val_enc_scaled = scaler_enc.transform(X_val_enc)
X_test_enc_scaled = scaler_enc.transform(X_test_enc)


X_train_enc_binary = (X_train_enc_scaled > 0).astype(int)
X_val_enc_binary = (X_val_enc_scaled > 0).astype(int)
X_test_enc_binary = (X_test_enc_scaled > 0).astype(int)


rbm = BernoulliRBM(
    n_components=16,  
    learning_rate=0.01,
    batch_size=4,
    n_iter=20,
    verbose=0,
    random_state=42
)

rbm.fit(X_train_enc_binary)
X_train_rbm = rbm.transform(X_train_enc_binary)
X_val_rbm = rbm.transform(X_val_enc_binary)
X_test_rbm = rbm.transform(X_test_enc_binary)


logistic = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
logistic.fit(X_train_rbm, y_train_labels)
y_val_pred = logistic.predict(X_val_rbm)
y_val_proba = logistic.predict_proba(X_val_rbm)[:, 1]
y_test_pred = logistic.predict(X_test_rbm)
y_test_proba = logistic.predict_proba(X_test_rbm)[:, 1]

# ===== Evaluation =====
print("\n" + "="*60)
print("VALIDATION SET EVALUATION")
print("="*60)

y_val_labels = (val_recon_error > threshold).astype(int)
print("\nClassification Report (Validation):")
print(classification_report(y_val_labels, y_val_pred, target_names=['Normal', 'Anomaly']))
print(f"ROC-AUC (Validation): {roc_auc_score(y_val_labels, y_val_proba):.4f}")

print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

print("\nClassification Report (Test):")
print(classification_report(y_test_labels, y_test_pred, target_names=['Normal', 'Anomaly']))
print(f"ROC-AUC (Test): {roc_auc_score(y_test_labels, y_test_proba):.4f}")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_val = confusion_matrix(y_val_labels, y_val_pred)
cm_test = confusion_matrix(y_test_labels, y_test_pred)

sns.heatmap(cm_val, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
axes[0].set_title("Validation Confusion Matrix")
axes[0].set_ylabel("True Label")
axes[0].set_xlabel("Predicted Label")

sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
axes[1].set_title("Test Confusion Matrix")
axes[1].set_ylabel("True Label")
axes[1].set_xlabel("Predicted Label")

plt.tight_layout()
plt.show()

# ROC curves
fpr_val, tpr_val, _ = roc_curve(y_val_labels, y_val_proba)
fpr_test, tpr_test, _ = roc_curve(y_test_labels, y_test_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr_val, tpr_val, label=f'Validation (AUC={roc_auc_score(y_val_labels, y_val_proba):.3f})', linewidth=2)
plt.plot(fpr_test, tpr_test, label=f'Test (AUC={roc_auc_score(y_test_labels, y_test_proba):.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("✓ Analysis Complete!")
print("="*60)