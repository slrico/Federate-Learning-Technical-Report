import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


X = features  
X_train_1, X_test_1 = train_test_split(X, test_size=0.3, random_state=42)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 4),
            nn.ReLU(),
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

def train_autoencoder(X_train, X_test, epochs=5, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]
    autoencoder = Autoencoder(input_dim).to(device)
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    autoencoder.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch, in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs, _ = autoencoder(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    autoencoder.eval()
    with torch.no_grad():
        _, X_encoded = autoencoder(test_tensor.to(device))
    X_encoded = X_encoded.cpu().numpy()

    print("Encoded data shape:", X_encoded.shape)
    return autoencoder, autoencoder.encoder, X_encoded


autoencoder, encoder, X_encoded_1 = train_autoencoder(
    X_train_1,  
    X_test_1,    
    epochs=5,
    batch_size=32
)

print("Encoded data shape:", X_encoded_1.shape)