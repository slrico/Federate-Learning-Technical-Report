import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split


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

edges = []
for _, row in df.iterrows():
    src_idx = ip_to_idx[row['Source']]
    dst_idx = ip_to_idx[row['Destination']]
    edges.append([src_idx, dst_idx])

all_edges = edges + [[dst, src] for src, dst in edges]
edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()


x = torch.FloatTensor(node_features)
node_indices = np.arange(num_nodes)
train_idx, test_idx = train_test_split(node_indices, test_size=0.3, random_state=42)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

data = Data(x=x, edge_index=edge_index, train_mask=train_mask, test_mask=test_mask)


class GATEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

class GCNDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        return z

class GAT_Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.encoder = GATEncoder(input_dim, hidden_dim, num_heads)
        self.decoder = GCNDecoder(hidden_dim, input_dim)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        reconstructed_x = self.decoder(z)
        return reconstructed_x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT_Autoencoder(input_dim=x.shape[1], hidden_dim=8, num_heads=4).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
model.train()
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    reconstructed_x = model(data.x, data.edge_index)
    loss = criterion(reconstructed_x[data.train_mask], data.x[data.train_mask])
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    reconstructed_x = model(data.x, data.edge_index)
    print("Reconstructed node features (Test Data):\n", reconstructed_x[data.test_mask])