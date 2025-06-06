import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Sử dụng GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mô hình Fusion đơn giản với 2 đầu vào
class FusionNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=128):
        super(FusionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Dataset huấn luyện từ weighted features



class WeightedFeatureDataset(Dataset):
    def __init__(self, num_samples=3000):
        self.data = []

        for _ in range(num_samples):
            
            cam_feat = np.random.uniform(1000, 10000)
            drn_feat = np.random.uniform(1000, 250000)
            
            delta_cam = np.random.uniform(1, 20000)
            delta_drn = np.random.uniform(1, 20000)
            
            epsilon = 1e-6
            w_cam = delta_cam + epsilon
            w_drn = delta_drn + epsilon
            total = w_cam + w_drn
            w_cam /= total
            w_drn /= total

            
            f1 = w_cam * cam_feat
            f2 = w_drn * drn_feat
            fused_features = [f1, f2]

           
            risk_score = f1 + f2
            label = 1.0 if risk_score > 7000 else 0.0

            self.data.append((
                torch.tensor(fused_features, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32)
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Wrapper model huấn luyện

class WeightedFusion:
    def __init__(self):
        self.fusion_network = FusionNetwork(input_size=2).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.fusion_network.parameters(), lr=0.001)

    def train(self, dataloader, epochs=20):
        self.fusion_network.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for fused_features, label in dataloader:
                fused_features = fused_features.to(device)
                label = label.to(device).unsqueeze(1)

                output = self.fusion_network(fused_features)
                loss = self.criterion(output, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    def predict(self, fused_feature_tensor):
        self.fusion_network.eval()
        with torch.no_grad():
            fused_feature_tensor = fused_feature_tensor.to(device)
            output = self.fusion_network(fused_feature_tensor)
            return output.item()

# ========== Chạy huấn luyện ==========
if __name__ == "__main__":
    dataset = WeightedFeatureDataset(num_samples=3000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    fusion_model = WeightedFusion()
    fusion_model.train(dataloader, epochs=20)
    torch.save(fusion_model.fusion_network.state_dict(), "fusion_model.pt")
    print("✅ Mô hình đã được lưu vào file: fusion_model.pt")
    # Test 1
