import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FusionNetwork(nn.Module):
    def __init__(self, input_size=256, hidden_size=128):
        super(FusionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Lớp Fully Connected đầu tiên
        self.relu = nn.ReLU()  # Hàm kích hoạt ReLU
        self.fc2 = nn.Linear(hidden_size, 1)  # Lớp Fully Connected thứ hai
        self.sigmoid = nn.Sigmoid()  # Chuẩn hóa đầu ra về khoảng [0, 1]

    def forward(self, x):
        x = self.fc1(x)  # Biến đổi đặc trưng đầu vào
        x = self.relu(x)  # Áp dụng hàm kích hoạt ReLU
        x = self.fc2(x)  # Biến đổi đặc trưng lần hai
        print("Before sigmoid:", x)
        x = self.sigmoid(x)  # Chuẩn hóa đầu ra
        return x
    
class WeightedFusion:
    def __init__(self):
        self.fusion_network = FusionNetwork(input_size=2, hidden_size=128).to(device)
        self.camera_only_net = CameraOnlyNetwork().to(device)
        self.drone_only_net = DroneOnlyNetwork().to(device)
        self.optimizer = optim.Adam(self.fusion_network.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def calculate_weights(self, delta_camera, delta_drone):
        """Trọng số tỷ lệ thuận với delta — dùng log để ổn định"""
        epsilon = 1e-6  # tránh log(0)
        delta_camera = max(delta_camera, 0.0)
        delta_drone = max(delta_drone, 0.0)

        # Nếu cả hai đều không thay đổi, chia đều
        if delta_camera == 0 and delta_drone == 0:
            return 0.5, 0.5

        w_camera = math.log1p(delta_camera + epsilon)
        w_drone = math.log1p(delta_drone + epsilon)
        total = w_camera + w_drone
        return w_camera / total, w_drone / total

    def extract_water_area_feature(self, results):
        masks = results[0].masks
        if masks is None:
            return torch.tensor([0.0], dtype=torch.float32).to(device)
        mask_data = masks.data.cpu().numpy()
        water_area = np.sum(mask_data)
        return torch.tensor([water_area], dtype=torch.float32).to(device)

    def extract_features(self, camera_image, drone_image, camera_model, drone_model):
        cam_res = camera_model(camera_image)
        drn_res = drone_model(drone_image)

        cam_feat = self.extract_water_area_feature(cam_res)
        drn_feat = self.extract_water_area_feature(drn_res)
        return cam_feat, drn_feat

    def extract_feature_camera(self, camera_image, camera_model):
        result = camera_model(camera_image)
        return self.extract_water_area_feature(result)

    def extract_feature_drone(self, drone_image, drone_model):
        result = drone_model(drone_image)
        return self.extract_water_area_feature(result)

    def fuse_features(self, camera_feature, drone_feature, w_camera, w_drone):
        return torch.cat([
            (w_camera * camera_feature).unsqueeze(0),
            (w_drone * drone_feature).unsqueeze(0)
        ], dim=1)

    def predict_flood_risk(self, fused_features):
        self.fusion_network.eval()
        with torch.no_grad():
            risk_score = self.fusion_network(fused_features)
        return risk_score.item()

    def predict_camera_only_risk(self, camera_feature):
        self.camera_only_net.eval()
        with torch.no_grad():
            risk_score = self.camera_only_net(camera_feature)
        return risk_score.item()

    def predict_drone_only_risk(self, drone_feature):
        self.drone_only_net.eval()
        with torch.no_grad():
            risk_score = self.drone_only_net(drone_feature)
        return risk_score.item()