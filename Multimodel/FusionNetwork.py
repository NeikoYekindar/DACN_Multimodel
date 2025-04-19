import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np


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
        """
        Khởi tạo mạng Fusion Network để kết hợp đặc trưng từ camera và drone
        """
        self.fusion_network = FusionNetwork(input_size=2, hidden_size=128).to(device)
        self.camera_only_net = CameraOnlyNetwork().to(device)
        self.drone_only_net = DroneOnlyNetwork().to(device)
        #khởi tạo bộ tối ưu hóa
        self.optimizer = optim.Adam(self.fusion_network.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def calculate_weights(self, delta_camera, delta_drone):
       
        w_camera_prime = 1 / (1+delta_camera)
        w_drone_prime = 1 / (1+delta_drone)
        
        w_camera = w_camera_prime / (w_camera_prime + w_drone_prime)
        w_drone = w_drone_prime / (w_camera_prime + w_drone_prime)
        
        return w_camera, w_drone
    
    def extract_features(self, camera_image, drone_image, camera_model, drone_model):

        
        result_camera = camera_model(camera_image)
        result_drone = drone_model(drone_image)

        
        def extract_water_area_feature(results):
            masks = results[0].masks
            if masks is None:
                return torch.tensor([0.0], dtype=torch.float32).to(device)
            mask_data = masks.data.cpu().numpy()
            water_area_per_obj = np.sum(mask_data, axis=(1, 2))
            total_water_area = np.sum(water_area_per_obj)
            return torch.tensor([total_water_area], dtype=torch.float32).to(device)
        
        camera_feature = extract_water_area_feature(result_camera)
        drone_feature = extract_water_area_feature(result_drone)

        return camera_feature, drone_feature
    def extract_feature_camera(self, camera_image, camera_model):
        result_camera = camera_model(camera_image)
        def extract_water_area_feature(results):
            masks = results[0].masks
            if masks is None:
                return torch.tensor([0.0], dtype=torch.float32).to(device)
            mask_data = masks.data.cpu().numpy()
            water_area_per_obj = np.sum(mask_data, axis=(1, 2))
            total_water_area = np.sum(water_area_per_obj)
            return torch.tensor([total_water_area], dtype=torch.float32).to(device)
        camera_feature = extract_water_area_feature(result_camera)
        return camera_feature
    
    def extract_feature_drone(self, drone_image, drone_model):
        result_drone = drone_model(drone_image)
        def extract_water_area_feature(result):
            masks= result[0].masks
            if masks is None:
                return torch.tensor([0.0], dtype=torch.float32).to(device)
            mask_data = masks.data.cpu().numpy()
            water_area_per_obj = np.sum(mask_data, axis=(1,2))
            total_water_area = np.sum(water_area_per_obj)
            return torch.tensor([total_water_area], dtype=torch.float32).to(device)
        drone_feature = extract_water_area_feature(result_drone)
        return drone_feature
            
    def fuse_features(self, camera_feature, drone_feature, w_camera, w_drone):
        # Tính tổng có trọng số (weighted sum)
        # weighted_sum = w_camera * camera_feature + w_drone * drone_feature
        fused_features = torch.cat([
            (w_camera * camera_feature).unsqueeze(0),
            (w_drone * drone_feature).unsqueeze(0)
        ], dim=1)
        #fused_features = torch.cat([camera_feature.unsqueeze(0), drone_feature.unsqueeze(0)], dim=1)
        return fused_features
    
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
    
class CameraOnlyNetwork(nn.Module):
    def __init__(self):
        super(CameraOnlyNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
class DroneOnlyNetwork(nn.Module):
    def __init__(self):
        super(DroneOnlyNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x