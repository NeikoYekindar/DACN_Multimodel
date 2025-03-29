import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torchvision.transforms as transforms
import time
import matplotlib.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")


class ImageProcessor:
    def __init__(self):
        self.target_size = (640, 640)
    def preprocess_image(self, image):
        resized_img = cv2.resize(image, self.target_size)

        lab = cv2.cvtColor(resized_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_img
    def load_image(self, path):
        """Đọc hình ảnh từ đường dẫn"""

        if os.path.exists(path):
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Không thể đọc hình ảnh từ {path}")
            return image
        else:
            raise FileNotFoundError(f"Không tìm thấy file {path}")
    def save_processed_image(self, image, path):
        """Lưu hình ảnh đã xử lý"""
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(path, image)



class WaterSegmentationModel:
    def __init__(self, camera_model_path, drone_model_path):
        """Khởi tạo mô hình YOLOv11n-seg cho camera và drone"""
        #camera_model_path: 
        #drone_model_path: 
        
        if not os.path.exists(camera_model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình camera tại {camera_model_path}")
        if not os.path.exists(drone_model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình drone tại {drone_model_path}")
        
        print("Đang tải mô hình YOLO11n-seg cho camera...")
        self.camera_model = YOLO(camera_model_path)
        print("Đang tải mô hình YOLO11n-seg cho drone...")
        self.drone_model = YOLO(drone_model_path)
        print("Đã tải xong cả hai mô hình!")


        # Tham chiếu diện tích an toàn (có thể điều chỉnh)
        self.reference_camera_area = 214139    # Diện tích mặc định, sẽ được cập nhật 214139 
        self.reference_drone_area = 114114 # Diện tích mặc định, sẽ được cập nhật 114114




    def set_reference_data(self, camera_ref_area, drone_ref_area):
        """Thiết lập dữ liệu tham chiếu cho diện tích vùng nước an toàn"""
        self.reference_camera_area = camera_ref_area
        self.reference_drone_area = drone_ref_area
        print(f"Đã thiết lập diện tích tham chiếu: Camera = {camera_ref_area}, Drone = {drone_ref_area}")



    def segment_water(self, image, is_drone = False):
        """
            Phân đoạn vùng nước từ hình ảnh sử dụng YOLOv11n-seg
        Args:
            image: Hình ảnh đầu vào
            is_drone: True nếu hình ảnh từ drone, False nếu từ camera
        
        Returns:
            water_mask: Mặt nạ vùng nước
            water_area: Diện tích vùng nước
            water_percentage: Phần trăm vùng nước trong hình ảnh
        """
        model = self.drone_model if is_drone else self.camera_model
        results = model(image, verbose=False)
        #Xử lý kết quả phân đoạn
        water_mask = None
        for r in results:
            if hasattr(r, 'masks') and r.masks is not None:
                masks = r.masks.data
                boxes = r.boxes.data


                # Tìm vùng nước trong các đối tượng được phát hiện
                for i, box in enumerate(boxes):
                    # Giả định rằng lớp 0 là nước 
                    if box[5].item() == 0:  # class_id = 0 cho nước
                        if water_mask is None:
                            water_mask = masks[i].cpu().numpy()
                        else:
                            water_mask = np.logical_or(water_mask, masks[i].cpu().numpy())
        # Nếu không tìm thấy vùng nước
        if water_mask is None:
            water_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            water_area = 0
            water_percentage = 0
        else:
            water_mask = water_mask.astype(np.uint8) * 255
            water_area = np.sum(water_mask > 0)
            water_percentage = (water_area / (image.shape[0] * image.shape[1])) * 100
        return water_mask, water_area, water_percentage


    def calculate_area_change(self, camera_area, drone_area):
        """
        Tính toán sự thay đổi diện tích so với dữ liệu tham chiếu
        
        Returns:
            delta_camera: Mức độ thay đổi diện tích từ camera
            delta_drone: Mức độ thay đổi diện tích từ drone
        """
        # tính sự thay đổi (có thể âm nếu diện tích hiện tại nhỏ hơn diện tích tham chiếu)

        delta_camera = camera_area - self.reference_camera_area
        delta_drone = drone_area - self.reference_drone_area

        # chuyển đổi sang giá trị tuyệt đối để tính trọng só
        # tránh trường hợp chia cho 0
        delta_camera_abs = abs(delta_camera)
        delta_drone_abs = abs(delta_drone)
        return delta_camera_abs, delta_drone_abs



# weight fusion 
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
    
class FloodWarningSystem:
    def __init__(self, risk_threshold = 0.65):
        self.risk_threshold = risk_threshold
    def generate_alert(self, risk_score):
        alert = risk_score >= self.risk_threshold
        if alert:
            message = f"FLOOD ALERT: Detected high risk of flooding with score {risk_score:.2f}"
        else:
            message = f"NO FLOOD DETECTED: Flood risk score is {risk_score:.2f}"
        
        return alert, message
    def visualize_alert(self, camera_image, drone_image, water_mask_camera, water_mask_drone, 
                         alert, message, risk_score):
        # Tạo figure với 2x2 ô
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Hiển thị hình ảnh camera và drone
        axs[0, 0].imshow(cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Camera Image")
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(cv2.cvtColor(drone_image, cv2.COLOR_BGR2RGB))
        axs[0, 1].set_title("Drone Image")
        axs[0, 1].axis('off')
        
        # Hiển thị mặt nạ vùng nước
        if water_mask_camera is not None:
            axs[1, 0].imshow(water_mask_camera, cmap='Blues')
        axs[1, 0].set_title("Camera Water Mask")
        axs[1, 0].axis('off')
        
        if water_mask_drone is not None:
            axs[1, 1].imshow(water_mask_drone, cmap='Blues')
        axs[1, 1].set_title("Drone Water Mask")
        axs[1, 1].axis('off')
        
        # Thêm thông báo cảnh báo
        plt.figtext(0.5, 0.01, message, ha="center", fontsize=14, 
                   bbox={"facecolor": "red" if alert else "green", "alpha": 0.5, "pad": 5})
        
        plt.figtext(0.5, 0.05, f"Risk Score: {risk_score:.4f}", ha="center", fontsize=12)
        
        plt.tight_layout()
        
        # Chuyển đổi figure thành hình ảnh
        fig.canvas.draw()
        visualization = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        visualization = visualization.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        visualization = visualization[:, :, 1:]
        
        plt.close(fig)
        
        return visualization


def visualize_detected_images(camera_image, drone_image, water_mask_camera, water_mask_drone):
    """
    Hiển thị và kết hợp ảnh gốc với mặt nạ vùng nước.
    Args:
        camera_image: Ảnh từ camera.
        drone_image: Ảnh từ drone.
        water_mask_camera: Mặt nạ vùng nước từ camera.
        water_mask_drone: Mặt nạ vùng nước từ drone.
    Returns:
        camera_detected: Ảnh camera kèm mặt nạ vùng nước.
        drone_detected: Ảnh drone kèm mặt nạ vùng nước.
    """
    colored_mask_camera = np.zeros_like(camera_image)
    colored_mask_camera[water_mask_camera > 0] = (0, 255, 255) 
    color_mask_drone = np.zeros_like(drone_image)
    color_mask_drone[water_mask_drone>0] =  (0, 255, 255)


    # Kết hợp ảnh gốc với mặt nạ vùng nước
    # camera_detected = cv2.addWeighted(camera_image, 0.7, cv2.cvtColor(water_mask_camera, cv2.COLOR_GRAY2BGR), 0.3, 0)
    # drone_detected = cv2.addWeighted(drone_image, 0.7, cv2.cvtColor(water_mask_drone, cv2.COLOR_GRAY2BGR), 0.3, 0)
    camera_detected = cv2.addWeighted(camera_image, 0.7, colored_mask_camera, 0.3, 0)
    drone_detected = cv2.addWeighted(drone_image, 0.7, color_mask_drone, 0.3, 0)

    # Hiển thị kết quả
    cv2.imshow("Detected Camera Image", camera_detected)
    cv2.imshow("Detected Drone Image", drone_detected)

    # Đợi người dùng nhấn phím để đóng cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return camera_detected, drone_detected




def main():
    camera_model_path = "E:/DACN/Curent_model/camera_model.pt"
    drone_model_path = "runs/segment/train2/weights/best.pt"

    camera_image_path = "E:/DACN/kenh_new.jpg"
    drone_image_path = "E:/DACN/9536 (1)2.jpg"



    use_drone = os.path.exists(drone_image_path)



    image_processor = ImageProcessor()



    
    water_segmentation_model = WaterSegmentationModel(camera_model_path, drone_model_path)
    weight_fusion = WeightedFusion()
    warning_system = FloodWarningSystem(risk_threshold=0.5)

    #xử lý hình ảnh
    camera_image = image_processor.load_image(camera_image_path)
    assert camera_image is not None, "Không thể đọc ảnh từ camera"
   
    camera_image = image_processor.preprocess_image(camera_image)
    
    # phân đoạn vùng nước
    #camera_mask, camera_area, camera_percentage = water_segmentation_model.segment_water(camera_image)
    camera_mask, camera_area, camera_percentage = water_segmentation_model.segment_water(camera_image)
    print(f"Diện tích vùng nước từ camera: {camera_area}, Phần trăm: {camera_percentage:.2f}%")



    if use_drone:

        drone_image = image_processor.load_image(drone_image_path)
        assert drone_image is not None, "Không thể đọc ảnh từ drone"
        print("Ảnh đã được đọc thành công!")
        drone_image = image_processor.preprocess_image(drone_image)
        

        drone_mask, drone_area, drone_percentage = water_segmentation_model.segment_water(drone_image, is_drone=True)
        print(f"Diện tích vùng nước từ drone: {drone_area}, Phần trăm: {drone_percentage:.2f}%")
        # tính sự thay đổi diện tích nước
        delta_camera, delta_drone = water_segmentation_model.calculate_area_change(camera_area, drone_area)
         # tính trọng só
        w_camera, w_drone = weight_fusion.calculate_weights(delta_camera, delta_drone)
        print(f"Trọng số vùng nước từ camera: {w_camera}")
        print(f"Trọng số vùng nước từ drone: {w_drone}")
        camera_feature, drone_feature = weight_fusion.extract_features(
            camera_image, drone_image, 
            water_segmentation_model.camera_model, 
            water_segmentation_model.drone_model
        )
        print(f"Feature vùng nước từ camera: {camera_feature}")
        print(f"Feature vùng nước từ drone: {drone_feature}")
        # Kết hợp đặc trưng
        fused_features = weight_fusion.fuse_features(camera_feature, drone_feature, w_camera, w_drone)
        risk_score = weight_fusion.predict_flood_risk(fused_features)

    else:
        print("Không tìm thấy ảnh drone. Sử dụng dữ liệu từ camera để phân tích.")
        camera_feature = weight_fusion.extract_feature_camera(camera_image, water_segmentation_model.camera_model)
        print(f"Feature vùng nước từ camera: {camera_feature}")
        fused_features = camera_feature
        risk_score = weight_fusion.predict_camera_only_risk(fused_features)


    # Dự đoán điểm rủi ro ngập lụt
    #risk_score = weight_fusion.predict_flood_risk(fused_features)
    print(f"Nguy cơ lũ lụt: {risk_score:.2f}")
    # Tạo cảnh báo
    alert, message = warning_system.generate_alert(risk_score)
    print(message)
    if use_drone:
        camera_detected, drone_detected = visualize_detected_images(
            camera_image, drone_image, camera_mask, drone_mask
        )
    else:
        colored_mask_camera = np.zeros_like(camera_image)
        colored_mask_camera[camera_mask > 0] = (0, 255, 255) 
        camera_detected = cv2.addWeighted(camera_image, 0.7, colored_mask_camera, 0.3, 0)
        cv2.imshow("Detected Camera Image", camera_detected)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # cv2.imwrite("E:/DACN/detected_camera.jpg", camera_detected)
    # cv2.imwrite("E:/DACN/detected_drone.jpg", drone_detected)
    
    # print("Kết quả đã được lưu tại E:/DACN/detected_camera.jpg và E:/DACN/detected_drone.jpg")

if __name__ == "__main__":
    main()

    











