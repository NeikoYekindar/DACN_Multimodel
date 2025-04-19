import os
from ultralytics import YOLO
import numpy as np

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