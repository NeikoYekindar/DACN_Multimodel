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
import datetime
import time
import FloodWarningSystem
from FusionNetwork import FusionNetwork
from FusionNetwork import WeightedFusion
import ImageProcessor
import WaterSegmentationModel

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

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
    if water_mask_drone.shape[:2] != drone_image.shape[:2]:
            water_mask_drone = cv2.resize(water_mask_drone, (drone_image.shape[1], drone_image.shape[0]), interpolation=cv2.INTER_NEAREST)
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

def is_drone_flying_now():
    now = datetime.datetime.now().time()
    morning = (datetime.time(6, 0), datetime.time(6, 10))
    noon = (datetime.time(12, 0), datetime.time(12, 10))
    afternoon = (datetime.time(14, 30), datetime.time(14, 35))

    def in_range(t, time_range): 
        return time_range[0]<=t<=time_range[1]
    return any(in_range(now, r) for r in [morning, noon, afternoon])

def wait_for_drone_image(drone_image_path, timeout=30, interval=3):
    print("Đang chờ ảnh từ drone ... ")
    waited = 0
    while waited < timeout:
        if os.path.exists(drone_image_path):
            print("Ảnh drone đã sẵn sàng!")
            return True
        time.sleep(interval)
        waited += interval
    return False




def main():
    drone_flying = True  # hoặc False tùy thời điểm thực tế
    # drone_flying = is_drone_flying_now()

    camera_model_path = "model/camera.pt"
    drone_model_path = "model/best.pt"

    camera_image_path = "image\camera\camera2.png"
    drone_image_path = "image\drone\drone2.png"

    image_processor = ImageProcessor.ImageProcessor()

    water_segmentation_model = WaterSegmentationModel.WaterSegmentationModel(camera_model_path, drone_model_path)
    weight_fusion = WeightedFusion()
    warning_system = FloodWarningSystem.FloodWarningSystem(risk_threshold=0.5)

    #xử lý hình ảnh

    if os.path.exists(camera_image_path):
        camera_image = image_processor.load_image(camera_image_path)
        assert camera_image is not None, "Không thể đọc ảnh từ camera"
        camera_image = image_processor.preprocess_image(camera_image)
        # phân đoạn vùng nước
        #camera_mask, camera_area, camera_percentage = water_segmentation_model.segment_water(camera_image)
        start_time = time.time()
        camera_mask, camera_area, camera_percentage = water_segmentation_model.segment_water(camera_image)
        print(f"Diện tích vùng nước từ camera: {camera_area}, Phần trăm: {camera_percentage:.2f}%")
    
    # if os.path.exists(drone_image_path):
    #     drone_image = image_processor.load_image(drone_image_path)
    #     assert drone_image is not None, "Không thể đọc ảnh từ drone"
    #     drone_image = image_processor.preprocess_image(drone_image)
    #     # phân đoạn vùng nước
    #     #camera_mask, camera_area, camera_percentage = water_segmentation_model.segment_water(camera_image)
    #     drone_mask, drone_area, drone_percentage = water_segmentation_model.segment_water(drone_image)
    #     print(f"Diện tích vùng nước từ camera: {drone_area}, Phần trăm: {drone_percentage:.2f}%")

    if drone_flying and os.path.exists(drone_image_path):
        if not wait_for_drone_image(drone_image_path):
            raise FileNotFoundError("Drone đang bay nhưng không nhận được ảnh sau 30 giây!")
        # Kiểm tra xem ảnh drone có tồn tại không (cẩn thận khi drone_flying = True nhưng chưa có ảnh)
        if not os.path.exists(drone_image_path):
            raise FileNotFoundError("Drone đang bay nhưng chưa có ảnh đầu vào từ drone!")
        drone_image = image_processor.load_image(drone_image_path)
        assert drone_image is not None, "Không thể đọc ảnh từ drone"
        print("Ảnh đã được đọc thành công!")
        drone_image = image_processor.preprocess_image(drone_image)
        

        drone_mask, drone_area, drone_percentage = water_segmentation_model.segment_water(drone_image, is_drone=True)
        print(f"Diện tích vùng nước từ drone: {drone_area}, Phần trăm: {drone_percentage:.2f}%")
        # tính sự thay đổi diện tích nước
        delta_camera, delta_drone = water_segmentation_model.calculate_area_change(camera_area, drone_area)

        print(f"Sự chênh lệch vùng nước từ camera: {delta_camera}")
        print(f"Sự chênh lệch vùng nước từ drone: {delta_drone}")
        if delta_camera < 0 and delta_drone < 0:
            print("Cả camera và drone đều giảm diện tích vùng nước. An toàn.")
            alert, message = warning_system.generate_alert(0.0)  # Cho điểm rủi ro bằng 0
            print(message)

            # Hiển thị ảnh phân đoạn để xác nhận
            camera_detected, drone_detected = visualize_detected_images(
                camera_image, drone_image, camera_mask, drone_mask
            )
            return
         # tính trọng só
        w_camera, w_drone = weight_fusion.calculate_weights(delta_camera, delta_drone)
        print(f"Trọng số vùng nước từ camera: {w_camera}")
        print(f"Trọng số vùng nước từ drone: {w_drone}")

        
        # Nếu chỉ một trong hai là âm → bỏ qua nguồn đó
        use_camera = delta_camera > 0
        use_drone = delta_drone > 0

        if not use_camera and use_drone:
            print("Bỏ qua ảnh camera vì diện tích vùng nước giảm.")
            drone_feature = weight_fusion.extract_feature_drone(drone_image, water_segmentation_model.drone_model)
            drone_feature = drone_feature / (drone_image.shape[0] * drone_image.shape[1])
            print(f"Feature vùng nước từ drone: {drone_feature}")
            risk_score = weight_fusion.predict_drone_only_risk(drone_feature)

        elif not use_drone and use_camera:
            print("Bỏ qua ảnh drone vì diện tích vùng nước giảm.")
            camera_feature = weight_fusion.extract_feature_camera(camera_image, water_segmentation_model.camera_model)
            camera_feature = camera_feature / (camera_image.shape[0] * camera_image.shape[1])
            print(f"Feature vùng nước từ camera: {camera_feature}")
            risk_score = weight_fusion.predict_camera_only_risk(camera_feature)

        elif use_camera and use_drone:
            # Trường hợp cả hai nguồn đều hợp lệ
            w_camera, w_drone = weight_fusion.calculate_weights(delta_camera, delta_drone)
            print(f"Trọng số vùng nước từ camera: {w_camera}")
            print(f"Trọng số vùng nước từ drone: {w_drone}")

            camera_feature, drone_feature = weight_fusion.extract_features(
                camera_image, drone_image, 
                water_segmentation_model.camera_model, 
                water_segmentation_model.drone_model
            )

            camera_img_area = camera_image.shape[0] * camera_image.shape[1]
            drone_img_area = drone_image.shape[0]*drone_image.shape[1]
            camera_feature = camera_feature / camera_img_area
            drone_feature = drone_feature / drone_img_area

            print(f"Feature vùng nước từ camera: {camera_feature}")
            print(f"Feature vùng nước từ drone: {drone_feature}")
            # Kết hợp đặc trưng




            fused_features = weight_fusion.fuse_features(camera_feature, drone_feature, w_camera, w_drone)
            # risk_score = weight_fusion.predict_flood_risk(fused_features)
            risk_score = w_camera * camera_feature.item() + w_drone * drone_feature.item()


        elif not use_drone and not use_camera:
            risk_score = 0
    elif not os.path.exists(camera_image_path) and os.path.exists(drone_image_path):
        print("Không tìm thấy ảnh camera. Sử dụng dữ liệu từ drone để phân tích.")
        drone_features = weight_fusion.extract_feature_drone(drone_image, water_segmentation_model.drone_model)
        if drone_features > water_segmentation_model.reference_drone_area:
            risk_score = 1
        else:
            risk_score = 0

        # drone_features = drone_feature / (drone_image.shape[0] * drone_image.shape[1])
        # print(f"Feature vùng nước từ drone: {drone_features}")
        # fused_features = drone_features
        # risk_score = weight_fusion.predict_drone_only_risk(fused_features)

    elif os.path.exists(camera_image_path) and not os.path.exists(drone_image_path):
        print("Không tìm thấy ảnh drone. Sử dụng dữ liệu từ camera để phân tích.")
        camera_feature = weight_fusion.extract_feature_camera(camera_image, water_segmentation_model.camera_model)
        # camera_feature = camera_feature / (camera_image.shape[0] * camera_image.shape[1])

        # print(f"Feature vùng nước từ camera: {camera_feature}")
        # fused_features = camera_feature
        # risk_score = weight_fusion.predict_camera_only_risk(fused_features)
        if camera_feature > water_segmentation_model.reference_camera_area:
            risk_score = 1
        else:
            risk_score = 0
    

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Thời gian xử lý ảnh: {processing_time:.2f} giây")

    # Dự đoán điểm rủi ro ngập lụt
    #risk_score = weight_fusion.predict_flood_risk(fused_features)
    print(f"Nguy cơ lũ lụt: {risk_score:.2f}")
    # Tạo cảnh báo
    alert, message = warning_system.generate_alert(risk_score)
    print(message)
    camera_yolo_results = water_segmentation_model.camera_model(camera_image)
    camera_annotated = camera_yolo_results[0].plot()
    cv2.imshow("YOLO Segment + BBox (Camera)", camera_annotated)

    if drone_flying and os.path.exists(drone_image_path):
        drone_yolo_results = water_segmentation_model.drone_model(drone_image)
        drone_annotated = drone_yolo_results[0].plot()
        cv2.imshow("YOLO Segment + BBox (Drone)", drone_annotated)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # if drone_flying and os.path.exists(drone_image_path):
    #     camera_detected, drone_detected = visualize_detected_images(
    #         camera_image, drone_image, camera_mask, drone_mask
    # )
    # elif not os.path.exists(camera_image_path):
        
    #     colored_mask_drone = np.zeros_like(drone_image)
    #     colored_mask_drone[drone_mask > 0] = (0,255,255)
    #     drone_detected =cv2.addWeighted(drone_image, 0.7, colored_mask_drone, 0.3, 0)
    #     cv2.imshow("Detected Camera Image", drone_detected)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # else:
    #     colored_mask_camera = np.zeros_like(camera_image)
    #     colored_mask_camera[camera_mask > 0] = (0, 255, 255) 
    #     camera_detected = cv2.addWeighted(camera_image, 0.7, colored_mask_camera, 0.3, 0)
    #     cv2.imshow("Detected Camera Image", camera_detected)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # cv2.imwrite("E:/DACN/detected_camera.jpg", camera_detected)
    # cv2.imwrite("E:/DACN/detected_drone.jpg", drone_detected)
    
    # print("Kết quả đã được lưu tại E:/DACN/detected_camera.jpg và E:/DACN/detected_drone.jpg")

if __name__ == "__main__":
    main()

    











