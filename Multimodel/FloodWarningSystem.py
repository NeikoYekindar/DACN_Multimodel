import matplotlib.pyplot as plt
import cv2
import numpy as np

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