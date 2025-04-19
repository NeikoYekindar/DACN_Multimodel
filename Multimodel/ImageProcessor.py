import cv2
import os


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