import torch
import time
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load model
model = YOLO('model/best.pt')
model.fuse()
model.to('cpu')  # Nếu muốn chạy trên CPU

# Load ảnh và xử lý
image_pil = Image.open('image/flood-drone.jpg').resize((640, 640)).convert('RGB')
image_np = np.array(image_pil) / 255.0
image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).float()

# Warm-up
for _ in range(5):
    _ = model(image_tensor)

# Benchmark
times = []
with torch.no_grad():
    for _ in range(100):
        start = time.time()
        results = model(image_tensor)
        end = time.time()
        times.append(end - start)

avg_latency = sum(times) / len(times)
fps = 1 / avg_latency

print(f"Avg latency: {avg_latency * 1000:.2f} ms")
print(f"FPS: {fps:.2f}")

# ----- Hiển thị ảnh sau detect -----
# Lấy ảnh có bounding boxes
detected_img = results[0].plot()  # trả về numpy array với box + label

# Hiển thị bằng OpenCV
cv2.imshow("Detected Image", cv2.cvtColor(detected_img, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
