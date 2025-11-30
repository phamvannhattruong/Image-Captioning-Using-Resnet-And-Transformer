import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import timm
from ultralytics import YOLO
from distance.coordinates import Distance3DCalculator

yolo_model = YOLO("yolo11n.pt") # Đảm bảo file model đúng tên
img = cv2.imread("example.jpg")

if img is None:
    print("Không tìm thấy ảnh!")
    exit()

h, w, _ = img.shape
distance_calculator = Distance3DCalculator(focal_length=1.0, image_width=w, image_height=h)

# 2. Chạy YOLO
results = yolo_model(img)

# 3. Xử lý từng vật thể
# Ultralytics trả về list kết quả (mỗi ảnh 1 kết quả), ta lấy results[0]
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from distance.coordinates import Distance3DCalculator  # Giả sử file này chứa class bạn đã viết

# 1. Khởi tạo
yolo_model = YOLO("yolo11n.pt") 
img = cv2.imread("example.jpg")

if img is None:
    print("Không tìm thấy ảnh!")
    exit()

h, w, _ = img.shape
# Lưu ý: Cần chỉnh lại x_angle, y_angle cho đúng FOV camera của bạn
distance_calculator = Distance3DCalculator(focal_length=1.0, image_width=w, image_height=h)

# 2. Chạy YOLO
results = yolo_model(img)

detected_objects = []  # Lưu thông tin các vật thể đã phát hiện

full_depth_map = distance_calculator.get_depth_map(img)

# 3. Xử lý từng vật thể
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Lấy tọa độ bounding box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # Safety check
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Cắt ảnh ROI
        roi_img_bgr = img[y1:y2, x1:x2]
        if roi_img_bgr.size == 0: continue
        
        z_value = np.median(full_depth_map[y1:y2, x1:x2]) # Giá trị Z tương đối
        
        # Tính tâm box (Pixel)
        cx_box = int((x1 + x2) / 2)
        cy_box = int((y1 + y2) / 2)
        
        # Tính tọa độ 3D (X, Y, Z)
        # ang_x, ang_y = distance_calculator.calculate_angle_from_center(cx_box, cy_box)
        X, Y, Z = distance_calculator.calculate_3d_coordinates(z_value)
        print (f"Object: {yolo_model.names[int(box.cls[0])]}, 3D Coord: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}")
        # Lưu thông tin vào list
        obj_info = {
            "label": yolo_model.names[int(box.cls[0])],
            "center_pixel": (cx_box, cy_box), # Để vẽ đường trên ảnh 2D
            "coord_3d": (X, Y, Z),            # Để tính khoảng cách thực tế
            "bbox": (x1, y1, x2, y2)
        }
        detected_objects.append(obj_info)

        # Vẽ Bounding Box & Tọa độ 3D của chính nó
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (cx_box, cy_box), 5, (0, 0, 255), -1) # Chấm đỏ ở tâm
        cv2.putText(img, f"Z:{Z:.1f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 4. Tính khoảng cách giữa CÁC CẶP vật thể
# Sử dụng 2 vòng lặp để nối từng cặp
for i in range(len(detected_objects)):
    for j in range(i + 1, len(detected_objects)):
        obj1 = detected_objects[i]
        obj2 = detected_objects[j]
        
        # Lấy tọa độ 3D
        p1_3d = obj1["coord_3d"]
        p2_3d = obj2["coord_3d"]
        
        # Tính khoảng cách Euclidean trong không gian 3D
        dist_3d = distance_calculator.calculate_euclidean_distance(p1_3d, p2_3d)
        
        # --- VẼ LÊN ẢNH ---
        pt1 = obj1["center_pixel"]
        pt2 = obj2["center_pixel"]
        
        # Vẽ đường nối màu vàng
        cv2.line(img, pt1, pt2, (0, 255, 255), 2)
        
        # Tính vị trí trung điểm để viết chữ
        mid_x = int((pt1[0] + pt2[0]) / 2)
        mid_y = int((pt1[1] + pt2[1]) / 2)
        
        # Viết khoảng cách lên đường nối
        dist_text = f"{dist_3d:.1f}"
        cv2.putText(img, dist_text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        print(f"Distance {obj1['label']} <-> {obj2['label']}: {dist_3d:.2f}")

# 5. Hiển thị
cv2.imshow("Distance Measurement", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
