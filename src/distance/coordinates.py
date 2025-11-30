import numpy as np
import torch
import math
import timm


class Distance3DCalculator:
    def __init__(self, focal_length, image_width, image_height, x_angle=60, y_angle=45):
        self.x_angle = x_angle
        self.y_angle = y_angle
        self.focal_length = focal_length
        self.cx = image_width / 2
        self.cy = image_height / 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load MiDaS
        self.midas = torch.hub.load("intel-isl/MiDas", "MiDaS_small").to(self.device)
        self.midas.eval()
        self.midas_transforms = torch.hub.load("intel-isl/MiDas", "transforms")
        self.transform = self.midas_transforms.small_transform

    def get_depth_map(self, image):
        input_batch = self.transform(image).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.cpu().numpy()
    
    def calculate_angle_from_center(self, x_center_box, y_center_box):
        x_norm = (x_center_box - self.cx) / self.cx 
        y_norm = (y_center_box - self.cy) / self.cy
        angle_x = x_norm * (self.x_angle / 2)
        angle_y = y_norm * (self.y_angle / 2)
        return angle_x, angle_y

    def calculate_3d_coordinates(self, depth_z):
        angle_x_rad = math.radians(self.x_angle)
        angle_y_rad = math.radians(self.y_angle)
        x = depth_z * math.tan(angle_x_rad)
        y = depth_z * math.tan(angle_y_rad)
        z = depth_z*7 # Chuyển đổi đơn vị nếu cần
        return x, y, z
    
    # Hàm tính khoảng cách giữa 2 điểm 3D
    def calculate_euclidean_distance(self, p1, p2):
        # p1, p2 là tuple (x, y, z)
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)