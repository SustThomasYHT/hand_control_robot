import numpy as np
import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR

def init_kinect():
    # 初始化Kinect
    pykinect.initialize_libraries()
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    return pykinect.start_device(config=device_config)


def get_3d_position(device, depth_image, landmarks, image_shape):
    """获取手部关键点的3D位置"""
    positions_3d = []
    valid_positions = 0
    
    # 选择要检测的关键点索引
    # 0: 手掌中心
    # 1-4: 拇指关键点
    # 9-12: 中指关键点
    # 13-16: 无名指关键点
    # 17-20: 小指关键点
    # 排除食指，因为食指用来控制夹爪了
    key_indices = [0,  # 手掌中心
                   1, 2, 3, 4,  # 拇指完整链
                   10, 11, 12,  # 中指关键点（除指尖）
                   14, 15, 16,  # 无名指关键点（除指尖）
                   18, 19, 20]  # 小指关键点（除指尖）
    
    for idx in key_indices:
        # 获取2D坐标
        x = int(landmarks[idx].x * image_shape[1])
        y = int(landmarks[idx].y * image_shape[0])
        
        # 确保坐标在图像范围内
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            # 获取该点的深度值
            depth = depth_image[y, x]
            
            if depth > 0:  # 确保深度值有效
                # 转换为3D坐标
                point_2d = pykinect.k4a_float2_t((x, y))
                point_3d = device.calibration.convert_2d_to_3d(
                    point_2d, 
                    depth, 
                    K4A_CALIBRATION_TYPE_COLOR, 
                    K4A_CALIBRATION_TYPE_COLOR
                )
                
                if point_3d is not None:
                    # 将point_3d转换为numpy数组
                    point_3d_array = np.array([point_3d.xyz.x, point_3d.xyz.y, point_3d.xyz.z])
                    positions_3d.append(point_3d_array)
                    valid_positions += 1
    
    if valid_positions > 0:
        # 将列表转换为numpy数组后再计算平均值
        positions_array = np.array(positions_3d)
        avg_position = np.mean(positions_array, axis=0)
        return avg_position, valid_positions
    
    return None, 0