import cv2
import mediapipe as mp
import numpy as np
import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR
from hand_control import map_hand_to_robot_coords
import requests
import time

def init_kinect():
    # 初始化Kinect
    pykinect.initialize_libraries()
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    return pykinect.start_device(config=device_config)

def init_hand_detector():
    # 初始化MediaPipe手部检测
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

def get_3d_position(device, depth_image, landmarks, image_shape):
    """获取手部关键点的3D位置"""
    positions_3d = []
    valid_positions = 0
    
    # 选择要检测的关键点索引（这里选择指尖和手掌中心点）
    key_indices = [4, 8, 12, 16, 20, 0]  # 拇指尖、食指尖、中指尖、无名指尖、小指尖、手掌中心
    
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

def is_one_finger(landmarks):
    """检测是否是数字1手势（只伸出食指）"""
    # 获取各个手指的关键点
    finger_tips = [8, 12, 16, 20]  # 食指、中指、无名指、小指的指尖索引
    finger_mids = [6, 10, 14, 18]  # 对应的中间关节索引
    
    # 检查食指是否伸直，其他手指是否弯曲
    is_index_up = landmarks[finger_tips[0]].y < landmarks[finger_mids[0]].y
    other_fingers_down = all(
        landmarks[finger_tips[i]].y > landmarks[finger_mids[i]].y
        for i in range(1, 4)
    )
    
    return is_index_up and other_fingers_down

def main():
    # 初始化设备
    device = init_kinect()
    hand_detector = init_hand_detector()
    
    cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
    
    # 添加这些变量
    last_send_time = 0
    MIN_SEND_INTERVAL = 0.1  # 最小发送间隔（秒）
    SERVER_URL = "http://localhost:5000"
    last_position = None
    
    # 添加夹爪控制相关变量
    current_gesture_is_one = False
    last_gripper_state = None
    
    while True:
        # 获取Kinect数据
        capture = device.update()
        
        # 获取彩色图像和深度图
        ret_color, color_image = capture.get_color_image()
        ret_depth, transformed_depth_image = capture.get_transformed_depth_image()
        
        if not ret_color or not ret_depth:
            continue
            
        # 转换图像格式用于MediaPipe处理
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)
        
        # 手部检测
        results = hand_detector.process(color_image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 检测当前手势
                gesture_is_one = is_one_finger(hand_landmarks.landmark)
                
                # 如果手势状态发生变化，发送夹爪控制请求
                if gesture_is_one != current_gesture_is_one:
                    current_gesture_is_one = gesture_is_one
                    gripper_angle = 135 if gesture_is_one else 20
                    
                    if gripper_angle != last_gripper_state:
                        try:
                            response = requests.post(
                                f"{SERVER_URL}/gripper",
                                json={"angle": gripper_angle},
                                timeout=0.1
                            )
                            if response.status_code == 200:
                                last_gripper_state = gripper_angle
                                # 在画面上显示夹爪状态
                                gripper_status = "夹紧" if gesture_is_one else "放松"
                                cv2.putText(color_image, f"夹爪状态: {gripper_status}", 
                                          (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                                          1, (0, 255, 255), 2)
                        except requests.exceptions.RequestException as e:
                            print(f"发送夹爪控制请求失败: {e}")
                
                # 获取3D位置
                position_3d, valid_count = get_3d_position(
                    device, 
                    transformed_depth_image, 
                    hand_landmarks.landmark,
                    color_image.shape
                )
                
                # 绘制手部关键点
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * color_image.shape[1])
                    y = int(landmark.y * color_image.shape[0])
                    cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)
                
                # 显示3D位置信息
                if position_3d is not None:
                    # 显示手部3D位置（白色）
                    hand_pos = {'x': position_3d[0], 'y': position_3d[1], 'z': position_3d[2]}
                    text = f"Hand Pos (mm): X:{position_3d[0]:.0f}, Y:{position_3d[1]:.0f}, Z:{position_3d[2]:.0f}"
                    cv2.putText(color_image, text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # 计算并显示机器人坐标（白色）
                    robot_pos = map_hand_to_robot_coords(hand_pos)
                    robot_text = f"Robot Pos: X:{robot_pos['x']:.2f}, Y:{robot_pos['y']:.2f}, Z:{robot_pos['z']:.2f}"
                    cv2.putText(color_image, robot_text, (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # 添加发送控制指令的逻辑
                    current_time = time.time()
                    if current_time - last_send_time >= MIN_SEND_INTERVAL:
                        try:
                            response = requests.post(
                                f"{SERVER_URL}/move",
                                json=robot_pos,
                                timeout=0.1  # 设置超时时间
                            )
                            if response.status_code == 200:
                                last_position = robot_pos
                            last_send_time = current_time
                        except requests.exceptions.RequestException as e:
                            print(f"发送请求失败: {e}")
        
        # 显示图像
        cv2.imshow('Hand Tracking', color_image)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    device.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 