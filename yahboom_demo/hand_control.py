import mediapipe as mp

def init_hand_detector():
    # 初始化MediaPipe手部检测
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
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


def map_hand_to_robot_coords(hand_pos):
    """将手部坐标映射到机器人坐标系"""
    # 手部工作空间范围
    hand_workspace = {
        'x_min': -200, 'x_max': 0,  # 手部X轴范围
        'y_min': -120, 'y_max': 0,   # 手部Y轴范围
        'z_min': 300, 'z_max': 400     # 手部Z轴范围
    }
    
    robot_workspace = {
        'x_min': -20, 'x_max': 20,     # 机器人X轴范围
        'y_min': 2, 'y_max': 18,   # 机器人Y轴范围
        'z_min': 2, 'z_max': 18     # 机器人Z轴范围
    }

    def linear_map(value, in_min, in_max, out_min, out_max):
        # 确保手部位置在一定范围内
        value = max(min(value, in_max), in_min)
        # 线性映射函数
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    # 应用映射并确保结果在机器人工作空间范围内
    robot_x = linear_map(hand_pos['x'], 
                       hand_workspace['x_min'], hand_workspace['x_max'],
                       robot_workspace['x_min'], robot_workspace['x_max'])
    
    # 手部Y轴范围是负的，所以需要反转
    robot_y = linear_map(hand_pos['y'],
                       hand_workspace['y_min'], hand_workspace['y_max'],
                       robot_workspace['y_max'], robot_workspace['y_min'])
    
    robot_z = linear_map(hand_pos['z'],
                       hand_workspace['z_min'], hand_workspace['z_max'],
                       robot_workspace['z_min'], robot_workspace['z_max'])
    
    return {
        'x': robot_x,
        'y': robot_z,
        'z': robot_y
    }