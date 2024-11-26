import cv2
import numpy as np
import pykinect_azure as pykinect
import time
import math
import json
import requests
from pymycobot import MyCobot280Socket

def calculate_finger_distance_kinect(body, hand_type='right'):
    """使用Kinect的骨骼追踪数据计算手指捏合程度
    
    Args:
        body: Kinect骨骼数据
        hand_type: 'right' 或 'left'，指定要计算的手
        
    Returns:
        float: 归一化后的手指捏合程度 (0.0-1.0)，0表示完全闭合，1表示完全张开
    """
    try:
        if hand_type == 'right':
            thumb_tip = body.joints[pykinect.K4ABT_JOINT_HANDTIP_RIGHT]
            middle_tip = body.joints[pykinect.K4ABT_JOINT_THUMB_RIGHT]
            palm = body.joints[pykinect.K4ABT_JOINT_WRIST_RIGHT]
        else:
            thumb_tip = body.joints[pykinect.K4ABT_JOINT_HANDTIP_LEFT]
            middle_tip = body.joints[pykinect.K4ABT_JOINT_THUMB_LEFT]
            palm = body.joints[pykinect.K4ABT_JOINT_WRIST_LEFT]

        # 计算拇指到中指的距离
        finger_distance = math.sqrt(
            (thumb_tip.position.x - middle_tip.position.x) ** 2 +
            (thumb_tip.position.y - middle_tip.position.y) ** 2 +
            (thumb_tip.position.z - middle_tip.position.z) ** 2
        )

        # 计算手掌大小作为参考（从手腕到中指尖的距离）
        hand_size = math.sqrt(
            (palm.position.x - middle_tip.position.x) ** 2 +
            (palm.position.y - middle_tip.position.y) ** 2 +
            (palm.position.z - middle_tip.position.z) ** 2
        )

        # 归一化距离
        if hand_size > 0:
            normalized_distance = min(1.0, max(0.0, finger_distance / hand_size))
        else:
            normalized_distance = 0.0

        return normalized_distance

    except Exception as e:
        print(f"计算手指距离时出错: {str(e)}")
        return None

def draw_coordinate_system(image, body, body_frame, right_gripper_value=None, left_gripper_value=None):
    """在图像上绘制坐标系参考、距离信息和左右手状态"""
    # 获取图像尺寸
    height, width = image.shape[:2]
    
    # 在图像中心绘制一个十字线表示相机原点
    """
    xyz对应的方向是：
    x: 水平方向
    y: 垂直方向
    z: 垂直于屏幕方向
    """
    center_x, center_y = width // 2, height // 2
    cv2.line(image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)  # X轴
    cv2.line(image, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)  # Y轴
    cv2.putText(image, "Camera Origin", (center_x + 25, center_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if body is not None:
        # 获取左右手腕位置
        right_wrist = body.joints[pykinect.K4ABT_JOINT_WRIST_RIGHT]
        left_wrist = body.joints[pykinect.K4ABT_JOINT_WRIST_LEFT]
        
        # 右手信息（显示在右上角）
        right_info_text = [
            f"Right Hand Distance:",
            f"X: {right_wrist.position.x:>6.2f} mm",
            f"Y: {right_wrist.position.y:>6.2f} mm",
            f"Z: {right_wrist.position.z:>6.2f} mm"
        ]
        
        # 左手信息（显示在左上角）
        left_info_text = [
            f"Left Hand Distance:",
            f"X: {left_wrist.position.x:>6.2f} mm",
            f"Y: {left_wrist.position.y:>6.2f} mm",
            f"Z: {left_wrist.position.z:>6.2f} mm"
        ]
        
        # 在右上角显示右手信息
        y_offset = 30
        for text in right_info_text:
            # 计算文本宽度以便正确放置在右侧
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            x_position = width - text_size[0] - 10  # 10是边距
            cv2.putText(image, text, (x_position, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # 在左上角显示左手信息
        y_offset = 30
        for text in left_info_text:
            cv2.putText(image, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # 显示右手夹爪信息
        if right_gripper_value is not None:
            text = f"Right Gripper: {right_gripper_value*100:.0f}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            x_position = width - text_size[0] - 10
            cv2.putText(image, text,
                       (x_position, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7,
                       (0, 255, 0),
                       2)
        
        # 显示左手夹爪信息
        if left_gripper_value is not None:
            text = f"Left Gripper: {left_gripper_value*100:.0f}%"
            cv2.putText(image, text,
                       (10, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7,
                       (0, 255, 0),
                       2)
            
        # 把信息整理成json格式打印出来
        hand_info = {
            "timestamp": time.time(),
            "right_hand": {
                "position": {
                    "x": float(right_wrist.position.x),
                    "y": float(right_wrist.position.y),
                    "z": float(right_wrist.position.z)
                },
                "gripper_value": float(right_gripper_value) if right_gripper_value is not None else None
            }
        }
        
    
    return image

def map_hand_to_robot_coords(hand_pos):
    """将手部坐标映射到机器人坐标系"""
    # 手部工作空间范围
    hand_workspace = {
        'x_min': -300, 'x_max': -150,  # 手部X轴范围
        'y_min': -150, 'y_max': 150,   # 手部Y轴范围
        'z_min': 600, 'z_max': 800     # 手部Z轴范围
    }
    
    # 机器人工作空间范围
    robot_workspace = {
        'x_min': -80, 'x_max': 80,     # 机器人X轴范围
        'y_min': -200, 'y_max': -80,   # 机器人Y轴范围
        'z_min': 140, 'z_max': 210     # 机器人Z轴范围
    }

    def linear_map(value, in_min, in_max, out_min, out_max):
        # 确保输入值在范围内
        value = max(min(value, in_max), in_min)
        # 线性映射函数
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    # 应用映射并确保结果在机器人工作空间范围内
    robot_x = linear_map(hand_pos['x'], 
                       hand_workspace['x_min'], hand_workspace['x_max'],
                       robot_workspace['x_min'], robot_workspace['x_max'])
    
    robot_y = linear_map(hand_pos['y'],
                       hand_workspace['y_min'], hand_workspace['y_max'],
                       robot_workspace['y_min'], robot_workspace['y_max'])
    
    robot_z = linear_map(hand_pos['z'],
                       hand_workspace['z_min'], hand_workspace['z_max'],
                       robot_workspace['z_min'], robot_workspace['z_max'])
    
    return {
        'x': robot_x,
        'y': robot_y,
        'z': robot_z
    }

def control_gripper(mc, gripper_value):
    """
    根据手指捏合程度控制机械臂夹爪
    
    Args:
        mc: MyCobot280Socket实例
        gripper_value: 手指捏合程度 (0.0-1.0)，0表示完全闭合，1表示完全张开
    """
    try:
        # 将gripper_value反转并映射到夹爪控制范围
        # 当手指完全张开(1.0)时夹爪打开(0)，手指闭合(0.0)时夹爪关闭(1)
        gripper_state = 1 if gripper_value < 0.5 else 0
        
        # 设置夹爪状态，速度固定为80
        mc.set_gripper_state(gripper_state, 80)
        
        # 打印夹爪状态
        status = "关闭" if gripper_state == 1 else "打开"
        print(f"夹爪状态: {status} (手指捏合度: {gripper_value:.2f})")
        
    except Exception as e:
        print(f"控制夹爪时出错: {str(e)}")

def main():
    # 初始化机械臂
    robot_ip = "192.168.31.43"  # 根据实际IP修改
    mc = MyCobot280Socket(robot_ip, 9001)
    
    # 初始化库，启用身体追踪功能
    pykinect.initialize_libraries(track_body=True)

    # 配置相机
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

    # 启动设备
    device = pykinect.start_device(config=device_config)

    # 启动身体追踪器
    bodyTracker = pykinect.start_body_tracker()

    cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
    
    # 调整最大显示尺寸
    MAX_DISPLAY_WIDTH = 640
    MAX_DISPLAY_HEIGHT = 480
    
    # 用于存储最新的夹爪数据
    current_right_gripper_value = None
    current_left_gripper_value = None
    
    while True:
        # 获取捕获帧
        capture = device.update()

        # 获取身体追踪帧
        body_frame = bodyTracker.update()

        # 获取彩色图像
        ret_color, color_image = capture.get_color_image()

        # 获取深度图像
        ret_depth, depth_image = capture.get_colored_depth_image()

        if not ret_color or not ret_depth:
            continue
            
        # 调整深度图像尺寸以匹配彩色图像
        depth_image_resized = cv2.resize(depth_image, 
                                       (color_image.shape[1], color_image.shape[0]))

        # 确保深度图像有3个通道
        if len(depth_image_resized.shape) == 2:
            depth_image_resized = cv2.cvtColor(depth_image_resized, cv2.COLOR_GRAY2BGR)
        
        # 合并彩色和深度图像
        try:
            combined_image = cv2.addWeighted(color_image[:,:,:3], 0.6, 
                                           depth_image_resized, 0.4, 0)
            
            # 绘制骨骼
            combined_image = body_frame.draw_bodies(combined_image)
            
            # 水平翻转图像
            combined_image = cv2.flip(combined_image, 1)  # 1表示水平翻转
            
            # 获取原始图像尺寸
            height, width = combined_image.shape[:2]
            
            # 计算缩放比例
            scale = min(MAX_DISPLAY_WIDTH/width, MAX_DISPLAY_HEIGHT/height)
            
            # 如果图像太大，就按比例缩小
            if scale < 1:
                new_width = int(width * scale)
                new_height = int(height * scale)
                combined_image = cv2.resize(combined_image, (new_width, new_height))
                cv2.resizeWindow('Hand Tracking', new_width, new_height)
            else:
                cv2.resizeWindow('Hand Tracking', width, height)
                
        except Exception as e:
            print(f"Color image shape: {color_image.shape}")
            print(f"Depth image shape: {depth_image_resized.shape}")
            raise e
            
        # 获取身体追踪数据
        num_bodies = body_frame.get_num_bodies()
        
        if num_bodies > 0:
            body = body_frame.get_body(0)
            
            # 计算右手夹爪值
            right_gripper = calculate_finger_distance_kinect(body, 'right')
            if right_gripper is not None:
                current_right_gripper_value = right_gripper
                # 控制夹爪
                control_gripper(mc, current_right_gripper_value)
            
            # 计算左手夹爪值
            left_gripper = calculate_finger_distance_kinect(body, 'left')
            if left_gripper is not None:
                current_left_gripper_value = left_gripper
            
            # 获取右手腕位置
            right_wrist = body.joints[pykinect.K4ABT_JOINT_WRIST_RIGHT]
            
            # 构造手部位置字典
            hand_pos = {
                'x': float(right_wrist.position.x),
                'y': float(right_wrist.position.y),
                'z': float(right_wrist.position.z)
            }
            
            # 映射到机器人坐标系
            robot_pos = map_hand_to_robot_coords(hand_pos)
            
            # 控制机械臂移动
            mc.send_coords([
                robot_pos['x'],
                robot_pos['y'], 
                robot_pos['z'],
                0, 180, 90  # 保持末端姿态不变
            ], 20, 0)
            
            # 更新显示
            combined_image = draw_coordinate_system(
                combined_image, 
                body, 
                body_frame,
                right_gripper_value=current_right_gripper_value,
                left_gripper_value=current_left_gripper_value
            )
        else:
            combined_image = draw_coordinate_system(combined_image, None, body_frame)
            
        # 显示图像
        cv2.imshow('Hand Tracking', combined_image)

        # 按q键退出
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 