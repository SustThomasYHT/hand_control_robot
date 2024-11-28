import cv2
import time
import concurrent.futures

from hand_control import init_hand_detector, map_hand_to_robot_coords, is_one_finger
from kinect_control import init_kinect, get_3d_position
from request_control import send_gripper_request, send_move_request


def main():
    # 初始化设备
    device = init_kinect()
    hand_detector = init_hand_detector()
    
    cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
    
    # 添加这些变量
    last_send_time = 0
    MIN_SEND_INTERVAL = 0.1  # 最小发送间隔（秒）
    SERVER_URL = "http://192.168.31.155:5000"
    last_position = None
    
    # 添加夹爪控制相关变量
    current_gesture_is_one = False
    last_gripper_state = None
    
    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor() as executor:
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
                            # 使用线程池异步发送请求
                            executor.submit(send_gripper_request, SERVER_URL, gripper_angle)
                            last_gripper_state = gripper_angle
                    
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
                            # 使用线程池异步发送请求
                            executor.submit(send_move_request, SERVER_URL, robot_pos)
                            last_send_time = current_time
            
            # 显示图像
            cv2.imshow('Hand Tracking', color_image)
            
            if cv2.waitKey(1) == ord('q'):
                break
        
        device.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 