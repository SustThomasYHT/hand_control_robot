from flask import Flask, request, jsonify
from flask_cors import CORS
from Arm_Lib import Arm_Device
from check4arm import *
from collections import deque
from threading import Thread, Lock
import numpy as np
import time

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})
Arm = Arm_Device()

class PositionCommand:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.timestamp = time.time()

move_queue = deque(maxlen=5)  # 减少队列长度，只保留最近的5个点
move_lock = Lock()  # 线程锁
is_moving = False
last_move_time = 0
MOVE_INTERVAL = 0.2  # 减少移动间隔到0.2秒
MAX_COMMAND_AGE = 0.5  # 指令最大年龄(秒)，超过这个时间的指令将被丢弃

def arm_move(pos, s_time=500):
    print(f"执行机械臂移动，舵机角度: {pos}")
    for i in range(4):
        id = i + 1         
        Arm.Arm_serial_servo_write(id, pos[i], int(s_time))
        time.sleep(.01)
    time.sleep(s_time / 1000)

def move2xyz(x1, y1, z1):
    print(f"计算逆运动学: x={x1:.2f}, y={y1:.2f}, z={z1:.2f}")
    Label, s1, s2, s3, s4 = backward_kinematics(x1, y1, z1)
    if Label:
        pos = [s1, s2, s3, s4, 90, 0]
        print(f"计算得到的舵机角度: {pos}")
        return pos
    else:
        print(f"超过手臂长度限制！坐标: x={x1:.2f}, y={y1:.2f}, z={z1:.2f}")
        return None

def move_to_coord(x, y, z):
    pos = move2xyz(x, y, z)
    if pos:
        arm_move(pos)
        return True
    else:
        return False

def control_gripper(angle, speed=500):
    print(f"控制夹爪，目标角度: {angle}")
    try:
        angle = max(0, min(180, angle))
        Arm.Arm_serial_servo_write(6, angle, speed)
        time.sleep(speed / 1000)
        return True
    except Exception as e:
        print(f"控制夹爪时出错: {e}")
        return False

def get_current_angles():
    """读取当前所有舵机角度，带重试机制"""
    MAX_RETRIES = 3
    angles = []
    
    for i in range(6):
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                angle = Arm.Arm_serial_servo_read(i+1)
                if angle is not None and 0 <= angle <= 180:  # 确保角度在有效范围内
                    angles.append(angle)
                    break
                retry_count += 1
                time.sleep(0.05)  # 增加重试间隔
            except Exception as e:
                print(f"读取舵机 {i+1} 角度失败: {e}")
                retry_count += 1
                time.sleep(0.05)
                
        if retry_count >= MAX_RETRIES:
            # 如果重试失败，使用上一次的有效角度或默认角度
            default_angles = [90, 90, 90, 90, 90, 90]
            angles.append(default_angles[i])
            
    return angles

def get_current_position():
    """获取当前机械臂末端的xyz坐标"""
    try:
        angles = get_current_angles()
        if len(angles) >= 4:
            valid, x, y, z = forward_kinematics(angles[0], angles[1], angles[2], angles[3])
            if valid:
                return {
                    'x': x,
                    'y': y,
                    'z': z,
                    'angles': angles
                }
        return None
    except Exception as e:
        print(f"获取位置时出错: {e}")
        return None

def is_valid_movement(current_pos, target_pos):
    """验证移动是否合理"""
    if not current_pos:
        return True
        
    # 计算移动距离
    distance = ((target_pos['x'] - current_pos['x'])**2 + 
                (target_pos['y'] - current_pos['y'])**2 + 
                (target_pos['z'] - current_pos['z'])**2)**0.5
                
    # 如果移动距离过大，认为是异常移动
    MAX_DISTANCE = 20  # 最大允许移动距离(mm)
    return distance <= MAX_DISTANCE

def clean_old_commands():
    """清理过期的指令"""
    global move_queue
    current_time = time.time()
    with move_lock:
        # 将队列转换为列表进行过滤
        valid_commands = [cmd for cmd in move_queue 
                         if current_time - cmd.timestamp <= MAX_COMMAND_AGE]
        move_queue.clear()
        move_queue.extend(valid_commands)

def smooth_coordinates(queue):
    """对队列中的坐标进行平滑处理，考虑时间因素"""
    if not queue:
        return None
    
    # 获取最新的命令
    latest_cmd = list(queue)[-1]
    
    # 如果队列中只有一个命令，直接返回
    if len(queue) == 1:
        return {'x': latest_cmd.x, 'y': latest_cmd.y, 'z': latest_cmd.z}
    
    # 对最近的点进行加权平均，越新的点权重越大
    weights = []
    points = []
    current_time = time.time()
    
    for cmd in queue:
        age = current_time - cmd.timestamp
        weight = max(0, 1 - age/MAX_COMMAND_AGE)  # 根据年龄计算权重
        weights.append(weight)
        points.append((cmd.x, cmd.y, cmd.z))
    
    # 归一化权重
    total_weight = sum(weights)
    if total_weight == 0:
        return {'x': latest_cmd.x, 'y': latest_cmd.y, 'z': latest_cmd.z}
    
    weights = [w/total_weight for w in weights]
    
    # 计算加权平均
    x = sum(p[0] * w for p, w in zip(points, weights))
    y = sum(p[1] * w for p, w in zip(points, weights))
    z = sum(p[2] * w for p, w in zip(points, weights))
    
    return {'x': x, 'y': y, 'z': z}

def move_processor():
    """后台处理移动队列的线程"""
    global is_moving, last_move_time
    
    while True:
        try:
            current_time = time.time()
            
            # 清理过期指令
            clean_old_commands()
            
            with move_lock:
                if (len(move_queue) > 0 and 
                    current_time - last_move_time >= MOVE_INTERVAL):
                    
                    smooth_pos = smooth_coordinates(move_queue)
                    if smooth_pos:
                        is_moving = True
                        try:
                            success = move_to_coord(
                                smooth_pos['x'], 
                                smooth_pos['y'], 
                                smooth_pos['z']
                            )
                            if not success:
                                print(f"移动失败: {smooth_pos}")
                        except Exception as e:
                            print(f"执行移动时出错: {e}")
                        finally:
                            last_move_time = current_time
                            is_moving = False
                            move_queue.clear()  # 移动后清空队列
                            
        except Exception as e:
            print(f"移动处理器错误: {e}")
            is_moving = False
        finally:
            time.sleep(0.01)  # 减少循环间隔以提高响应速度

@app.route('/move', methods=['POST'])
def handle_move():
    try:
        data = request.get_json()
        x = float(data.get('x', 0))
        y = float(data.get('y', 0))
        z = float(data.get('z', 0))
        
        # 创建新的位置命令
        new_command = PositionCommand(x, y, z)
        
        # 将新的移动指令添加到队列
        with move_lock:
            move_queue.append(new_command)
        
        return jsonify({
            'status': 'success',
            'message': '移动指令已加入队列'
        })
            
    except Exception as e:
        print(f"处理移动请求时出错: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/gripper', methods=['POST'])
def handle_gripper():
    try:
        data = request.get_json()
        angle = float(data.get('angle', 0))
        angle = max(20, angle)
        angle = min(135, angle)
        # speed = float(data.get('speed', 500))
        speed = 500
        
        success = control_gripper(angle, speed)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'夹爪角度已设置为 {angle}'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': '控制夹爪失败'
            }), 400
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/position', methods=['GET'])
def handle_get_position():
    try:
        position = get_current_position()
        if position:
            return jsonify({
                'status': 'success',
                'data': position
            })
        else:
            return jsonify({
                'status': 'error',
                'message': '无法获取有效位置'
            }), 400
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # 启动移动处理线程
    move_thread = Thread(target=move_processor, daemon=True)
    move_thread.start()
    
    app.run(host='0.0.0.0', port=5000) 