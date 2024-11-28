import requests

def send_gripper_request(server_url, gripper_angle):
    try:
        response = requests.post(
            f"{server_url}/gripper",
            json={"angle": gripper_angle},
            timeout=0.1
        )
        if response.status_code == 200:
            print(f"夹爪状态更新成功: {gripper_angle}")
    except requests.exceptions.RequestException as e:
        print(f"发送夹爪控制请求失败: {e}")

def send_move_request(server_url, robot_pos):
    try:
        response = requests.post(
            f"{server_url}/move",
            json=robot_pos,
            timeout=0.1
        )
        if response.status_code == 200:
            print("机器人位置更新成功")
    except requests.exceptions.RequestException as e:
        print(f"发送请求失败: {e}")
