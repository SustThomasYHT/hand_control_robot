import paramiko
import time
import threading

def print_output(channel, prefix=''):
    """持续读取并打印channel的输出"""
    while True:
        if channel.recv_ready():
            output = channel.recv(1024).decode('utf-8', errors='ignore')
            if output:
                print(f"{prefix}{output}", end='')
        if channel.recv_stderr_ready():
            error = channel.recv_stderr(1024).decode('utf-8', errors='ignore')
            if error:
                print(f"{prefix}ERROR: {error}", end='')

def execute_ssh_commands():
    hostname = "192.168.31.43"
    username = "er"
    password = "elephant"

    try:
        # 创建两个SSH连接
        ssh1 = paramiko.SSHClient()
        ssh2 = paramiko.SSHClient()
        ssh1.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        print("正在连接服务器...")
        ssh1.connect(hostname, username=username, password=password)
        ssh2.connect(hostname, username=username, password=password)
        print("连接成功！")

        # 获取交互式shell
        shell1 = ssh1.invoke_shell()
        shell2 = ssh2.invoke_shell()

        # 设置终端大小以避免输出问题
        shell1.set_combine_stderr(True)
        shell2.set_combine_stderr(True)

        # 创建输出监控线程
        thread1 = threading.Thread(target=print_output, args=(shell1, '[Camera] '))
        thread2 = threading.Thread(target=print_output, args=(shell2, '[Server] '))
        thread1.daemon = True
        thread2.daemon = True
        thread1.start()
        thread2.start()

        # 执行命令
        commands1 = [
            "cd arm-demo/server/\n",
            "conda activate server\n",
            "python camera_server.py\n"
        ]

        commands2 = [
            "cd arm-demo/server/\n",
            "conda activate server\n",
            "python server_280.py\n"
        ]

        print("正在启动camera_server.py...")
        for command in commands1:
            shell1.send(command)
            time.sleep(1)

        print("正在启动server_280.py...")
        for command in commands2:
            shell2.send(command)
            time.sleep(1)

        print("服务器已启动，正在显示输出（按Ctrl+C终止程序）...")
        
        # 保持主程序运行
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n正在关闭连接...")
        ssh1.close()
        ssh2.close()
        print("连接已关闭")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        if 'ssh1' in locals():
            ssh1.close()
        if 'ssh2' in locals():
            ssh2.close()

if __name__ == "__main__":
    execute_ssh_commands()