# Azure Kinect DK 手部追踪项目

## 项目概述
- **设备**：Azure Kinect DK
- **目标**：使用 pykinect-azure 库获取手部姿势数据，用于机械臂控制
- **期望输出**：
  - 手掌根部的 3D 坐标
  - 手部骨骼关键点的 3D 坐标

## 环境配置

### 1. 官方文档
- [Azure Kinect DK 中文文档](https://learn.microsoft.com/zh-cn/previous-versions/azure/kinect-dk/)

### 2. 必要 SDK 下载
1. **Azure Kinect SDK 1.4.0**
   - 用途：连接深度相机，下载这个版本是因为pykinect-azure库默认支持这个版本，其他版本需要手动在初始化时配置
   - [下载链接](https://download.microsoft.com/download/4/5/a/45aa3917-45bf-4f24-b934-5cff74df73e1/Azure%20Kinect%20SDK%201.4.0.exe)

2. **Azure Kinect Body Tracking SDK v1.1.2**
   - 用途：官方提供的人体跟踪功能
   - [下载链接](https://www.microsoft.com/en-us/download/details.aspx?id=104221)

### 3. Python 库配置
#### pyKinectAzure 库 控制 Azure Kinect DK
- 安装命令：`pip install pykinect_azure`
- 开发参考：可在 pyKinectAzure 项目的 [examples 目录](https://github.com/ibaiGorordo/pyKinectAzure/tree/master/examples)下查看示例代码

#### mediapipe 库 手部关键点检测
- 安装命令：`pip install mediapipe`
- 开发参考：[MediaPipe 官方文档](https://mediapipe.dev/getting_started/python)

## 运行项目
### 1. 运行 `hand_control.py` 文件
  - 可以看到一个视频窗口，显示关键信息，包括左右手的掌根3D位置以及手指张开程度
  - 终端打印出 json 格式的信息，详细见 [hand_control.py#L146](hand_control.py#L146)
