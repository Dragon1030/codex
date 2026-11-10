# 乒乓球初学者动作智能评估系统

这是一个基于 Python/Tkinter 的桌面应用，用于对乒乓球练习视频进行姿态分析和动作质量评估。程序使用 OpenCV 读取视频，使用 MediaPipe Pose 提取人体关键点，并通过角度、肩部对齐和重心转移等指标给出正手攻球、反手推挡和发球的分析结果。

## 功能

- 用户登录和注册
- 视频文件选择、逐帧姿态分析和实时进度显示
- 正手攻球、反手推挡、发球动作识别
- 肘部、膝部、腰部、肩部对齐和重心转移指标
- 关键点、角度曲线、统计图表和三维姿态可视化
- 分析历史、训练进度和文字反馈
- SQLite 保存用户、登录、分析和进度记录

## 目录

```text
GraduationProject/
├── main.py                         # 程序入口
├── login_window.py                 # 登录和注册界面
├── main_window.py                  # 主界面和视频分析逻辑
├── database.py                     # SQLite 数据库访问
├── assets/                         # 登录界面背景图
├── theme/                          # Tk 主题资源
├── Azure-ttk-theme-main/           # Azure 主题源码和示例
└── Sun-Valley-ttk-theme-main/      # Sun Valley 主题源码和示例
```

运行时会在当前目录生成以下文件，这些文件不属于源代码：

```text
pingpong_analysis.db               # SQLite 数据库
pingpong_analysis.log              # 运行日志
```

## 环境要求

- Python 3.9 或更高版本
- 支持 Tkinter 的桌面系统
- Windows、macOS 或带图形显示的 Linux
- 可选摄像头或本地视频文件

服务器上的无显示 Docker 容器不能直接显示 Tkinter 窗口。需要在本地桌面运行，或者配置 X11/远程桌面后再运行。

## 安装

在项目根目录执行：

```bash
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Linux 如果缺少 Tkinter，需要先安装系统包：

```bash
sudo apt-get install python3-tk
```

## 启动

必须在项目根目录运行，这样相对路径 `assets/` 和 `theme/` 才能正确找到：

```bash
python main.py
```

首次启动时，程序会自动创建 `pingpong_analysis.db`，并创建测试账号：

```text
用户名：admin
密码：123
```

登录后可以在界面中注册新账号。实际使用时建议修改或删除默认账号。

## 使用流程

1. 启动 `main.py` 并登录。
2. 进入视频分析页面，选择本地视频文件。
3. 选择动作类型或使用自动识别。
4. 开始分析，等待视频处理完成。
5. 在可视化、动作分析、反馈和历史记录页面查看结果。
6. 分析记录会写入当前目录的 SQLite 数据库。

## 常见问题

### 窗口无法显示

确认当前机器有桌面显示环境，并检查：

```bash
python -c "import tkinter; print('tkinter ok')"
```

通过 SSH 连接的纯命令行服务器通常不能直接运行此 GUI。

### 找不到背景图或主题

不要从其他目录直接执行脚本。先进入项目根目录：

```bash
cd /path/to/GraduationProject
python main.py
```

### MediaPipe 或 OpenCV 安装失败

优先使用 Python 3.9/3.10 的 64 位环境，并确认 `pip` 与运行程序的 `python` 属于同一个环境：

```bash
python -m pip --version
python --version
```

## 说明

本项目是桌面端动作分析应用，不是模型训练脚本。项目中的高分参考曲线包含演示数据；实际评估结果来自用户选择的视频和 MediaPipe 姿态关键点。
