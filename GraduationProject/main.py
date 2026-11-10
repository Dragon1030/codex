# main.py
import tkinter as tk
from login_window import LoginWindow
if __name__ == "__main__":
    root = tk.Tk()
    root.title("乒乓球初学者动作智能评估系统")

    # 窗口居中
    window_width = 1350
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # 启动登录界面
    LoginWindow(root)
    root.mainloop()