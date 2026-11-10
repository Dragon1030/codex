# login_window.py
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import database


class LoginWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("乒乓球初学者动作智能评估系统 - 登录")
        self.master.geometry("1350x800")
        self.master.resizable(False, False)
        self.db = database.PingPongDB()

        # 设置窗口透明度（使用Tkinter支持的alpha值）
        self.master.attributes('-alpha', 0.35)

        # 加载背景图
        try:
            self.bg_image = Image.open("assets/bg.jpg")
            self.bg_photo = ImageTk.PhotoImage(self.bg_image.resize((1350, 800), Image.LANCZOS))
        except Exception as e:
            print(f"加载背景图失败: {e}")
            self.bg_photo = None

        self.setup_ui()

    def setup_ui(self):
        # 背景图层
        if self.bg_photo:
            self.bg_label = tk.Label(self.master, image=self.bg_photo)
        else:
            self.bg_label = tk.Label(self.master, bg="#f0f2f5")
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # 半透明覆盖层（模拟10%不透明度）
        self.overlay = tk.Canvas(
            self.master,
            bg='#ffffff',  # 白色
            bd=0,
            highlightthickness=0
        )
        self.overlay.place(relx=0.5, rely=0.5, anchor="center", width=380, height=480)

        # 设置覆盖层透明度（通过调整alpha值）
        self.master.attributes('-alpha', 0.9)  # 整体窗口透明度

        # 主容器（完全透明）
        main_frame = tk.Frame(
            self.master,
            bg='',
            bd=0
        )
        main_frame.place(relx=0.5, rely=0.5, anchor="center", width=380, height=480)

        # 标题
        title1 = tk.Label(
            main_frame,
            text="乒乓球初学者动作智能评估系统",
            font=("Microsoft YaHei", 18),
            bg="#ffffff",
            fg="#2c3e50"
        )
        title1.pack(pady=(10, 10))
        title = tk.Label(
            main_frame,
            text="欢迎登录",
            font=("Microsoft YaHei", 24, "bold"),
            bg="#ffffff",
            fg="#2c3e50"
        )
        title.pack(pady=(20, 10))

        # 输入框容器
        input_frame = tk.Frame(main_frame, bg="#ffffff")
        input_frame.pack(padx=30, pady=10, fill=tk.X)

        # 用户名输入
        tk.Label(
            input_frame,
            text="用户名",
            bg="#ffffff",
            fg="#7f8c8d",
            font=("Microsoft YaHei", 11)
        ).pack(anchor="w", pady=(5, 0))

        self.username_entry = tk.Entry(
            input_frame,
            font=("Microsoft YaHei", 12),
            bd=0,
            bg="#f8f9fa",  # 浅灰色背景
            highlightthickness=1,
            highlightbackground="#dfe6e9",
            highlightcolor="#3498db",
            relief=tk.FLAT
        )
        self.username_entry.pack(fill=tk.X, pady=5, ipady=8)

        # 密码输入
        tk.Label(
            input_frame,
            text="密码",
            bg="#ffffff",
            fg="#7f8c8d",
            font=("Microsoft YaHei", 11)
        ).pack(anchor="w", pady=(15, 0))

        self.password_entry = tk.Entry(
            input_frame,
            show="•",
            font=("Microsoft YaHei", 12),
            bd=0,
            bg="#f8f9fa",  # 浅灰色背景
            highlightthickness=1,
            highlightbackground="#dfe6e9",
            highlightcolor="#3498db",
            relief=tk.FLAT
        )
        self.password_entry.pack(fill=tk.X, pady=5, ipady=8)

        # 登录按钮
        login_btn = tk.Button(
            main_frame,
            text="登 录",
            command=self.login,
            bg="#3498db",
            fg="white",
            bd=0,
            padx=30,
            pady=12,
            font=("Microsoft YaHei", 12, "bold"),
            activebackground="#2980b9",
            relief=tk.FLAT,
            cursor="hand2"
        )
        login_btn.pack(fill=tk.X, padx=30, pady=(30, 15))

        # 注册链接
        register_link = tk.Label(
            main_frame,
            text="没有账号？立即注册",
            fg="#3498db",
            bg="#ffffff",
            font=("Microsoft YaHei", 10),
            cursor="hand2"
        )
        register_link.pack(pady=(0, 30))
        register_link.bind("<Button-1>", lambda e: self.register())

        # 绑定回车键
        self.master.bind("<Return>", lambda e: self.login())

    def login(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()

        if not username or not password:
            messagebox.showwarning("提示", "请输入用户名和密码")
            return

        user_id = self.db.verify_user(username, password)
        if user_id:
            self.master.destroy()
            from main_window import MainWindow
            root = tk.Tk()
            MainWindow(root, user_id)
            root.mainloop()
        else:
            messagebox.showerror("错误", "用户名或密码错误")

    def register(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()

        if not username or not password:
            messagebox.showwarning("提示", "请输入用户名和密码")
            return

        if len(password) < 6:
            messagebox.showwarning("提示", "密码长度至少6位")
            return

        if len(username) < 4:
            messagebox.showwarning("提示", "用户名长度至少4位")
            return

        if self.db.add_user(username, password):
            messagebox.showinfo("成功", "注册成功，请登录")
        else:
            messagebox.showerror("错误", "用户名已存在")