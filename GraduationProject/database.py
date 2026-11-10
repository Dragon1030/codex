# database.py
import sqlite3
from datetime import datetime
import os
import json


class PingPongDB:
    def __init__(self):
        self.db_path = "pingpong_analysis.db"
        self.init_db()

    def init_db(self):
        with self.get_connection() as conn:
            # 用户表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 分析记录表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    video_path TEXT NOT NULL,
                    stroke_type TEXT NOT NULL,
                    score REAL NOT NULL,
                    elbow_score REAL NOT NULL,
                    knee_score REAL NOT NULL,
                    waist_score REAL NOT NULL,
                    shoulder_score REAL NOT NULL,
                    weight_transfer_score REAL NOT NULL,
                    analysis_data TEXT NOT NULL,
                    analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)

            # 用户登录记录表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS login_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)

            # 用户成长记录表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS progress_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    record_date DATE NOT NULL,
                    forehand_count INTEGER DEFAULT 0,
                    backhand_count INTEGER DEFAULT 0,
                    serve_count INTEGER DEFAULT 0,
                    avg_score REAL DEFAULT 0,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)

            # 测试数据
            if not self.user_exists("admin"):
                self.add_user("admin", "123")

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def user_exists(self, username):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM users WHERE username=?", (username,))
            return cursor.fetchone() is not None

    def add_user(self, username, password):
        with self.get_connection() as conn:
            try:
                conn.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)",
                    (username, password)
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def verify_user(self, username, password):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM users WHERE username=? AND password=?",
                (username, password)
            )
            result = cursor.fetchone()
            if result:
                user_id = result[0]
                self.record_login(user_id)
                return user_id
            return None

    def record_login(self, user_id):
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO login_records (user_id) VALUES (?)",
                (user_id,)
            )
            conn.commit()

    def add_analysis_record(self, user_id, video_path, stroke_type, score,
                           elbow_score, knee_score, waist_score,
                           shoulder_score, weight_transfer_score, analysis_data):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO analysis_records 
                   (user_id, video_path, stroke_type, score, 
                    elbow_score, knee_score, waist_score,
                    shoulder_score, weight_transfer_score, analysis_data)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, os.path.basename(video_path), stroke_type, score,
                 elbow_score, knee_score, waist_score,
                 shoulder_score, weight_transfer_score, analysis_data)
            )
            record_id = cursor.lastrowid
            conn.commit()

            # 更新成长记录
            self.update_progress_record(user_id, stroke_type, score)

            return record_id

    def update_progress_record(self, user_id, stroke_type, score):
        today = datetime.now().strftime('%Y-%m-%d')
        with self.get_connection() as conn:
            # 检查今天是否已有记录
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM progress_records WHERE user_id=? AND record_date=?",
                (user_id, today)
            )
            record = cursor.fetchone()

            if record:
                # 更新现有记录
                record_id = record[0]
                if stroke_type == "正手攻球":
                    conn.execute(
                        "UPDATE progress_records SET forehand_count = forehand_count + 1, avg_score = (avg_score + ?)/2 WHERE user_id=? AND record_date=?",
                        (score, user_id, today)
                    )
                elif stroke_type == "反手推挡":
                    conn.execute(
                        "UPDATE progress_records SET backhand_count = backhand_count + 1, avg_score = (avg_score + ?)/2 WHERE user_id=? AND record_date=?",
                        (score, user_id, today)
                    )
                else:  # 发球
                    conn.execute(
                        "UPDATE progress_records SET serve_count = serve_count + 1, avg_score = (avg_score + ?)/2 WHERE user_id=? AND record_date=?",
                        (score, user_id, today)
                    )
            else:
                # 创建新记录
                forehand = 1 if stroke_type == "正手攻球" else 0
                backhand = 1 if stroke_type == "反手推挡" else 0
                serve = 1 if stroke_type == "发球" else 0

                conn.execute(
                    """INSERT INTO progress_records 
                       (user_id, record_date, forehand_count, backhand_count, serve_count, avg_score)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (user_id, today, forehand, backhand, serve, score)
                )
            conn.commit()

    def get_user_records(self, user_id, limit=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT id, video_path, stroke_type, score, elbow_score, knee_score, waist_score,
                       shoulder_score, weight_transfer_score,
                       strftime('%Y-%m-%d %H:%M', analysis_time) 
                FROM analysis_records 
                WHERE user_id=? 
                ORDER BY analysis_time DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            cursor.execute(query, (user_id,))
            return cursor.fetchall()

    def get_progress_records(self, user_id, days=30):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT record_date, forehand_count, backhand_count, serve_count, avg_score
                   FROM progress_records 
                   WHERE user_id=?
                   ORDER BY record_date DESC
                   LIMIT ?""",
                (user_id, days)
            )
            return cursor.fetchall()

    def get_stroke_stats(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT stroke_type, COUNT(*) as count, AVG(score) as avg_score,
                          AVG(elbow_score) as avg_elbow, AVG(knee_score) as avg_knee,
                          AVG(waist_score) as avg_waist
                   FROM analysis_records 
                   WHERE user_id=?
                   GROUP BY stroke_type""",
                (user_id,)
            )
            return cursor.fetchall()

    def search_records(self, user_id, search_term):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT id, video_path, stroke_type, score, elbow_score, knee_score, waist_score,
                       strftime('%Y-%m-%d %H:%M', analysis_time) 
                   FROM analysis_records 
                   WHERE user_id=? AND (video_path LIKE ? OR stroke_type LIKE ?)
                   ORDER BY analysis_time DESC""",
                (user_id, f"%{search_term}%", f"%{search_term}%")
            )
            return cursor.fetchall()

    def clear_user_records(self, user_id):
        """清空指定用户的所有记录"""
        with self.get_connection() as conn:
            try:
                conn.execute("DELETE FROM analysis_records WHERE user_id=?", (user_id,))
                conn.execute("DELETE FROM progress_records WHERE user_id=?", (user_id,))
                conn.commit()
                return True
            except sqlite3.Error as e:
                print(f"Error clearing records: {e}")
                return False

    def get_login_records(self, user_id, days=30):
        """获取用户登录记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT strftime('%Y-%m-%d', login_time), COUNT(*) 
                   FROM login_records 
                   WHERE user_id=? 
                   AND login_time >= date('now', ?)
                   GROUP BY strftime('%Y-%m-%d', login_time)
                   ORDER BY login_time DESC""",
                (user_id, f"-{days} days")
            )
            return cursor.fetchall()