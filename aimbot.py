import cv2
import numpy as np
import time
from ultralytics import YOLO
from mss import mss
import ctypes
import win32api
import win32con
import threading

# ============== 配置区域（根据你的情况修改） ==============
MODEL_PATH = ' runs/detect/csgo_v11_optimized/train_20250708_191323/weights/best.pt'  # 训练好的模型路径
SCREEN_REGION = {'left': 640, 'top': 360, 'width': 1000, 'height': 800}  # 屏幕中心监控区域，宽度已增加
FIRE_THRESHOLD = 0.7  # 开火置信度阈值
HEAD_CLASS = 1  # 头部类别ID
BODY_CLASS = 0  # 身体类别ID

# 新增：后坐力补偿参数 (需要根据游戏和武器进行微调)
RECOIL_COMPENSATION_X = 5  # 水平方向补偿像素
RECOIL_COMPENSATION_Y = 15  # 垂直方向补偿像素


# ============== Windows API 键盘监控 ==============
class KeyboardMonitor:
    def __init__(self):
        self.exit_flag = False
        self.monitor_thread = threading.Thread(target=self.monitor_keyboard, daemon=True)
        self.monitor_thread.start()

    def monitor_keyboard(self):
        """独立线程监控键盘事件"""
        # 使用Windows API直接监控键盘状态
        while not self.exit_flag:
            # 检查Q键是否被按下
            if ctypes.windll.user32.GetAsyncKeyState(ord('Q')) & 0x8000:
                self.exit_flag = True
                print("检测到Q键按下，程序即将退出...")
                break
            time.sleep(0.01)  # 降低CPU占用率

    def should_exit(self):
        """检查是否应该退出程序"""
        return self.exit_flag


# ============== Windows API 鼠标控制 ==============
class MouseController:
    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.screen_width = self.user32.GetSystemMetrics(0)
        self.screen_height = self.user32.GetSystemMetrics(1)

    def move_to(self, x, y):
        """将鼠标移动到绝对坐标位置 (0,0) 到 (65535,65535)"""
        x = int(x * 65535 / self.screen_width)
        y = int(y * 65535 / self.screen_height)
        self.user32.mouse_event(win32con.MOUSEEVENTF_ABSOLUTE | win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)

    def move_relative(self, dx, dy):
        """相对移动鼠标"""
        self.user32.mouse_event(win32con.MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

    def left_down(self):
        """按下鼠标左键"""
        self.user32.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    def left_up(self):
        """释放鼠标左键"""
        self.user32.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def smooth_move(self, start_x, start_y, end_x, end_y, steps=10):
        """平滑移动鼠标（模拟人类移动），无延迟"""
        dx = (end_x - start_x) / steps
        dy = (end_y - start_y) / steps

        current_x, current_y = start_x, start_y
        for _ in range(steps):
            current_x += dx
            current_y += dy
            self.move_to(current_x, current_y)
            # 移除了步进之间的延迟


# ====================================================

class CSGOAimbot:
    def __init__(self):
        # 加载训练好的模型
        self.model = YOLO(MODEL_PATH)
        # 屏幕捕获工具
        self.sct = mss()
        # 计算屏幕中心点 (相对于捕获区域)
        self.center_x = SCREEN_REGION['width'] // 2
        self.center_y = SCREEN_REGION['height'] // 2
        # 鼠标控制器
        self.mouse = MouseController()
        # 键盘监控器
        self.keyboard_monitor = KeyboardMonitor()

        print(f"模型加载完成！监控区域: {SCREEN_REGION}")

    def capture_screen(self):
        """捕获屏幕指定区域的图像."""
        screen = np.array(self.sct.grab(SCREEN_REGION))
        # 将 BGRA 格式转换为 BGR 格式
        return cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

    def calculate_distance(self, x, y):
        """计算目标中心到屏幕中心的距离"""
        return ((x - self.center_x) ** 2 + (y - self.center_y) ** 2) ** 0.5

    def find_targets(self, frame):
        """识别帧中的目标并优先排序."""
        # 执行 YOLO 模型推理
        results = self.model(frame, verbose=False)[0]
        head_targets = []  # 存储头部目标
        body_targets = []  # 存储身体目标

        # 如果没有检测到任何目标，返回空列表
        if results.boxes is None:
            return [], [], results

        for box in results.boxes:
            conf = float(box.conf)  # 获取置信度
            cls = int(box.cls)  # 获取类别ID
            # 获取边界框坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 只处理置信度达到阈值的目标
            if conf > FIRE_THRESHOLD:
                # 计算边界框的中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                # 计算到屏幕中心的距离
                distance = self.calculate_distance(center_x, center_y)

                # 根据目标类型存储
                if cls == HEAD_CLASS:
                    head_targets.append((center_x, center_y, conf, distance, (x1, y1, x2, y2)))
                elif cls == BODY_CLASS:
                    body_targets.append((center_x, center_y, conf, distance, (x1, y1, x2, y2)))

        # 对目标进行排序：优先近距离目标
        head_targets.sort(key=lambda t: t[3])  # 按距离排序
        body_targets.sort(key=lambda t: t[3])  # 按距离排序

        # 返回所有符合条件的敌人目标和完整的检测结果
        return head_targets, body_targets, results

    def move_and_fire(self, x, y):
        """移动鼠标到目标中心并开火，然后执行后坐力补偿."""
        # 将局部捕获区域坐标转换为全局屏幕坐标
        global_x = SCREEN_REGION['left'] + x
        global_y = SCREEN_REGION['top'] + y

        # 获取当前鼠标位置（用于平滑移动）
        current_x, current_y = win32api.GetCursorPos()

        # 平滑移动鼠标到目标位置（无延迟版本）
        self.mouse.smooth_move(current_x, current_y, global_x, global_y)

        # 开火（无延迟点击）
        self.mouse.left_down()
        self.mouse.left_up()  # 立即释放，不等待
        print(f"开火! 目标全局坐标: ({global_x}, {global_y})")

        # 执行后坐力补偿（无延迟）
        self.mouse.move_relative(RECOIL_COMPENSATION_X, RECOIL_COMPENSATION_Y)
        print(f"执行后坐力补偿: dx={RECOIL_COMPENSATION_X}, dy={RECOIL_COMPENSATION_Y}")

    def run(self):
        """主运行循环，持续捕获屏幕、检测目标、瞄准和开火."""
        print("自动瞄准系统启动! 按 'Q' 键退出")
        print("=== 注意: 游戏窗口必须处于前台状态 ===")

        while True:
            # 1. 检查键盘退出信号（使用Windows API实时监控）
            if self.keyboard_monitor.should_exit():
                print("收到退出信号，程序终止...")
                break

            # 2. 捕获屏幕指定区域的图像
            frame = self.capture_screen()

            # 3. 检测目标并获取完整的检测结果
            head_targets, body_targets, detection_results = self.find_targets(frame)

            # 创建一个帧副本用于调试可视化，避免在原始帧上绘制
            debug_frame = frame.copy()

            # 绘制所有检测到的边界框（包括头部和身体）
            if detection_results.boxes is not None:
                for box in detection_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    cls = int(box.cls)

                    label = f"{self.model.names[cls]} {conf:.2f}"
                    color = (0, 255, 0)  # 默认绿色表示身体
                    if cls == HEAD_CLASS:
                        color = (0, 0, 255)  # 红色表示头部

                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(debug_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 4. 优先选择目标：先头部后身体，先近距离后远距离
            target = None
            target_type = "未知"  # 初始化默认值

            # 优先选择近距离的头部目标
            if head_targets:
                target = head_targets[0]
                target_type = "头部"
            # 如果没有头部目标，选择近距离的身体目标
            elif body_targets:
                target = body_targets[0]
                target_type = "身体"

            # 如果有目标，则瞄准并开火
            if target:
                x, y, conf, distance, bbox = target
                print(f"发现{target_type}目标! 距离: {distance:.1f}px, 置信度: {conf:.2f} 局部坐标: ({x}, {y})")
                # 移动鼠标并开火，然后进行后坐力补偿
                self.move_and_fire(x, y)

            # 5. 显示调试画面
            # 在调试帧上绘制监控区域的中心点
            cv2.circle(debug_frame, (self.center_x, self.center_y),
                       10, (0, 255, 0), -1)  # 绿色实心圆
            cv2.imshow('CSGO 自动瞄准 - 按Q退出', debug_frame)

            # 6. 检查OpenCV窗口退出信号
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("通过OpenCV窗口检测到退出信号...")
                break

        # 退出循环后，关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        print("程序已安全退出")


if __name__ == "__main__":
    print("=== CSGO 自动瞄准系统 ===")
    print("正在初始化...")
    aimbot = CSGOAimbot()
    try:
        aimbot.run()
    except Exception as e:
        print(f"程序运行时发生错误: {e}")
        cv2.destroyAllWindows()