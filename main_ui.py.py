import sys
import os
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
                             QFileDialog, QLabel, QMainWindow, QSplitter, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap

from classification import Classification

# 实例化分类器
classfication = Classification()
face_cascade = cv2.CascadeClassifier(r'model_data/haarcascade_frontalface_alt.xml')


def face_detect(img):
    """
    检测图片中的人脸
    :param img: 输入图像
    :return: 原图、灰度图和检测到的人脸位置
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(30, 30)
    )
    return img, img_gray, faces


def process_image(img):
    """
    处理图像，检测人脸并识别表情
    :param img: 输入图像
    :return: 处理后的图像
    """
    border_color = (255, 0, 0)  # 框框颜色
    font_color = (0, 0, 255)  # 文字颜色

    img, img_gray, faces = face_detect(img)

    if len(faces) == 0:
        cv2.putText(img, '未检测到人脸', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    for (x, y, w, h) in faces:
        # 确保不会越界
        y_start = max(0, y - 10)
        y_end = min(img.shape[0], y + h + 10)
        x_start = max(0, x - 10)
        x_end = min(img.shape[1], x + w + 10)

        face_img_gray = img_gray[y_start:y_end, x_start:x_end]

        # 转换格式
        face_img = Image.fromarray(face_img_gray)

        # 预测结果
        class_name = classfication.detect_image(face_img)
        emotion = class_name[0]
        probability = class_name[1]

        # 画出人脸矩形
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), border_color, thickness=2)
        # 显示情绪类别
        cv2.putText(img, emotion, (x + 30, max(30, y - 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
        # 显示置信度
        cv2.putText(img, f"{probability:.2f}", (x + 30, max(60, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, font_color, 1)

    return img


# 摄像头线程
class CameraThread(QThread):
    update_frame = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error_signal.emit("无法打开摄像头！")
            return

        self.running = True

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.error_signal.emit("无法读取摄像头画面！")
                break

            # 水平镜像翻转
            frame = cv2.flip(frame, 1)

            # 处理图像
            processed_frame = process_image(frame)

            # 发送信号
            self.update_frame.emit(processed_frame)

            # 添加短暂延迟，降低CPU占用
            self.msleep(30)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# 视频处理线程
class VideoThread(QThread):
    update_frame = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    error_signal = pyqtSignal(str)

    def __init__(self, video_path, output_path):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error_signal.emit(f"无法打开视频：{self.video_path}")
            return

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 设置输出视频
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        self.running = True
        frame_idx = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # 处理图像
            processed_frame = process_image(frame)

            # 写入输出视频
            out.write(processed_frame)

            # 更新界面显示
            self.update_frame.emit(processed_frame)

            # 更新进度
            frame_idx += 1
            progress_percent = int((frame_idx / frame_count) * 100)
            self.progress.emit(progress_percent)

            # 添加短暂延迟，避免UI卡顿
            self.msleep(30)

        cap.release()
        out.release()
        self.finished.emit()

    def stop(self):
        self.running = False
        self.wait()


class MicroExpressionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("微表情识别系统")
        self.setGeometry(100, 100, 1000, 600)

        # 创建中央窗口
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 创建主布局
        self.main_layout = QVBoxLayout(self.central_widget)

        # 创建按钮布局
        self.button_layout = QHBoxLayout()

        # 创建按钮
        self.camera_btn = QPushButton("摄像头实时识别")
        self.video_btn = QPushButton("视频识别")
        self.image_btn = QPushButton("图片识别")

        # 将按钮添加到布局
        self.button_layout.addWidget(self.camera_btn)
        self.button_layout.addWidget(self.video_btn)
        self.button_layout.addWidget(self.image_btn)

        # 将按钮布局添加到主布局
        self.main_layout.addLayout(self.button_layout)

        # 创建状态标签
        self.status_label = QLabel("请选择识别方式")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)

        # 创建显示区域布局
        self.display_layout = QHBoxLayout()

        # 创建左侧显示标签
        self.left_display = QLabel("原始图像")
        self.left_display.setAlignment(Qt.AlignCenter)
        self.left_display.setMinimumSize(400, 300)
        self.left_display.setStyleSheet("border: 1px solid #cccccc;")

        # 创建右侧显示标签
        self.right_display = QLabel("识别结果")
        self.right_display.setAlignment(Qt.AlignCenter)
        self.right_display.setMinimumSize(400, 300)
        self.right_display.setStyleSheet("border: 1px solid #cccccc;")

        # 创建分隔器
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.left_display)
        self.splitter.addWidget(self.right_display)
        self.splitter.setSizes([500, 500])  # 设置初始大小均等

        # 添加分隔器到显示布局
        self.display_layout.addWidget(self.splitter)

        # 将显示布局添加到主布局
        self.main_layout.addLayout(self.display_layout)

        # 设置按钮连接
        self.camera_btn.clicked.connect(self.start_camera)
        self.video_btn.clicked.connect(self.process_video)
        self.image_btn.clicked.connect(self.process_image)

        # 初始化摄像头线程和定时器
        self.camera_thread = None
        self.video_thread = None

        # 确保输出目录存在
        if not os.path.exists("output"):
            os.makedirs("output")

    def convert_cv_qt(self, cv_img):
        """
        将OpenCV图像转换为QPixmap
        """
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(qt_image)
        except Exception as e:
            print(f"转换图像错误: {e}")
            return None

    def update_camera_frame(self, frame):
        """
        更新摄像头帧到右侧显示区域
        """
        pixmap = self.convert_cv_qt(frame)
        if pixmap:
            # 在摄像头模式下，只显示在右侧
            self.right_display.setPixmap(pixmap.scaled(self.right_display.size(), Qt.KeepAspectRatio))
            # 左侧显示信息
            self.left_display.setText("摄像头实时识别\n\n请面对摄像头")

    def update_video_frame(self, frame):
        """
        更新视频帧到右侧显示区域
        """
        pixmap = self.convert_cv_qt(frame)
        if pixmap:
            self.right_display.setPixmap(pixmap.scaled(self.right_display.size(), Qt.KeepAspectRatio))

    def handle_error(self, message):
        """
        处理错误信息
        """
        self.status_label.setText(f"错误: {message}")
        self.progress_bar.setVisible(False)

    def start_camera(self):
        """
        开始摄像头识别
        """
        # 停止所有正在运行的线程
        self.stop_all_threads()

        self.status_label.setText("正在启动摄像头...")
        self.left_display.setText("正在启动摄像头...")
        self.right_display.setText("等待画面...")
        self.progress_bar.setVisible(False)

        # 创建并启动摄像头线程
        self.camera_thread = CameraThread()
        self.camera_thread.update_frame.connect(self.update_camera_frame)
        self.camera_thread.error_signal.connect(self.handle_error)
        self.camera_thread.start()

        self.status_label.setText("摄像头实时识别中...")

    def process_video(self):
        """
        处理视频
        """
        # 停止所有正在运行的线程
        self.stop_all_threads()

        video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)")
        if not video_path:
            return

        # 设置输出路径
        output_path = os.path.join("output", "processed_" + os.path.basename(video_path))

        self.status_label.setText(f"正在处理视频：{os.path.basename(video_path)}")
        self.left_display.setText("正在加载视频...")
        self.right_display.setText("等待处理结果...")

        # 显示原始视频第一帧
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            pixmap = self.convert_cv_qt(frame)
            if pixmap:
                self.left_display.setPixmap(pixmap.scaled(self.left_display.size(), Qt.KeepAspectRatio))
        else:
            self.left_display.setText("无法读取视频")
            return

        # 显示进度条
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # 创建并启动视频处理线程
        self.video_thread = VideoThread(video_path, output_path)
        self.video_thread.update_frame.connect(self.update_video_frame)
        self.video_thread.progress.connect(self.update_progress)
        self.video_thread.finished.connect(self.video_processing_finished)
        self.video_thread.error_signal.connect(self.handle_error)
        self.video_thread.start()

    def update_progress(self, value):
        """
        更新进度
        """
        self.progress_bar.setValue(value)
        self.status_label.setText(f"视频处理进度：{value}%")

    def video_processing_finished(self):
        """
        视频处理完成
        """
        self.status_label.setText("视频处理完成！结果已保存到output文件夹。")
        self.progress_bar.setVisible(False)

    def process_image(self):
        """
        处理图片
        """
        # 停止所有正在运行的线程
        self.stop_all_threads()

        img_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.png *.jpeg)")
        if not img_path:
            return

        self.status_label.setText(f"正在处理图片：{os.path.basename(img_path)}")
        self.progress_bar.setVisible(False)

        try:
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                self.status_label.setText("无法读取图片！")
                return

            # 显示原始图片
            pixmap = self.convert_cv_qt(img)
            if pixmap:
                self.left_display.setPixmap(pixmap.scaled(self.left_display.size(), Qt.KeepAspectRatio))

            # 处理图片并显示
            processed_img = process_image(img)
            processed_pixmap = self.convert_cv_qt(processed_img)
            if processed_pixmap:
                self.right_display.setPixmap(processed_pixmap.scaled(self.right_display.size(), Qt.KeepAspectRatio))

            # 保存处理后的图片
            output_path = os.path.join("output", "processed_" + os.path.basename(img_path))
            cv2.imwrite(output_path, processed_img)

            self.status_label.setText(f"图片处理完成，已保存至：{output_path}")

        except Exception as e:
            self.status_label.setText(f"处理图片时出错：{str(e)}")

    def stop_all_threads(self):
        """
        停止所有正在运行的线程
        """
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread = None

        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None

    def closeEvent(self, event):
        """
        窗口关闭事件
        """
        self.stop_all_threads()
        event.accept()

    def resizeEvent(self, event):
        """
        窗口大小改变事件，用于更新显示的图像
        """
        super().resizeEvent(event)
        # 如果左右显示区域有pixmap，重新缩放显示
        if not self.left_display.pixmap() is None:
            self.left_display.setPixmap(self.left_display.pixmap().scaled(
                self.left_display.size(), Qt.KeepAspectRatio))
        if not self.right_display.pixmap() is None:
            self.right_display.setPixmap(self.right_display.pixmap().scaled(
                self.right_display.size(), Qt.KeepAspectRatio))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MicroExpressionUI()
    window.show()
    sys.exit(app.exec_())