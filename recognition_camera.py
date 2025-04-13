"""
微表情识别摄像头检测
视频检测
"""

import os
import cv2
import numpy as np
from PIL import Image

from classification import Classification

# 实例化分类器
classification = Classification()

# 加载 OpenCV 人脸检测模型
face_cascade = cv2.CascadeClassifier(r'model_data/haarcascade_frontalface_alt.xml')


def face_detect(img_gray):
    """
    检测图像中的人脸
    :param img_gray: 灰度图像
    :return:
    """
    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(30, 30)
    )
    return faces


def generate_faces(face_img, img_size=48):
    """
    将探测到的人脸进行增广（本代码未直接用到）
    """
    face_img = face_img / 255.
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img[:, :])
    resized_images.append(face_img[2:45, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))
    resized_images.append(face_img[0:45, 0:45])
    resized_images.append(face_img[2:47, 0:45])
    resized_images.append(face_img[2:47, 2:47])

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images


def predict_expression():
    border_color = (255, 0, 0)    # 人脸框颜色
    font_color = (0, 0, 255)      # 情绪文本颜色

    # 开启摄像头
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # 水平镜像
        frame = cv2.flip(frame, 1)

        # 转为灰度图
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = face_detect(img_gray)

        if len(faces) == 0:
            cv2.putText(frame, 'No Face Detect.', (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        for (x, y, w, h) in faces:
            # 获取灰度人脸区域
            face_img_gray = img_gray[y:y + h + 10, x:x + w + 10]

            # 转换为 PIL 格式传入分类器
            face_img_gray_pil = Image.fromarray(face_img_gray)

            # 预测情绪
            class_name = classification.detect_image(face_img_gray_pil)
            emotion = class_name[0]
            probability = class_name[1]

            # 绘制矩形框
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), border_color, thickness=2)
            # 绘制情绪标签
            cv2.putText(frame, emotion, (x + 30, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 1)
            # 显示概率
            cv2.putText(frame, str(probability), (x + 30, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, font_color, 1)

        # 显示图像
        cv2.imshow("Cam", frame)

        key = cv2.waitKey(30)
        if key == 27:  # Esc 键退出
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict_expression()
