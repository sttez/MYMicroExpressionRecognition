# 测试脚本 test_env.py
import tensorflow as tf
print("TF Version:", tf.__version__)
print("GPU Available:", tf.test.is_gpu_available())

import keras
print("Keras Version:", keras.__version__)

import cv2
print("OpenCV Version:", cv2.__version__)

# 应输出：
# TF Version: 1.13.1
# GPU Available: True
# Keras Version: 2.1.5
# OpenCV Version: 4.1.2