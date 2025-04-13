import os
import shutil
import pandas as pd
import random

# 定义常量
CASME2_RAW_PATH = 'CASME2-RAW'  # CASME2-RAW文件夹路径
EXCEL_FILE1 = 'CASME2-coding-20140508.xlsx'  # 第一个Excel文件
EXCEL_FILE2 = 'CASME2-ObjectiveClasses.xlsx'  # 第二个Excel文件
OUTPUT_PATH = 'datasets'  # 输出文件夹路径
TRAIN_RATIO = 0.7  # 训练集比例
TEST_RATIO = 0.3  # 验证集比例


# 创建保存图像的文件夹
def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


# 读取Excel文件
def read_excel_files():
    # 读取文件1
    file1 = pd.read_excel(EXCEL_FILE1, header=None, names=[
        'Subject', 'Filename', 'Unnamed: 2', 'OnsetFrame', 'ApexFrame',
        'OffsetFrame', 'Unnamed: 6', 'Action Units', 'Estimated Emotion'
    ])

    # 读取文件2
    file2 = pd.read_excel(EXCEL_FILE2, header=None, names=[
        'Subject', 'Filename', 'Objective Class'
    ])

    # 合并两个文件
    merged_data = pd.merge(file1, file2, on=['Subject', 'Filename'])

    # 建立情感标签与客观类别的映射
    emotion_to_class = {}
    for _, row in merged_data.iterrows():
        emotion = row['Estimated Emotion']
        obj_class = row['Objective Class']
        if emotion not in emotion_to_class:
            emotion_to_class[emotion] = obj_class

    return merged_data, emotion_to_class


# 创建输出文件夹结构
def create_output_structure(emotions):
    for split in ['train', 'test']:
        split_path = os.path.join(OUTPUT_PATH, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        for emotion in emotions:
            emotion_path = os.path.join(split_path, emotion)
            if not os.path.exists(emotion_path):
                os.makedirs(emotion_path)


# 分类图片和视频
def classify_data(merged_data, emotion_to_class):
    # 提取所有情感标签
    emotions = merged_data['Estimated Emotion'].unique()

    # 创建输出文件夹结构
    create_output_structure(emotions)

    # 按情感标签分组
    grouped = merged_data.groupby('Estimated Emotion')

    # 为每个情感标签提取图片和视频并划分训练集和验证集
    for emotion, group in grouped:
        image_paths = []
        video_paths = []
        for _, row in group.iterrows():
            subject = row['Subject']
            filename = row['Filename']

            # 构建视频文件夹路径
            subject_path = os.path.join(CASME2_RAW_PATH, f'sub{subject}')
            video_folder = os.path.join(subject_path, filename)

            # 检查视频文件夹是否存在
            if not os.path.exists(video_folder):
                continue

            # 获取图片列表
            images = [f for f in os.listdir(video_folder) if f.startswith('img') and f.endswith('.jpg')]
            images.sort(key=lambda x: int(x[3:-4]))  # 按数字排序

            # 获取OnsetFrame和OffsetFrame
            onset_frame = row['OnsetFrame']
            offset_frame = row['OffsetFrame']

            # 提取OnsetFrame到OffsetFrame之间的图片
            valid_images = [img for img in images if onset_frame <= int(img[3:-4]) <= offset_frame]

            # 添加到列表
            for img in valid_images:
                image_paths.append(os.path.join(video_folder, img))

            # 添加视频路径
            video_path = os.path.join(video_folder, f"{filename}.avi")
            if os.path.exists(video_path):
                video_paths.append(video_path)

        # 按7:3比例划分训练集和验证集
        if len(image_paths) > 0:
            random.shuffle(image_paths)
            split_point = int(len(image_paths) * TRAIN_RATIO)

            train_paths = image_paths[:split_point]
            test_paths = image_paths[split_point:]

            # 复制图片到输出文件夹
            for img_path in train_paths:
                dst_path = os.path.join(OUTPUT_PATH, 'train', emotion, os.path.basename(img_path))
                makedir(os.path.dirname(dst_path))
                shutil.copy(img_path, dst_path)
                print(f"Copied to train: {dst_path}")

            for img_path in test_paths:
                dst_path = os.path.join(OUTPUT_PATH, 'test', emotion, os.path.basename(img_path))
                makedir(os.path.dirname(dst_path))
                shutil.copy(img_path, dst_path)
                print(f"Copied to test: {dst_path}")

        # 复制视频到输出文件夹
        if len(video_paths) > 0:
            random.shuffle(video_paths)
            split_point = int(len(video_paths) * TRAIN_RATIO)

            train_videos = video_paths[:split_point]
            test_videos = video_paths[split_point:]

            for video_path in train_videos:
                dst_path = os.path.join(OUTPUT_PATH, 'train', emotion, os.path.basename(video_path))
                makedir(os.path.dirname(dst_path))
                shutil.copy(video_path, dst_path)
                print(f"Copied to train: {dst_path}")

            for video_path in test_videos:
                dst_path = os.path.join(OUTPUT_PATH, 'test', emotion, os.path.basename(video_path))
                makedir(os.path.dirname(dst_path))
                shutil.copy(video_path, dst_path)
                print(f"Copied to test: {dst_path}")


# 主函数
def main():
    merged_data, emotion_to_class = read_excel_files()
    classify_data(merged_data, emotion_to_class)
    print("Classification completed!")


if __name__ == "__main__":
    main()