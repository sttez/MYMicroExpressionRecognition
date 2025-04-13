import os
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from nets import freeze_layers, get_model_from_name
from utils.callbacks import ExponentDecayScheduler, LossHistory
from utils.dataloader import ClsDatasets
from utils.utils import get_classes
from nets.Loss import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    classes_path = 'model_data/cls_classes.txt'
    input_shape = [224, 224]
    backbone = "vgg16"
    alpha = 0.25
    model_path = "model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    Freeze_Train = True
    annotation_path = "cls_train.txt"
    val_split = 0.1
    num_workers = 1

    class_names, num_classes = get_classes(classes_path)
    assert backbone in ["mobilenet", "resnet50", "vgg16"]

    # 构建模型
    if backbone == "mobilenet":
        model = get_model_from_name[backbone](input_shape=[input_shape[0], input_shape[1], 3],
                                              classes=num_classes, alpha=alpha)
    else:
        model = get_model_from_name[backbone](input_shape=[input_shape[0], input_shape[1], 3],
                                              classes=num_classes)

    if model_path != "":
        print('✅ 加载预训练权重:', model_path)
        model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # Callbacks
    logging = TensorBoard(log_dir='logs/')
    checkpoint = ModelCheckpoint(
        'logs/best_model.h5', monitor='val_loss',
        save_weights_only=False, save_best_only=True, period=1
    )
    reduce_lr = ExponentDecayScheduler(decay_rate=0.94, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1)
    loss_history = LossHistory('logs/')

    # 数据划分
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # 冻结主干
    if Freeze_Train:
        for i in range(freeze_layers[backbone]):
            model.layers[i].trainable = False

    # 冻结训练阶段
    if True:
        batch_size = 16
        Lr = 1e-3
        Init_Epoch = 0
        Freeze_Epoch = 50
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("❌ 数据集过小，请扩充数据集。")

        print(f"🚀 冻结训练: {num_train} 训练样本, {num_val} 验证样本, batch size = {batch_size}")

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=Lr),
                      metrics=['categorical_accuracy'])

        train_dataloader = ClsDatasets(lines[:num_train], input_shape, batch_size, num_classes, train=True)
        val_dataloader = ClsDatasets(lines[num_train:], input_shape, batch_size, num_classes, train=False)

        model.fit_generator(
            generator=train_dataloader,
            steps_per_epoch=epoch_step,
            validation_data=val_dataloader,
            validation_steps=epoch_step_val,
            epochs=Freeze_Epoch,
            initial_epoch=Init_Epoch,
            use_multiprocessing=num_workers > 1,
            workers=num_workers,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )

    # 解冻主干
    if Freeze_Train:
        for i in range(freeze_layers[backbone]):
            model.layers[i].trainable = True

    # 解冻训练阶段
    if True:
        batch_size = 16
        Lr = 1e-4
        Freeze_Epoch = 50
        Epoch = 100
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("❌ 数据集过小，请扩充数据集。")

        print(f"🚀 解冻训练: {num_train} 训练样本, {num_val} 验证样本, batch size = {batch_size}")

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=Lr),
                      metrics=['categorical_accuracy'])

        train_dataloader = ClsDatasets(lines[:num_train], input_shape, batch_size, num_classes, train=True)
        val_dataloader = ClsDatasets(lines[num_train:], input_shape, batch_size, num_classes, train=False)

        model.fit_generator(
            generator=train_dataloader,
            steps_per_epoch=epoch_step,
            validation_data=val_dataloader,
            validation_steps=epoch_step_val,
            epochs=Epoch,
            initial_epoch=Freeze_Epoch,
            use_multiprocessing=num_workers > 1,
            workers=num_workers,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )
