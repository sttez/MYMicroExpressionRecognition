import os
import datetime
import keras
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
from keras import backend as K


class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_dir):
        curr_time = datetime.datetime.now()
        time_str = curr_time.strftime('%Y_%m_%d_%H_%M_%S')
        self.log_dir = os.path.join(log_dir, f"train_{time_str}")
        self.losses = []
        self.val_losses = []

        os.makedirs(self.log_dir, exist_ok=True)

        self.loss_file = os.path.join(self.log_dir, "loss_log.txt")
        self.val_loss_file = os.path.join(self.log_dir, "val_loss_log.txt")
        self.loss_plot_file = os.path.join(self.log_dir, "loss_curve.png")
        self.model_save_path = os.path.join(self.log_dir, "final_model.h5")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        self.losses.append(loss)
        self.val_losses.append(val_loss)

        with open(self.loss_file, 'a') as f:
            f.write(f"{loss}\n")

        with open(self.val_loss_file, 'a') as f:
            f.write(f"{val_loss}\n")

        self.loss_plot()

    def on_train_end(self, logs=None):
        # Save final model with full structure + weights
        self.model.save(self.model_save_path)
        print(f"\n✅ 最终模型已保存至: {self.model_save_path}")

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_losses, 'blue', linewidth=2, label='val loss')

        try:
            smooth_window = 5 if len(self.losses) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, smooth_window, 3),
                     'green', linestyle='--', linewidth=2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_losses, smooth_window, 3),
                     'orange', linestyle='--', linewidth=2, label='smooth val loss')
        except Exception as e:
            print(f"平滑失败: {e}")

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(self.loss_plot_file)
        plt.cla()
        plt.close('all')


class ExponentDecayScheduler(keras.callbacks.Callback):
    def __init__(self, decay_rate, verbose=0):
        super(ExponentDecayScheduler, self).__init__()
        self.decay_rate = decay_rate
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        old_lr = float(K.get_value(self.model.optimizer.lr))
        new_lr = old_lr * self.decay_rate
        K.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose > 0:
            print(f"\n📉 学习率已更新: {old_lr:.6f} -> {new_lr:.6f}")
