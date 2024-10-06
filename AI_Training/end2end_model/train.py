import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import cv2
import datetime

# 设置数据路径
data_directory = './data/lunar/training/output/'
metadata_file = os.path.join(data_directory, 'metadata.csv')

# 加载元数据
metadata = pd.read_csv(metadata_file)

# 准备数据集
def load_dataset(metadata, data_directory):
    images = []
    labels = []
    durations = []

    for idx, row in metadata.iterrows():
        # 加载 STFT 图像
        stft_file = row['augmented_stft_plot_file']
        stft_path = os.path.join(data_directory, os.path.basename(stft_file))
        if not os.path.exists(stft_path):
            continue  # 跳过不存在的文件
        # 使用 cv2.imread 加载图像
        image = cv2.imread(stft_path)
        if image is None:
            continue
        # 转换为 RGB 格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

        # 处理到达时间
        arrival_times = row['new_arrival_times']
        if pd.isna(arrival_times) or arrival_times == '':
            arrival_times = []
        else:
            arrival_times = [float(t) for t in arrival_times.split(',')]

        # 构建标签
        # 如果有两个到达时间，则取前两个；不足则用 0 填充
        label_arrival_times = [0.0, 0.0]
        label_confidences = [0.0, 0.0]
        for i in range(min(len(arrival_times), 2)):
            label_arrival_times[i] = arrival_times[i]
            label_confidences[i] = 1.0  # 有到达时间，置信度为 1

        labels.append({
            'arrival_times': label_arrival_times,
            'confidences': label_confidences
        })

        # 读取 total_duration
        total_duration = 86402# row['total_duration']
        durations.append(total_duration)

    images = np.array(images).astype('float32')
    labels_arrival_times = np.array([label['arrival_times'] for label in labels]).astype('float32')
    labels_confidences = np.array([label['confidences'] for label in labels]).astype('float32')
    durations = np.array(durations).astype('float32')

    return images, labels_arrival_times, labels_confidences, durations

# 加载数据集
images, arrival_times, confidences, durations = load_dataset(metadata, data_directory)

# 划分训练集和验证集
X_train, X_val, y_arrival_train, y_arrival_val, y_conf_train, y_conf_val, durations_train, durations_val = train_test_split(
    images, arrival_times, confidences, durations, test_size=0.05, random_state=42)

def output_branch(input):
    conv_2 = tf.keras.layers.Conv2D(32, 3, strides=(1, 1), padding="SAME")(input)
    activation_2 = tf.keras.layers.ReLU()(conv_2)
    x = keras.layers.GlobalAveragePooling2D()(activation_2)
    return x 

# 构建模型
def build_model():
    # 输入层
    inputs = keras.Input(shape=(None, None, 3))
    
    # 使用卷积层作为卷积基
    conv_1 = tf.keras.layers.Conv2D(32, 3, strides=(1, 1), padding="SAME")(inputs)  # 修改此处的 input 为 inputs
    activation_1 = tf.keras.layers.ReLU()(conv_1)
    conv_2 = tf.keras.layers.Conv2D(32, 3, strides=(1, 1), padding="SAME")(activation_1)
    activation_2 = tf.keras.layers.ReLU()(conv_2)

    x1 = output_branch(activation_2)
    x2 = output_branch(activation_2)
    x3 = output_branch(activation_2)
    x4 = output_branch(activation_2)

    # 输出层
    first_arrival_time = keras.layers.Dense(1, name='first_arrival_time')(x1)
    first_confidence = keras.layers.Dense(1, activation='sigmoid', name='first_confidence')(x2)
    second_arrival_time = keras.layers.Dense(1, name='second_arrival_time')(x3)
    second_confidence = keras.layers.Dense(1, activation='sigmoid', name='second_confidence')(x4)

    # 构建模型
    model = keras.Model(inputs=inputs, outputs=[first_arrival_time, first_confidence, second_arrival_time, second_confidence])
    
    return model

# 获取输入形状
input_shape = X_train.shape[1:]  # (height, width, channels)

# 创建模型
model = build_model()

# 定义损失函数和优化器
mse_loss_fn = keras.losses.MeanAbsoluteError()
bce_loss_fn = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=1e-4)

# 设置 TensorBoard 日志记录
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

# 自定义训练循环
epochs = 10000
batch_size = 16

# 创建 TensorFlow 数据集
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_arrival_train, y_conf_train, durations_train))
train_dataset = train_dataset.shuffle(buffer_size=256).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_arrival_val, y_conf_val, durations_val))
val_dataset = val_dataset.batch(batch_size)

# 定义绘图函数，使用 cv2
def plot_predictions(images, true_times, pred_times, pred_confs, durations, epoch, output_dir='predictions'):
    os.makedirs(output_dir, exist_ok=True)
    num_samples = min(5, len(images))  # 只绘制前5个样本
    for i in range(num_samples):
        image = images[i].copy()
        height, width, _ = image.shape

        total_duration = durations[i]

        # 避免除以零
        if total_duration == 0:
            total_duration = 1.0

        # 绘制真实到达时间
        for t in true_times[i]:
            if t > 0:
                x_pixel = int(t / total_duration * width)
                cv2.line(image, (x_pixel, 0), (x_pixel, height - 1), (0, 255, 0), 2)  # 绿色线

        # 绘制预测到达时间，置信度 > 0.5
        for t, conf in zip(pred_times[i], pred_confs[i]):
            if t > 0 and conf > 0.5:
                x_pixel = int(t / total_duration * width)
                cv2.line(image, (x_pixel, 0), (x_pixel, height - 1), (255, 0, 0), 3)  # 红色线

        # 保存图像
        output_path = os.path.join(output_dir, f'epoch_{epoch}_sample_{i}.png')
        image_bgr = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)

# 开始训练
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    # 训练阶段
    train_losses = []
    for step, (x_batch, y_arrival_batch, y_conf_batch, durations_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # 前向传播
            outputs = model(x_batch, training=True)
            y_pred_arrival_first = outputs[0]
            y_pred_conf_first = outputs[1]
            y_pred_arrival_second = outputs[2]
            y_pred_conf_second = outputs[3]


            # 计算损失
            # 对于到达时间，只在置信度为1时计算损失
            mask_first = tf.reshape(y_conf_batch[:, 0], (-1, 1))
            mask_second = tf.reshape(y_conf_batch[:, 1], (-1, 1))

            loss_arrival_first = mse_loss_fn(y_arrival_batch[:, 0:1] * mask_first, y_pred_arrival_first * mask_first)
            loss_arrival_second = mse_loss_fn(y_arrival_batch[:, 1:2] * mask_second, y_pred_arrival_second * mask_second)

            # 置信度损失
            loss_conf_first = bce_loss_fn(y_conf_batch[:, 0], tf.reshape(y_pred_conf_first, (-1,)))
            loss_conf_second = bce_loss_fn(y_conf_batch[:, 1], tf.reshape(y_pred_conf_second, (-1,)))

            # 总损失
            total_loss = loss_arrival_first + loss_arrival_second + loss_conf_first + loss_conf_second

        # 计算梯度并更新参数
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_losses.append(total_loss.numpy())

        # 记录训练损失到 TensorBoard
        with train_summary_writer.as_default():
            tf.summary.scalar('total_loss', total_loss.numpy(), step=epoch * len(train_dataset) + step)
            tf.summary.scalar('loss_arrival_first', loss_arrival_first.numpy(), step=epoch * len(train_dataset) + step)
            tf.summary.scalar('loss_arrival_second', loss_arrival_second.numpy(), step=epoch * len(train_dataset) + step)
            tf.summary.scalar('loss_conf_first', loss_conf_first.numpy(), step=epoch * len(train_dataset) + step)
            tf.summary.scalar('loss_conf_second', loss_conf_second.numpy(), step=epoch * len(train_dataset) + step)

        if step % 100 == 0:
            print(f'Step {step}, Loss: {total_loss.numpy()}')

    train_loss = np.mean(train_losses)
    if epoch % 10==0:
        # 验证阶段
        val_losses = []
        val_images = []
        val_true_times = []
        val_pred_times = []
        val_pred_confs = []
        val_durations = []
        for x_batch, y_arrival_batch, y_conf_batch, durations_batch in val_dataset:
            outputs = model(x_batch, training=False)
            y_pred_arrival_first = outputs[0]
            y_pred_conf_first = outputs[1]
            y_pred_arrival_second = outputs[2]
            y_pred_conf_second = outputs[3]

            # 计算损失
            mask_first = tf.reshape(y_conf_batch[:, 0], (-1, 1))
            mask_second = tf.reshape(y_conf_batch[:, 1], (-1, 1))

            loss_arrival_first = mse_loss_fn(y_arrival_batch[:, 0:1] * mask_first, y_pred_arrival_first * mask_first)
            loss_arrival_second = mse_loss_fn(y_arrival_batch[:, 1:2] * mask_second, y_pred_arrival_second * mask_second)

            loss_conf_first = bce_loss_fn(y_conf_batch[:, 0], tf.reshape(y_pred_conf_first, (-1,)))
            loss_conf_second = bce_loss_fn(y_conf_batch[:, 1], tf.reshape(y_pred_conf_second, (-1,)))

            total_loss = loss_arrival_first + loss_arrival_second + loss_conf_first + loss_conf_second
            val_losses.append(total_loss.numpy())

            # 收集用于绘图的数据
            val_images.extend(x_batch.numpy())
            true_times_batch = y_arrival_batch.numpy()
            val_true_times.extend(true_times_batch)
            pred_times_batch = np.concatenate([y_pred_arrival_first.numpy(), y_pred_arrival_second.numpy()], axis=1)
            val_pred_times.extend(pred_times_batch)
            pred_confs_batch = np.concatenate([y_pred_conf_first.numpy(), y_pred_conf_second.numpy()], axis=1)
            val_pred_confs.extend(pred_confs_batch)
            val_durations.extend(durations_batch.numpy())

        val_loss = np.mean(val_losses)

        # 记录验证损失到 TensorBoard
        with val_summary_writer.as_default():
            tf.summary.scalar('val_total_loss', val_loss, step=epoch)

        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        if epoch % 100==0:
            # 绘制预测结果
            plot_predictions(val_images, val_true_times, val_pred_times, val_pred_confs, val_durations, epoch+1)
            model.save('small_arrive_model{}.h5'.format(epoch))

# 保存模型
