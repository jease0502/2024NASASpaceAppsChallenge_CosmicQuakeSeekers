import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling1D, Dense, Dropout, Input, Softmax, Reshape, Add, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import tensorflow_model_optimization as tfmot

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 讀取資料夾的圖像並標籤
def load_images_and_labels(data_directory, img_height, img_width):
    images = []
    labels = []
    
    # 定義 mq_type 的編碼
    mq_type_encoding = {'impact_mq': 0, 'deep_mq': 1, 'shallow_mq': 2, 'no_wave': 3}
    num_classes = len(mq_type_encoding)

    # 處理資料夾中的所有檔案
    for subdir in ['0', '1']:
        dir_path = os.path.join(data_directory, subdir)
        
        for filename in os.listdir(dir_path):
            # print("dir_path", dir_path)
            filepath = os.path.join(dir_path, filename)
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (img_width, img_height))
            images.append(image)
            
            # 從檔名中提取 mq_type
            mq_type_str = filename.split('_')[-1].split('.')[0]
            mq_type = int(mq_type_str)
            
            # 生成 one-hot 編碼
            one_hot_label = np.zeros(num_classes)
            one_hot_label[mq_type] = 1
            labels.append(one_hot_label)
            # print("np.array(one_hot_label).shape", np.array(one_hot_label))
    print("labels.shape", np.array(labels).shape)
    return np.array(images), np.array(labels)

# 噪聲注入
def inject_noise(images, noise_factor=0.1):
    noise = np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    images_noisy = images + noise_factor * noise
    images_noisy = np.clip(images_noisy, 0., 1.)
    return images_noisy


# # # # 建立堆疊的 CNN 模型
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
# from tensorflow.keras.models import Model

# def create_stacked_mobilenetv2_model(img_height, img_width):
#     # MobileNetV2 backbone, 不包含最上層的全連接層
#     base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')


#     # 輸入層
#     inputs = Input(shape=(img_height, img_width, 3))

#     # 使用 MobileNetV2 的卷積層作為 backbone
#     x = base_model(inputs)
    
#     # 全局平均池化
#     x = GlobalAveragePooling2D()(x)
    
#     # 全連接層 + Dropout
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.25)(x)
    
#     # 輸出層
#     outputs = Dense(4, activation='softmax')(x)
    
#     model = Model(inputs=inputs, outputs=outputs)
#     return model

# from tensorflow.keras.layers import Input, Reshape, LSTM, Dense, Dropout
# from tensorflow.keras.models import Model

# def create_stacked_lstm_model(img_height, img_width):
#     # 輸入層
#     inputs = Input(shape=(img_height, img_width, 3))
    
#     # 將圖像展平成序列數據，將高度和寬度展平成序列長度，並保留 RGB 通道
#     # 假設將每一行視為一個時間步
#     x = Reshape((img_height, img_width * 3))(inputs)
    
#     # 第一層 LSTM
#     x = LSTM(128, return_sequences=True)(x)
    
#     # 第二層 LSTM
#     x = LSTM(64)(x)
    
#     # 全連接層 + Dropout
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.25)(x)
    
#     # 輸出層
#     outputs = Dense(4, activation='softmax')(x)
    
#     model = Model(inputs=inputs, outputs=outputs)
#     return model

# def create_lstm_model(timesteps, features):
    
#     # LSTM 層
#     model.add(LSTM(64, return_sequences=True, input_shape=(timesteps, features)))
#     model.add(Dropout(0.2))
    
#     model.add(LSTM(64))
#     model.add(Dropout(0.2))
    
#     # 全連接層
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.5))
    
#     # 輸出層
#     model.add(Dense(1, activation='sigmoid'))
    
#     return model



# 建立堆疊的 CNN 模型
def create_stacked_cnn_with_attention(img_height, img_width):
    inputs = Input(shape=(img_height, img_width, 3))
    
    # 第一層卷積 + 池化
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # 第二層卷積 + 池化
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # 第三層卷積 + 池化
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    # 獲取特徵圖的空間維度
    feature_map_shape = x.shape[2]  # 獲取寬度
    
    # 引入位置編碼
    position_encoding = tf.range(start=0, limit=feature_map_shape, delta=1)
    position_encoding = tf.keras.layers.Embedding(input_dim=feature_map_shape, output_dim=128)(position_encoding)
    position_encoding = tf.expand_dims(position_encoding, axis=0)  # 增加 batch 維度
    position_encoding = tf.expand_dims(position_encoding, axis=1)  # 增加 height 維度
    position_encoding = tf.tile(position_encoding, [tf.shape(x)[0], tf.shape(x)[1], 1, 1])  # 與特徵圖匹配
    x = Add()([x, position_encoding])
    
    # 自注意力機制
    x = Reshape((-1, 128))(x)  # 調整形狀以適應自注意力
    attention_scores = tf.matmul(x, x, transpose_b=True)  # 計算注意力分數
    attention_scores = Softmax()(attention_scores)  # 應用 softmax
    x = tf.matmul(attention_scores, x)  # 應用注意力權重
    
    # 使用 GlobalAveragePooling1D
    x = GlobalAveragePooling1D()(x)
    
    # 全連接層 + Dropout
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # 輸出層
    outputs = Dense(4, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 計算模型的 FLOPs 和參數量
def calculate_flops_and_params(model):
    # 計算參數量
    model.summary()
    total_params = model.count_params()
    param_size_kb = total_params * 4 / 1024  # 每個參數4字節，轉換為KB
    print(f"Total Parameters: {total_params}")
    print(f"Parameter Size: {param_size_kb:.2f} KB")
    
    # 計算 FLOPs
    concrete_func = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete_func.get_concrete_function(tf.TensorSpec([1, IMG_HEIGHT, IMG_WIDTH, 3], model.inputs[0].dtype))
    
    # 使用 TensorFlow 的 profiler 來計算 FLOPs
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
        print(f"FLOPs: {flops.total_float_ops}")




# 每10個epoch畫圖並保存
def plot_loss_and_accuracy(train_loss, val_loss, train_acc, val_acc, epoch):
    plt.figure(figsize=(12, 4))

    # Loss 圖
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.title("Loss")

    # Accuracy 圖
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")

    # 保存圖表
    plot_path = f'./training_plots/epoch_{epoch + 1}.png'
    os.makedirs('./training_plots', exist_ok=True)  # 確保資料夾存在
    plt.savefig(plot_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, epoch):
    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred)
    
    # 計算準確度
    accuracy = accuracy_score(y_true, y_pred)
    
    # 顯示混淆矩陣
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['impact_mq', 'deep_mq', 'shallow_mq', 'no_wave'])
    disp.plot(cmap=plt.cm.Blues)
    
    # 在標題中添加準確度
    plt.title(f'Confusion Matrix at Epoch {epoch + 1} - Accuracy: {accuracy:.5f}')
    
    # 自動調整佈局以確保標籤不被遮擋
    plt.tight_layout()
    
    # 保存混淆矩陣的圖表
    plot_path = f'./confusion_matrices/epoch_{epoch + 1}.png'
    os.makedirs('./confusion_matrices', exist_ok=True)  # 確保資料夾存在
    plt.savefig(plot_path)
    plt.close()

if __name__ == "__main__":
    

    # 設定參數
    data_directory = './data/lunar/training/output/'
    IMG_HEIGHT = 64
    IMG_WIDTH = 128
    
    epochs = 101
    batch_size = 32
    checkpoint_filepath = './best_weightsh5'
    initial_learning_rate = 0.001
    train = True


    
    # # 創建 CNN 模型
    # model = create_stacked_cnn_model(IMG_HEIGHT, IMG_WIDTH)
    model = create_stacked_cnn_with_attention(IMG_HEIGHT, IMG_WIDTH)

    # 計算模型的 FLOPs 和參數量
    calculate_flops_and_params(model)
    if train:
        # 加載數據
        X, y = load_images_and_labels(data_directory, IMG_HEIGHT, IMG_WIDTH)

        # 將數據標準化 (歸一化至 [0, 1])
        X = X / 255.0

        # 切分數據集為訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # # 加入噪聲至訓練數據
        # X_train_noisy = inject_noise(X_train)
        # # 設置學習率衰減
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)

        optimizer = Adam(learning_rate=lr_schedule)

        # 自定義訓練迴圈
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        train_acc_metric = tf.keras.metrics.BinaryAccuracy()
        val_acc_metric = tf.keras.metrics.BinaryAccuracy()

        # 用於繪製的變量
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []

        best_val_acc = 0.0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0.0
            num_batches = 0

            # 加入噪聲至訓練數據
            X_train_noisy = inject_noise(X_train)

            # 訓練迴圈
            for step in range(0, len(X_train_noisy), batch_size):
                X_batch = X_train_noisy[step:step + batch_size]
                y_batch = y_train[step:step + batch_size]
                

                with tf.GradientTape() as tape:
                    logits = model(X_batch, training=True)
                    loss_value = loss_fn(y_batch, logits)
                
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                epoch_loss += loss_value.numpy()
                train_acc_metric.update_state(y_batch, logits)
                num_batches += 1

            train_loss = epoch_loss / num_batches
            train_acc = train_acc_metric.result().numpy()
            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)
            train_acc_metric.reset_states()

            # 驗證
            val_logits = model(X_test, training=False)
            val_loss = loss_fn(y_test, val_logits).numpy()  # y_test 和 val_logits 的形狀應該一致
            val_acc_metric.update_state(y_test, val_logits)
            val_acc = val_acc_metric.result().numpy()
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            val_acc_metric.reset_states()

            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

            # 模型檢查點
            if val_acc > best_val_acc:
                print("Saving best model weights...")
                model.save(checkpoint_filepath)
                best_val_acc = val_acc

            # 每10個epoch保存圖表和混淆矩陣
            if (epoch + 1) % 10 == 0:
                plot_loss_and_accuracy(train_loss_history, val_loss_history, train_acc_history, val_acc_history, epoch)

                # 計算混淆矩陣
                val_logits = model(X_test, training=False)
                y_pred_classes = np.argmax(val_logits, axis=1)
                y_true_classes = np.argmax(y_test, axis=1)
                plot_confusion_matrix(y_true_classes, y_pred_classes, epoch)

        # 評估模型
        model.load_weights(checkpoint_filepath)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print(f'\nTest accuracy: {test_acc}')

        # 計算混淆矩陣
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_true_classes, y_pred_classes)

        # 繪製混淆矩陣
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['impact_mq', 'deep_mq', 'shallow_mq', 'no_wave'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.savefig('./confusion_matrix.png')
        plt.close()

        # 保存模型
        model.save('kws_seismic_model_custom.h5')