import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.fftpack import dct, idct
import tensorflow as tf
from tensorflow.keras.models import load_model


# 壓縮感知函數
def apply_compressed_sensing(data, compression_rate):
    signal_length = len(data)
    dct_coefficients = dct(data, norm='ortho')
    k = int(signal_length * compression_rate)
    compressed_signal = np.zeros_like(dct_coefficients)
    compressed_signal[:k] = dct_coefficients[:k]
    reconstructed_signal = idct(compressed_signal, norm='ortho')
    return reconstructed_signal

# 下采样函数
def downsample_data(times, data, downsample_rate):
    downsampled_times = times[::downsample_rate]
    downsampled_data = data[::downsample_rate]
    return downsampled_times, downsampled_data

# 生成 STFT 圖像
def create_stft_image(signal_segment, idx, output_directory, sampling_rate):
    f, t, Zxx = stft(signal_segment, fs=sampling_rate, nperseg=256)
    Zxx_magnitude = np.abs(Zxx)
    Zxx_magnitude_normalized = cv2.normalize(Zxx_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    stft_colored = cv2.applyColorMap(Zxx_magnitude_normalized, cv2.COLORMAP_JET)
    stft_image_filename = f'segment_{idx}_stft.png'
    stft_image_filepath = os.path.join(output_directory, stft_image_filename)
    cv2.imwrite(stft_image_filepath, stft_colored)
    return stft_image_filepath

if __name__ == "__main__":
    

    # 設置常量和路徑
    model_filepath = './best_weights.h5'
    csv_file = 'input.csv'
    output_directory = './data/lunar/training/output/test_segments'
    sampling_rate = 6.625
    window_size_seconds = 8000
    window_size = int(window_size_seconds * sampling_rate)
    stride = 10000
    downsample_rate = 10
    compression_rate = 0.5

    # 載入模型
    model = load_model(model_filepath)

    # 載入 CSV 檔案
    data_cat = pd.read_csv(csv_file)
    csv_times = data_cat['time_rel(sec)'].values
    csv_data = data_cat['velocity(m/s)'].values

    # 創建輸出目錄
    os.makedirs(output_directory, exist_ok=True)

    # 用於記錄每個片段的預測機率和類別
    probabilities = []
    time_points = []
    predicted_classes = []  # 新增一個列表來存儲預測的類別

    # 逐步處理訊號
    for start_idx in range(0, len(csv_data) - window_size, stride):
        end_idx = start_idx + window_size
        signal_segment = csv_data[start_idx:end_idx]

        # 下采样和压缩感知
        downsampled_times, downsampled_segment = downsample_data(csv_times[start_idx:end_idx], signal_segment, downsample_rate)
        compressed_segment = apply_compressed_sensing(downsampled_segment, compression_rate)

        # 生成 STFT 圖像
        stft_image_path = create_stft_image(compressed_segment, start_idx, output_directory, sampling_rate)

        # 讀取並調整圖像尺寸為模型所需的輸入大小
        img = cv2.imread(stft_image_path)
        
        img = cv2.resize(img, (128, 64))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        # 使用模型進行預測
        prediction = model.predict(img)
        pred_class = np.argmax(prediction, axis=1)[0]  # 使用 argmax 獲取預測類別
        pred_prob = prediction[0][pred_class]  # 獲取該類別的預測機率

        # 調整機率顯示
        if pred_class == 3:  # 如果預測為 no_wave
            adjusted_prob = 1 - pred_prob  # 置信度越高，紅線越靠近 0
        else:
            adjusted_prob = pred_prob  # 正常顯示置信度

        # 將每個片段的機率和類別存儲起來
        probabilities.append(adjusted_prob)
        # 將時間點往前移動 1500
        time_points.append((end_idx - 15000) / sampling_rate)
        predicted_classes.append(pred_class)  # 存儲預測的類別

        # 顯示每個區段的預測機率和類別
        print(f'Segment {start_idx} to {end_idx} predicted class: {pred_class}, probability: {pred_prob:.4f}')

        # 根據預測類別進行輸出
        if pred_class == 0:
            print(f'--> Segment {start_idx} is impact_mq (Probability: {pred_prob:.4f})')
        elif pred_class == 1:
            print(f'--> Segment {start_idx} is deep_mq (Probability: {pred_prob:.4f})')
        elif pred_class == 2:
            print(f'--> Segment {start_idx} is shallow_mq (Probability: {pred_prob:.4f})')
        elif pred_class == 3:
            print(f'--> Segment {start_idx} is no_wave (Probability: {pred_prob:.4f})')

    # 繪製機率變化與原始波形的重疊折線圖
    plt.figure(figsize=(12, 6))
    amplification_factor = 1e8
    plt.plot(csv_times, csv_data * amplification_factor, label='Original Waveform', color='blue', alpha=0.5)
    plt.plot(time_points, probabilities, label='Probability of Seismic Waves', color='red', linewidth=2)

    # 繪製每個片段的機率點，根據地震類型使用不同顏色
    colors = ['green', 'orange', 'purple', 'black']  # 定義不同類別的顏色
    labels = ['impact_mq', 'deep_mq', 'shallow_mq', 'no_wave']  # 類別標籤

    # 用於追蹤哪些標籤經被添加
    added_labels = set()

    for i, (time_point, prob, pred_class) in enumerate(zip(time_points, probabilities, predicted_classes)):
        if labels[pred_class] not in added_labels:
            plt.scatter(time_point, prob, color=colors[pred_class], label=labels[pred_class])
            added_labels.add(labels[pred_class])
        else:
            plt.scatter(time_point, prob, color=colors[pred_class])

    # 確保所有類別的標籤都顯示在圖例中，即使某些類別在數據中不存在
    for label, color in zip(labels, colors):
        if label not in added_labels:
            plt.scatter([], [], color=color, label=label)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude / Probability')
    plt.title('Seismic Wave Detection with Original Waveform Overlay')
    plt.grid(True)
    plt.legend()

    # 保存重疊的折線圖
    plot_filename = os.path.join(output_directory, 'seismic_wave_probabilities_with_waveform.png')
    plt.savefig(plot_filename)
    print(f'Probability plot with waveform saved to {plot_filename}')
    plt.show()