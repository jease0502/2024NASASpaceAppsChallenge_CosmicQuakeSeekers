import pandas as pd
import numpy as np
import hashlib
import os
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.fftpack import dct, idct
import cv2

def apply_compressed_sensing(data, compression_rate):
    signal_length = len(data)
    dct_coefficients = dct(data, norm='ortho')
    k = int(signal_length * compression_rate)
    compressed_signal = np.zeros_like(dct_coefficients)
    compressed_signal[:k] = dct_coefficients[:k]
    reconstructed_signal = idct(compressed_signal, norm='ortho')
    return reconstructed_signal, compressed_signal

def downsample_data(times, data, downsample_rate):
    downsampled_times = times[::downsample_rate]
    downsampled_data = data[::downsample_rate]
    return downsampled_times, downsampled_data

def plot_loss_and_accuracy(train_loss, val_loss, train_acc, val_acc, epoch, output_directory):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plot_path = os.path.join(output_directory, f'training_plots/epoch_{epoch + 1}.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

if __name__ == "__main__":
    # 初始化必要的目录路径
    cat_directory = './data/lunar/training/catalogs/'
    cat_file = os.path.join(cat_directory, 'apollo12_catalog_GradeA_final.csv')
    data_directory = './data/lunar/training/data/S12_GradeA/'
    output_directory = './data/lunar/training/output/'
    no_wave_directory = os.path.join(output_directory, '0')
    wave_directory = os.path.join(output_directory, '1')

    # 确保输出目录存在
    os.makedirs(no_wave_directory, exist_ok=True)
    os.makedirs(wave_directory, exist_ok=True)

    # 读取目录文件
    cat = pd.read_csv(cat_file)
    row_count = len(cat)
    print(f"總行數: {row_count}")

    # 创建一个字典来存储波形数据的哈希值和对应的文件信息
    hash_to_data = {}
    # 定義 mq_type 的編碼
    mq_type_encoding = {
        'impact_mq': [1, 0, 0, 0],
        'deep_mq': [0, 1, 0, 0],
        'shallow_mq': [0, 0, 1, 0],
        'no_wave': [0, 0, 0, 1]
    }

    for i in range(row_count):
        print(f"Processing row {i}")
        row = cat.iloc[i]
        arrival_time_rel = row['time_rel(sec)']
        test_filename = row['filename']
        mq_type = row['mq_type']
        # print("row", row['mq_type'])

        print(f"Filename: {test_filename}, Arrival Time: {arrival_time_rel}")

        # 读取数据文件
        csv_file = os.path.join(data_directory, f'{test_filename}.csv')
        data_cat = pd.read_csv(csv_file)
        # print(data_cat.head())
        # 提取时间和速度数据
        csv_times = data_cat['time_rel(sec)'].values
        csv_data = data_cat['velocity(m/s)'].values

        # 计算波形数据的哈希值
        data_hash = hashlib.sha256(csv_data.tobytes()).hexdigest()

        # 将文件信息添加到字典中
        if data_hash not in hash_to_data:
            hash_to_data[data_hash] = {
                'csv_times': csv_times,
                'csv_data': csv_data,
                'mq_type': mq_type_encoding[mq_type],
                'arrival_times': [],
                'num_waves': 0,
            }
        hash_to_data[data_hash]['arrival_times'].append(arrival_time_rel)
        hash_to_data[data_hash]['num_waves'] = len(hash_to_data[data_hash]['arrival_times'])

    # 设置参数
    downsample_rate = 10
    compression_rate = 0.5
    sampling_rate = 6.625
    n_fft = 256

    # 准备收集所有元数据的列表
    metadata_records = []
    no_wave_metadata_records = []

    # 遍历哈希字典，进行数据增强、绘图并保存波形到对应的資料夾
    for data_hash, info in hash_to_data.items():
        print(f"Processing data for data_hash: {data_hash}")
        
        csv_times = info['csv_times']
        csv_data = info['csv_data']
        arrival_times = info['arrival_times']
        mq_type = info['mq_type']
        # 对数据进行下采样和压缩感知
        csv_times_downsampled, csv_data_downsampled = downsample_data(csv_times, csv_data, downsample_rate)
        csv_data_compressed, compressed_coefficients = apply_compressed_sensing(csv_data_downsampled, compression_rate)

        # 進行有震波資料的處理和保存到 1 資料夾
        for augment_idx in range(100):
            print(f"Augmenting {augment_idx + 1}/10")

            # 数据增强：将整个波形分成10段，包括到达时间段
            N = 10
            segment_indices = np.linspace(0, len(csv_times_downsampled), num=N+1, dtype=int)
            segments = []

            for i in range(N):
                seg_start = segment_indices[i]
                seg_end = segment_indices[i + 1]
                segment_times = csv_times_downsampled[seg_start:seg_end]
                segment_data = csv_data_compressed[seg_start:seg_end]
                segment_arrival_time_offsets = []

                for arrival_time in arrival_times:
                    if segment_times[0] <= arrival_time <= segment_times[-1]:
                        arrival_time_offset = arrival_time - segment_times[0]
                        segment_arrival_time_offsets.append(arrival_time_offset)
                
                segment = {
                    'times': segment_times,
                    'data': segment_data,
                    'arrival_time_offsets': segment_arrival_time_offsets,
                    'length': len(segment_times),
                    'mq_type': mq_type
                }
                segments.append(segment)

            np.random.shuffle(segments)

            dt = np.mean(np.diff(csv_times_downsampled))
            augmented_data = []
            augmented_times = []
            new_arrival_times = []
            cumulative_time = 0

            for segment in segments:
                segment_length = segment['length']
                adjusted_times = np.arange(segment_length) * dt + cumulative_time
                augmented_times.append(adjusted_times)
                augmented_data.append(segment['data'])

                for arrival_time_offset in segment['arrival_time_offsets']:
                    new_arrival_time = cumulative_time + arrival_time_offset
                    new_arrival_times.append(new_arrival_time)

                cumulative_time += segment_length * dt

            augmented_times = np.concatenate(augmented_times)
            augmented_data = np.concatenate(augmented_data)

            noise = np.random.normal(0, np.std(augmented_data) * 0.01, size=augmented_data.shape)
            augmented_data_with_noise = augmented_data + noise

            x = np.random.uniform(2000, 6000)
            y = 8000 - x
            start_time = new_arrival_times[0] - x
            end_time = new_arrival_times[0] + y

            mask = (augmented_times >= start_time) & (augmented_times <= end_time)
            final_augmented_times = augmented_times[mask]
            final_augmented_data = augmented_data_with_noise[mask]
            final_arrival_times = [arrival for arrival in new_arrival_times if start_time <= arrival <= end_time]

            if len(final_arrival_times) != 1:
                print(f"Skipping augmentation {augment_idx + 1} due to multiple arrival times")
                continue

            f, t, Zxx = stft(final_augmented_data, fs=sampling_rate, nperseg=n_fft)
            Zxx_magnitude = np.abs(Zxx)
            Zxx_magnitude_normalized = cv2.normalize(Zxx_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            stft_colored = cv2.applyColorMap(Zxx_magnitude_normalized, cv2.COLORMAP_JET)

            stft_image_filename = f'{data_hash}_wave_stft_{augment_idx + 1}_{mq_type}.png'  # 使用變數 mq_type
            stft_image_filepath = os.path.join(wave_directory, stft_image_filename)
            cv2.imwrite(stft_image_filepath, stft_colored)

            metadata_record = {
                'data_hash': data_hash,
                'stft_image_file': stft_image_filepath,
                'num_waves': info['num_waves'],
                'new_arrival_times': ','.join(map(str, final_arrival_times)),
                'x': x,
                'y': y,
                'mq_type': mq_type_encoding[row['mq_type']]  # 使用獨熱編碼
            }
            metadata_records.append(metadata_record)

        # 進行無震波資料的處理和保存到 0 資料夾
        for augment_idx in range(100):
            print(f"Augmenting no-wave {augment_idx + 1}/10")

            N = 10
            segment_indices = np.linspace(0, len(csv_times_downsampled), num=N+1, dtype=int)
            segments = []

            for i in range(N):
                seg_start = segment_indices[i]
                seg_end = segment_indices[i + 1]
                segment_times = csv_times_downsampled[seg_start:seg_end]
                segment_data = csv_data_compressed[seg_start:seg_end]
                segment = {
                    'times': segment_times,
                    'data': segment_data,
                    'length': len(segment_times),
                    'mq_type': 3

                }
                segments.append(segment)

            np.random.shuffle(segments)

            dt = np.mean(np.diff(csv_times_downsampled))
            augmented_data = []
            augmented_times = []
            cumulative_time = 0

            for segment in segments:
                segment_length = segment['length']
                adjusted_times = np.arange(segment_length) * dt + cumulative_time
                augmented_times.append(adjusted_times)
                augmented_data.append(segment['data'])

                cumulative_time += segment_length * dt

            augmented_times = np.concatenate(augmented_times)
            augmented_data = np.concatenate(augmented_data)

            noise = np.random.normal(0, np.std(augmented_data) * 0.01, size=augmented_data.shape)
            augmented_data_with_noise = augmented_data + noise

            x = np.random.uniform(2000, 6000)
            y = 8000 - x
            start_time = np.random.uniform(augmented_times[0], augmented_times[-1] - (x + y))
            end_time = start_time + x + y

            mask = (augmented_times >= start_time) & (augmented_times <= end_time)
            final_augmented_times = augmented_times[mask]
            final_augmented_data = augmented_data_with_noise[mask]

            f, t, Zxx = stft(final_augmented_data, fs=sampling_rate, nperseg=n_fft)
            Zxx_magnitude = np.abs(Zxx)
            Zxx_magnitude_normalized = cv2.normalize(Zxx_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            stft_colored = cv2.applyColorMap(Zxx_magnitude_normalized, cv2.COLORMAP_JET)

            stft_image_filename = f'{data_hash}_no_wave_stft_{augment_idx + 1}_3.png'  # 使用固定值 3
            stft_image_filepath = os.path.join(no_wave_directory, stft_image_filename)
            cv2.imwrite(stft_image_filepath, stft_colored)

            no_wave_metadata_record = {
                'data_hash': data_hash,
                'stft_image_file': stft_image_filepath,
                'x': x,
                'y': y,
                'mq_type': mq_type_encoding['no_wave']  # 使用獨熱編碼
            }
            no_wave_metadata_records.append(no_wave_metadata_record)

    # 保存有震波和无震波的元数据到 CSV 文件
    metadata_csv_file = os.path.join(output_directory, 'wave_metadata.csv')
    df_metadata = pd.DataFrame(metadata_records)
    df_metadata.to_csv(metadata_csv_file, index=False)
    print(f"所有有震波元数据已保存到 {metadata_csv_file}")

    no_wave_metadata_csv_file = os.path.join(output_directory, 'no_wave_metadata.csv')
    df_no_wave_metadata = pd.DataFrame(no_wave_metadata_records)
    df_no_wave_metadata.to_csv(no_wave_metadata_csv_file, index=False)
    print(f"所有无震波元数据已保存到 {no_wave_metadata_csv_file}")