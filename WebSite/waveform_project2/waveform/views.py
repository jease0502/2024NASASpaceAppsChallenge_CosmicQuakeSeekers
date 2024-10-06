from django.shortcuts import render
import pandas as pd
import numpy as np
from scipy.signal import stft
from scipy.fftpack import dct, idct
import cv2
import os
import json
import base64
from tensorflow.keras.models import load_model
from django.http import JsonResponse

# 加载模型
model = load_model('./best_weights.h5')

# 设置常量
sampling_rate = 6.625  # 每秒的采样频率
window_size_seconds = 2000  # 每个片段的时间长度（秒）
window_size = int(window_size_seconds * sampling_rate)
downsample_rate = 10  # 下采样的比例
compression_rate = 0.5  # 压缩感知的比例
output_directory = './static'  # STFT 图像保存目录

# 确保输出目录存在
os.makedirs(output_directory, exist_ok=True)

# 压缩感知函数
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

# 生成 STFT 图像并保存
def create_stft_image(signal_segment, idx, output_directory):
    f, t, Zxx = stft(signal_segment, fs=sampling_rate, nperseg=256)
    Zxx_magnitude = np.abs(Zxx)
    Zxx_magnitude_normalized = cv2.normalize(
        Zxx_magnitude, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    stft_colored = cv2.applyColorMap(Zxx_magnitude_normalized, cv2.COLORMAP_JET)
    stft_image_path = os.path.join(output_directory, f'stft_image_{idx}.png')
    cv2.imwrite(stft_image_path, stft_colored)
    return stft_image_path

# 使用模型进行预测
def model_predict(stft_image_path):
    # 读取并调整图像尺寸为模型所需的输入大小
    img = cv2.imread(stft_image_path)
    if img is None:
        print(f"无法读取图像：{stft_image_path}")
        return None, None, None
    img = cv2.resize(img, (128, 64))  # 宽度128，高度64
    img = np.expand_dims(img, axis=0)  # 增加 batch 维度
    img = img / 255.0  # 归一化至 [0, 1] 范围

    prediction = model.predict(img)
    pred_class = np.argmax(prediction, axis=1)[0]  # 获取预测的类别
    pred_prob = prediction[0][pred_class]  # 获取该类别的预测概率
    if pred_class == 3:  # 如果预测为 no_wave
        adjusted_prob = 1 - pred_prob  # 置信度越高，红线越靠近 0
    else:
        adjusted_prob = pred_prob  # 正常显示置信度

    # 使用模型进行预测
    print("预测的类别索引：", pred_class)
    print("预测的概率：", pred_prob)

    # 直接使用原始概率
    adjusted_prob = pred_prob

    return adjusted_prob, pred_class, pred_prob




# 读取并解析 CSV 数据
def waveform_view(request):
    # 载入 CSV 文件
    csv_file_path = 'input2.csv'  # 确保使用正确的文件路径
    data = pd.read_csv(csv_file_path)

    # 提取 time 和 velocity 数据，转换为列表
    times = data['time_rel(sec)'].tolist()
    velocities = data['velocity(m/s)'].tolist()

    # 下采样和压缩感知处理
    downsampled_times, downsampled_velocities = downsample_data(
        np.array(times), np.array(velocities), downsample_rate
    )
    compressed_velocities = apply_compressed_sensing(
        downsampled_velocities, compression_rate
    ).tolist()

    # 初始化 STFT 图像和预测结果
    stft_image_base64 = ''
    initial_prediction = 'N/A'
    initial_pred_class = 3  # 默认设置为 no_wave

    # 将数据传递到前端
    context = {
        'times': json.dumps(times),
        'velocities': json.dumps(velocities),
        'downsampled_times': json.dumps(downsampled_times.tolist()),
        'downsampled_velocities': json.dumps(downsampled_velocities.tolist()),
        'compressed_velocities': json.dumps(compressed_velocities),
        'stft_image': stft_image_base64,
        'prediction': initial_prediction,
        'pred_class': initial_pred_class
    }

    return render(request, 'waveform.html', context)


# 处理 STFT 图像更新和模型预测的 API
def update_stft_image(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        signal_data = request_data.get('signal', [])
        current_time = request_data.get('current_time', None)  # 获取前端发送的时间信息

        if not signal_data or current_time is None:
            return JsonResponse({'error': 'No signal data or current time provided'}, status=400)

        # 将前端传来的信号数据转换为 NumPy 数组
        signal_segment = np.array(signal_data)
        print(f"Received signal_segment of length: {len(signal_segment)}")

        if len(signal_segment) < 256:  # 检查信号段长度是否足够
            return JsonResponse({'error': 'Signal segment too short for STFT'}, status=400)

        # 下采样和压缩感知
        downsampled_times_segment, downsampled_segment = downsample_data(
            np.arange(len(signal_segment)), signal_segment, downsample_rate
        )
        compressed_segment = apply_compressed_sensing(
            downsampled_segment, compression_rate
        )

        # 生成 STFT 图像并保存到临时文件
        stft_image_path = create_stft_image(
            compressed_segment, idx=current_time, output_directory=output_directory
        )

        if stft_image_path is None:
            return JsonResponse({'error': 'Failed to create STFT image'}, status=500)

        # 使用模型进行预测
        adjusted_prob, pred_class, pred_prob = model_predict(stft_image_path)

        if adjusted_prob is None:
            return JsonResponse({'error': 'Model prediction failed'}, status=500)

        # 将 STFT 图像转换为 base64 格式
        with open(stft_image_path, "rb") as image_file:
            stft_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        print("STFT 图像路径:", stft_image_path)
        # 删除临时 STFT 图像文件
        os.remove(stft_image_path)

        return JsonResponse({
            'stft_image': stft_image_base64,
            'adjusted_prob': float(adjusted_prob),
            'pred_prob': float(pred_prob),
            'pred_class': int(pred_class),
            'current_time': current_time  # 返回时间信息
        })
    return JsonResponse({'error': 'Invalid request method'}, status=405)


