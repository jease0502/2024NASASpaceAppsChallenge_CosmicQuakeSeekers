import tensorflow as tf

# 加載模型
model = tf.keras.models.load_model('461kb_model/best_weights.h5')

# 創建 TFLite 轉換器
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 啟用權重量化
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 量化模型
tflite_model = converter.convert()

# 將量化後的模型保存為TFLite格式
with open('model_quant.tflite', 'wb') as f:
    f.write(tflite_model)
import os

# 原模型的大小
original_model_size = os.path.getsize('461kb_model/best_weights.h5') # 或 .pth
print(f'原始模型大小: {original_model_size / (1024 * 1024):.2f} MB')

# 量化模型的大小
quant_model_size = os.path.getsize('model_quant.tflite') # 或 .pth
print(f'量化模型大小: {quant_model_size / (1024 * 1024):.2f} MB')
