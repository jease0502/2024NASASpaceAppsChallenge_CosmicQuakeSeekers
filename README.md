# Datat analysis

## (1) Dataset_Compare.ipynb

In this ipynb, we mainly observed the differences between the open earthquake data of the moon and the earth, and also analyzed the differences between the moon data and Mars data, which proved that the difference between the two is too large, and the existing labels are different.

## (2) Data_Exploration.ipynb
In this notebook, we primarily demonstrate the data mining process and analyze the optimal data pipeline.

### Research on Data Augmentation Methods
#### Signal Scaling (Zoom In/Out)
We demonstrate how signal scaling impacts model prediction reliability. By scaling signals at different levels, we compare the model's performance (e.g., classification accuracy or prediction error) to observe if the scaled signals can still capture seismic characteristics effectively.

#### Adding Random Noise
We illustrate how adding different levels of random noise to the original seismic signal affects model predictions, examining the robustness of the model in the presence of noise.

#### Random Shuffling
We investigate the effect of randomly shuffling parts of the signal on the model. For example, by randomly rearranging the main structure of the seismic waveform, we assess whether the model can still learn effectively and make accurate predictions.

### Application of Downsampling and Compressed Sensing
We describe how downsampling is used to reduce the sampling rate of the data, thereby lowering processing costs. We also apply compressed sensing techniques to compress the data while retaining key features to minimize data size.

We showcase the application of downsampling and compressed sensing on lunar seismic data, comparing the accuracy of signal reconstruction before and after downsampling, and discussing the quality of reconstruction using compressed sensing.

### Noise Filtering in Waveforms
We introduce the noise filtering methods used, such as high-pass or low-pass filters, or wavelet transforms, to remove background noise from signals. We present waveform plots before and after filtering and compare how noise reduction influences the model's prediction accuracy.

### Research on Sliding Window Techniques
We explain the application of sliding window techniques and discuss how to determine the optimal window size to capture a complete seismic waveform.

By experimenting with different window sizes, we analyze the effects of windows that are too small, which might lead to incomplete seismic waveforms, and those that are too large, which could introduce unnecessary information. The goal is to set an appropriate window size to improve the model's performance.

###  Conclusion and Future Work
We summarize the findings from the data mining process, discussing the impact of different data augmentation methods on the model, as well as the effectiveness of compression and noise filtering on signal processing.

Future improvements could include more precise data augmentation techniques, more efficient compression methods, or exploring alternative noise reduction approaches.


# AI training

## Building the Training Dataset
This is the data input pipeline, where we generate the data required for training. During this process, we detect and discard multiple overlapping waveforms occurring within a specific time period in the dataset. In addition to random noise and random shuffle, we also introduce random shift, and then convert the data to Short-Time Fourier Transform (STFT) representations and save them.

    python build_data

## Training the AI Model
In this script, we provide different model configurations presented in our slides. We also apply noise injection to the input images during the data pipeline to enhance robustness. The script calculates the number of parameters and plots confusion matrices along with accuracy and loss curves.

    python train.py

## Testing
We provide the input data, along with the same data input pipeline, to visualize the prediction results.

    python test.py

## Quantization
To further reduce the model size, we perform post-training quantization (PTQ), further decreasing the model's footprint.

    python PTQ.py


We also saved the best weights from different versions of training, along with training detailed .

Additionally, we provided the training code for the end-to-end model, which allows others to reproduce our results and further explore the model's capabilities.