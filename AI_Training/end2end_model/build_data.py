import pandas as pd
import numpy as np
import hashlib
import os
import matplotlib.pyplot as plt
from scipy.signal import stft
from multiprocessing import Pool

# Initialize necessary directory paths
cat_directory = './data/lunar/training/catalogs/'
cat_file = os.path.join(cat_directory, 'apollo12_catalog_GradeA_final.csv')
data_directory = './data/lunar/training/data/S12_GradeA/'
output_directory = './data/lunar/training/output/'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Read the catalog file
cat = pd.read_csv(cat_file)
row_count = len(cat)
print(f"Total rows: {row_count}")

# Create a dictionary to store waveform data hashes and corresponding file information
hash_to_data = {}

for i in range(2):
    print(f"Processing row {i}")
    row = cat.iloc[i]
    arrival_time_rel = row['time_rel(sec)']
    test_filename = row['filename']
    print(f"Filename: {test_filename}, Arrival Time: {arrival_time_rel}")

    # Read the data file
    csv_file = os.path.join(data_directory, f'{test_filename}.csv')
    data_cat = pd.read_csv(csv_file)

    # Extract time and velocity data
    csv_times = data_cat['time_rel(sec)'].values
    csv_data = data_cat['velocity(m/s)'].values

    # Compute the hash of the waveform data
    data_hash = hashlib.sha256(csv_data.tobytes()).hexdigest()

    # Add file information to the dictionary
    if data_hash not in hash_to_data:
        hash_to_data[data_hash] = {
            'csv_times': csv_times,
            'csv_data': csv_data,
            'arrival_times': [],
            'num_waves': 0,
        }
    hash_to_data[data_hash]['arrival_times'].append(arrival_time_rel)
    hash_to_data[data_hash]['num_waves'] = len(hash_to_data[data_hash]['arrival_times'])

# Prepare a list to collect all metadata records
metadata_records = []

# Function to process data augmentation for each augmentation index
def process_augmentation(args):
    (aug_index, data_hash, info, dt, fs, output_directory, original_plot_filepath, original_stft_plot_filepath) = args

    csv_times = info['csv_times']
    csv_data = info['csv_data']
    arrival_times = info['arrival_times']

    # Data augmentation
    total_length = len(csv_times)
    L = total_length // 20  # Length of normal segments
    extended_L = L * 2  # Length of arrival time segments

    segments = []
    used_indices = np.zeros(total_length, dtype=bool)  # Track which data points have been assigned

    # First handle segments near arrival times
    for arrival_time in arrival_times:
        arrival_idx = np.searchsorted(csv_times, arrival_time)

        # Determine start and end indices, ensure they are within bounds
        start_idx = max(arrival_idx - extended_L // 2, 0)
        end_idx = min(arrival_idx + extended_L // 2, total_length)

        # Mark these indices as used
        used_indices[start_idx:end_idx] = True

        segment_times = csv_times[start_idx:end_idx]
        segment_data = csv_data[start_idx:end_idx]
        segment_arrival_time_offsets = [arrival_time - segment_times[0]]

        segment = {
            'times': segment_times,
            'data': segment_data,
            'arrival_time_offsets': segment_arrival_time_offsets,
            'length': len(segment_times)
        }
        segments.append(segment)

    # Handle remaining data, divide into normal segments
    remaining_indices = np.where(~used_indices)[0]

    # If remaining data is insufficient to form a segment, skip
    if len(remaining_indices) < L:
        print(f"Data hash {data_hash} has insufficient remaining data. Skipping augmentation.")
        return None  # Skip this augmentation

    # Divide remaining data into normal segments
    segment_starts = np.arange(0, len(remaining_indices), L)
    for i in range(len(segment_starts)):
        seg_start = segment_starts[i]
        seg_end = segment_starts[i] + L
        if seg_end > len(remaining_indices):
            seg_end = len(remaining_indices)

        idx_range = remaining_indices[seg_start:seg_end]
        segment_times = csv_times[idx_range]
        segment_data = csv_data[idx_range]

        segment = {
            'times': segment_times,
            'data': segment_data,
            'arrival_time_offsets': [],
            'length': len(segment_times)
        }
        segments.append(segment)

    # Randomly shuffle all segments
    np.random.shuffle(segments)

    # Reconstruct the augmented waveform data and time axis
    augmented_data = []
    augmented_times = []
    new_arrival_times = []
    cumulative_time = 0

    for segment in segments:
        segment_length = segment['length']
        # Re-generate the time axis for the segment to ensure time continuity
        adjusted_times = np.arange(segment_length) * dt + cumulative_time
        augmented_times.append(adjusted_times)
        augmented_data.append(segment['data'])

        # Compute new arrival times
        for arrival_time_offset in segment['arrival_time_offsets']:
            new_arrival_time = cumulative_time + arrival_time_offset
            new_arrival_times.append(new_arrival_time)

        cumulative_time += segment_length * dt

    # Concatenate segment data and time axis
    augmented_times = np.concatenate(augmented_times)
    augmented_data = np.concatenate(augmented_data)

    # Plot and save augmented waveform, without axes and labels
    augmented_plot_filename = f'{data_hash}_augmented_{aug_index+1}.png'
    augmented_plot_filepath = os.path.join(output_directory, augmented_plot_filename)
    plt.figure(figsize=(12, 4))
    plt.plot(augmented_times, augmented_data)
    plt.axis('off')  # Remove axes and labels
    plt.savefig(augmented_plot_filepath, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Generate and save STFT plot
    f_stft_aug, t_stft_aug, Zxx_aug = stft(augmented_data, fs=fs, nperseg=256)
    augmented_stft_plot_filename = f'{data_hash}_augmented_stft_{aug_index+1}.png'
    augmented_stft_plot_filepath = os.path.join(output_directory, augmented_stft_plot_filename)
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t_stft_aug, f_stft_aug, np.abs(Zxx_aug), shading='gouraud')
    plt.axis('off')  # Remove axes and labels
    plt.savefig(augmented_stft_plot_filepath, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Prepare metadata record
    metadata_record = {
        'data_hash': data_hash,
        'augmentation_index': aug_index + 1,
        'original_plot_file': original_plot_filepath,
        'original_stft_plot_file': original_stft_plot_filepath,
        'augmented_plot_file': augmented_plot_filepath,
        'augmented_stft_plot_file': augmented_stft_plot_filepath,
        'num_waves': info['num_waves'],
        'original_arrival_times': ','.join(map(str, arrival_times)),
        'new_arrival_times': ','.join(map(str, new_arrival_times)),
    }
    return metadata_record

if __name__ == '__main__':
    # Iterate over the hash dictionary, perform data augmentation, plot, and save image files
    for data_hash, info in hash_to_data.items():
        print(f"Processing data for data_hash: {data_hash}")

        csv_times = info['csv_times']
        csv_data = info['csv_data']
        arrival_times = info['arrival_times']

        # Plot and save the original waveform (only once)
        original_plot_filename = f'{data_hash}_original.png'
        original_plot_filepath = os.path.join(output_directory, original_plot_filename)
        if not os.path.exists(original_plot_filepath):
            plt.figure(figsize=(12, 4))
            plt.plot(csv_times, csv_data)
            plt.axis('off')  # Remove axes and labels
            plt.savefig(original_plot_filepath, bbox_inches='tight', pad_inches=0)
            plt.close()

        # Generate and save the STFT plot of the original waveform (only once)
        original_stft_plot_filename = f'{data_hash}_original_stft.png'
        original_stft_plot_filepath = os.path.join(output_directory, original_stft_plot_filename)
        if not os.path.exists(original_stft_plot_filepath):
            dt = np.mean(np.diff(csv_times))  # Assuming constant sampling interval
            fs = 1 / dt  # Sampling frequency
            f_stft_orig, t_stft_orig, Zxx_orig = stft(csv_data, fs=fs, nperseg=256)
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(t_stft_orig, f_stft_orig, np.abs(Zxx_orig), shading='gouraud')
            plt.axis('off')  # Remove axes and labels
            plt.savefig(original_stft_plot_filepath, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            # If the original STFT plot exists, compute dt and fs for consistency
            dt = np.mean(np.diff(csv_times))  # Assuming constant sampling interval
            fs = 1 / dt  # Sampling frequency

        # Prepare arguments for augmentations
        args_list = []
        for aug_index in range(2):
            args = (
                aug_index,
                data_hash,
                info,
                dt,
                fs,
                output_directory,
                original_plot_filepath,
                original_stft_plot_filepath
            )
            args_list.append(args)

        # Use multiprocessing.Pool to process augmentations in parallel
        with Pool() as pool:
            results = pool.map(process_augmentation, args_list)

        # Filter out None results (in case augmentation was skipped)
        results = [res for res in results if res is not None]

        # Collect metadata records
        metadata_records.extend(results)

    # Convert all metadata into a DataFrame
    df_metadata = pd.DataFrame(metadata_records)

    # Save all metadata to a CSV file
    metadata_csv_file = os.path.join(output_directory, 'metadata.csv')
    df_metadata.to_csv(metadata_csv_file, index=False)
    print(f"All metadata has been saved to {metadata_csv_file}")
