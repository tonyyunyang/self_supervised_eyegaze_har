import os
import h5py
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_data(window_length, overlap_ratio, raw_data_path, prepared_data_path, file_name):
    training_data = []
    labels = []
    starting_indices = {}
    current_subject = None
    current_index = 0

    files = sorted([file for file in os.listdir(raw_data_path) if file.endswith(".csv")])

    # Create a folder for plots if it doesn't exist
    plots_folder = os.path.join(prepared_data_path, "plots")
    os.makedirs(plots_folder, exist_ok=True)

    for file in files:
        file_path = os.path.join(raw_data_path, file)
        subject, activity = extract_subject_activity(file)

        with open(file_path, "r") as f:
            data = []
            for line in f:
                x, y = map(float, line.strip().split(","))
                data.append([x, y])

            # Convert data to DataFrame
            df_original = pd.DataFrame(data, columns=['x', 'y'])

            # Preprocess the data
            df_preprocessed = preprocess_data(df_original.copy())

            # Plot and save the original and preprocessed data
            plot_data(df_original, df_preprocessed, subject, activity, plots_folder)

            # Create windows from preprocessed data
            windows = create_windows(df_preprocessed[['x', 'y']].values.tolist(), window_length, overlap_ratio)

            for window in windows:
                training_data.append(window)
                labels.append([subject, activity])

                if current_subject != subject:
                    starting_indices[subject] = current_index
                    current_subject = subject
                current_index += 1

    with h5py.File(os.path.join(prepared_data_path, file_name), "w") as h5_file:
        h5_file.create_dataset("training_data", data=training_data)
        h5_file.create_dataset("labels", data=labels)

    with open(os.path.join(prepared_data_path, "starting_indices.json"), "w") as json_file:
        json.dump(starting_indices, json_file)


def extract_subject_activity(file_name):
    # Extract subject and activity from the file name
    # Example: "P01_BROWSE.csv" -> ("P01", "BROWSE")
    subject, activity = file_name.split("_")
    activity = activity.split(".")[0]
    return subject, activity


def create_windows(data, window_length, overlap_ratio):
    windows = []
    step_size = int(window_length * (1 - overlap_ratio))

    for i in range(0, len(data) - window_length + 1, step_size):
        if i + window_length <= len(data):
            window = data[i:i + window_length]
            windows.append(window)

    return windows


def preprocess_data(df):
    # # Calculate the derivative
    # df['x_derivative'] = df['x'].diff().fillna(0)
    # df['y_derivative'] = df['y'].diff().fillna(0)
    #
    # # Compute the Median Absolute Deviation (MAD) and filter
    # for col in ['x', 'y']:
    #     median = np.median(df[f'{col}_derivative'])
    #     mad = np.median(np.abs(df[f'{col}_derivative'] - median))
    #     mad_threshold = 3 * mad
    #     mad_mask = ~(np.abs(df[f'{col}_derivative'] - median) > mad_threshold)
    #     valid_indices = np.where(mad_mask)[0]
    #     df[col] = np.interp(np.arange(len(df)), valid_indices, df.loc[mad_mask, col])
    #
    # # Z-score filtering
    # window_size = 30
    # threshold = 0.1
    # for col in ['x', 'y']:
    #     mean = df[col].rolling(window=window_size, center=True).mean()
    #     std = df[col].rolling(window=window_size, center=True).std()
    #     z_score_mask = (df[col] - mean).abs() <= threshold * std
    #     valid_indices = np.where(z_score_mask)[0]
    #     df[col] = np.interp(np.arange(len(df)), valid_indices, df.loc[z_score_mask, col])
    #
    # Standardize the data
    scaler = StandardScaler()
    df[['x', 'y']] = scaler.fit_transform(df[['x', 'y']])

    # Normalize the data
    normalizer = MinMaxScaler()
    df[['x', 'y']] = normalizer.fit_transform(df[['x', 'y']])

    return df


def plot_data(df_original, df_preprocessed, subject, activity, plots_folder):
    plt.figure(figsize=(12, 6))

    # Plot original data with alpha=0.5
    # plt.plot(df_original['x'], df_original['y'], alpha=0.5, label='Original')

    # Plot preprocessed data with full opacity
    plt.plot(df_preprocessed['x'], df_preprocessed['y'], label='Preprocessed')

    plt.title(f"Subject: {subject}, Activity: {activity}")
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(plots_folder, f"{subject}_{activity}.png"))
    plt.close()


def plot_data_timeline(df_original, df_preprocessed, subject, activity, plots_folder):
    plt.figure(figsize=(15, 10))

    # Plot original data
    plt.plot(df_original.index, df_original['x'], alpha=0.5, label='Original X', color='blue')
    plt.plot(df_original.index, df_original['y'], alpha=0.5, label='Original Y', color='red')

    # Plot preprocessed data
    plt.plot(df_preprocessed.index, df_preprocessed['x'], label='Preprocessed X', color='cyan')
    plt.plot(df_preprocessed.index, df_preprocessed['y'], label='Preprocessed Y', color='magenta')

    plt.title(f"Subject: {subject}, Activity: {activity}")
    plt.xlabel('Timeline (Data points)')
    plt.ylabel('Coordinate Values')
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(plots_folder, f"{subject}_{activity}.png"))
    plt.close()