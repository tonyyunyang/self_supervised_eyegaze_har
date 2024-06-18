import os
import h5py
import json

import numpy as np


def process_data(window_length, overlap_ratio, raw_data_path, prepared_data_path, file_name, standardize=False):
    training_data = []
    labels = []
    starting_indices = {}
    current_subject = None
    current_index = 0

    mean = None
    std = None

    files = sorted([file for file in os.listdir(raw_data_path) if file.endswith(".csv")])

    if standardize:
        all_data = []
        for file in files:
            file_path = os.path.join(raw_data_path, file)
            with open(file_path, "r") as f:
                for line in f:
                    x, y = map(float, line.strip().split(","))
                    all_data.append([x, y])
        all_data = np.array(all_data)
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)
        print(f"Mean: {mean}, Std: {std}")

    for file in files:
        file_path = os.path.join(raw_data_path, file)
        subject, activity = extract_subject_activity(file)

        with open(file_path, "r") as f:
            data = []
            for line in f:
                x, y = map(float, line.strip().split(","))
                if standardize:
                    x = (x - mean[0]) / std[0]
                    y = (y - mean[1]) / std[1]
                data.append([x, y])

            windows = create_windows(data, window_length, overlap_ratio)
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
