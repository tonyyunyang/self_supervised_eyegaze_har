import h5py
import numpy as np
from collections import defaultdict

from sklearn.model_selection import train_test_split


def get_self_supervised_pretrain_indices(subjects, leave_out_subject, last_index):
    pretrain_test_indices = []
    pretrain_train_indices = []

    # Ensure subjects are sorted by their starting indices to maintain order
    sorted_subjects = sorted(subjects.items(), key=lambda x: x[1])

    # Find the indices for the leave-out subject
    for i, (subject, start) in enumerate(sorted_subjects):
        # Calculate the end index based on whether it's the last subject in the list
        end = sorted_subjects[i + 1][1] if i + 1 < len(sorted_subjects) else last_index + 1

        if subject == leave_out_subject:
            pretrain_test_indices.extend(range(start, end))
        else:
            pretrain_train_indices.extend(range(start, end))

    return pretrain_test_indices, pretrain_train_indices


def split_leave_out_rest_sub_sample_indices(subjects, leave_out_subject, last_index):
    leave_out_sub_samples = []
    rest_sub_samples = []

    # Ensure subjects are sorted by their starting indices to maintain order
    sorted_subjects = sorted(subjects.items(), key=lambda x: x[1])

    # Find the indices for the leave-out subject
    for i, (subject, start) in enumerate(sorted_subjects):
        # Calculate the end index based on whether it's the last subject in the list
        end = sorted_subjects[i + 1][1] if i + 1 < len(sorted_subjects) else last_index + 1

        if subject == leave_out_subject:
            leave_out_sub_samples.extend(range(start, end))
        else:
            rest_sub_samples.extend(range(start, end))

    return leave_out_sub_samples, rest_sub_samples


def get_fully_supervised_finetune_indices(finetune_subject_indices, data_file_path, finetune_proportion=0.):
    # Sort the indices to ensure they're in increasing order
    finetune_subject_indices = np.sort(finetune_subject_indices)

    # Open the HDF5 file and retrieve the labels
    with h5py.File(data_file_path, 'r') as hf:
        all_labels = hf['labels'][:]
        finetune_labels = all_labels[finetune_subject_indices, 1]

    # Identify unique labels
    unique_labels = np.unique(finetune_labels)
    # print(f"Unique labels: {unique_labels}")

    # Create dictionaries to store indices for each label
    label_indices = defaultdict(list)
    for idx, label in zip(finetune_subject_indices, finetune_labels):
        label_indices[label].append(idx)

    finetune_train_indices = []
    finetune_test_indices = []

    print("Distribution of indices across labels for train set:")

    # Split indices for each label based on finetune_proportion
    for label in unique_labels:
        indices = np.array(label_indices[label])
        n_train = int(len(indices) * finetune_proportion)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        finetune_train_indices.extend(train_indices)
        finetune_test_indices.extend(test_indices)

        print(f"  Label {label}: {len(train_indices)} indices")

    # Convert lists to numpy arrays and sort them
    finetune_train_indices = np.sort(finetune_train_indices)
    finetune_test_indices = np.sort(finetune_test_indices)

    print(f"Total train indices: {len(finetune_train_indices)}")
    print(f"Total test indices: {len(finetune_test_indices)}")

    return finetune_test_indices, finetune_train_indices


def safe_train_test_split(data, train_size, random_state=None):
    if train_size == 1.0:
        return data, []
    elif train_size == 0.0:
        return [], data
    else:
        return train_test_split(data, train_size=train_size, random_state=random_state)