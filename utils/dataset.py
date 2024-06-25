import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def extract_parameters_from_datatype(data_type):
    # Split the data_type string by underscore
    parts = data_type.split('_')

    # Extract the overlap value
    overlap = float(parts[1])

    # Extract the window value and convert it to seconds
    window_value = parts[3]
    if window_value.endswith('s'):
        window_seconds = int(window_value[:-1])
    else:
        raise ValueError("Invalid window format in data_type")

    # Calculate the window length in samples
    window_length = int(window_seconds * 30)

    return overlap, window_seconds, window_length


class FullySupervisedDataset(Dataset):
    def __init__(self, file_path, indices, label_map):
        self.file = h5py.File(file_path, 'r')
        self.indices = indices

        # Invert the label map so we can look up by string label
        self.label_map = {v: k for k, v in label_map.items()}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        # Extract the index from the indices list
        index = self.indices[item]

        # Load the data from the file
        data = torch.from_numpy(self.file['training_data'][index]).float()

        label_str = self.file['labels'][index][1].decode('utf-8')

        label_int = torch.tensor(self.label_map[label_str])

        # print(f"label: {label_str} decoded as {label_int}")

        return data, label_int


class SelfSupervisedDataset(Dataset):
    def __init__(self, file_path, indices, mean_mask_length=5, masking_ratio=0.20):
        self.file = h5py.File(file_path, 'r')
        self.indices = indices
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        # Extract the index from the indices list
        index = self.indices[item]

        # Load the data from the file
        data = np.array(self.file['training_data'][index])

        # Generate noise mask for the data
        mask = noise_mask(data, self.masking_ratio, self.mean_mask_length)

        # Apply the mask to the data: 0s in mask mean masked (set to zero) in data
        masked_data = data.copy()  # Make a copy to avoid modifying the original data
        masked_data[mask == 0] = 0.0  # Apply mask

        # Convert data, mask, and masked_data to PyTorch tensors
        original_input = torch.from_numpy(data).float()
        mask = torch.from_numpy(mask).bool()
        masked_input = torch.from_numpy(masked_data).float()

        # Return the original data, the mask, the masked data, and the indices
        return original_input, mask, masked_input, index


def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask

def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def check_indices_overlap(train_indices, val_indices, test_indices):
    # Convert lists to sets
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)

    # Check for overlaps
    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)

    # Print results
    if train_val_overlap:
        print(f"Overlap between train and validation indices: {train_val_overlap}")
    else:
        print("No overlap between train and validation indices.")

    if train_test_overlap:
        print(f"Overlap between train and test indices: {train_test_overlap}")
    else:
        print("No overlap between train and test indices.")

    if val_test_overlap:
        print(f"Overlap between validation and test indices: {val_test_overlap}")
    else:
        print("No overlap between validation and test indices.")

    # Return boolean for no overlap condition
    return not train_val_overlap and not train_test_overlap and not val_test_overlap