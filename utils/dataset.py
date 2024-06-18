import h5py
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
    def __init__(self, file_path, indices):
        self.file = h5py.File(file_path, 'r')
        self.indices = indices

    def __getitem__(self, item):
        # Extract the index from the indices list
        index = self.indices[item]

        # Load the data from the file
        data = self.file['training_data'][index]

        return data