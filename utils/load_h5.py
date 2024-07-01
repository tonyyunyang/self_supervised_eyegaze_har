import h5py
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, fixed


def explore_h5_file(file_path):
    # Open the .h5 file in read mode
    with h5py.File(file_path, 'r') as file:
        # Function to recursively explore groups and datasets
        def explore_group(group, prefix=""):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    # If the item is a dataset, print its details
                    print(f"{prefix}/{key}:")
                    print(f"  Shape: {item.shape}")
                    print(f"  Dtype: {item.dtype}")
                elif isinstance(item, h5py.Group):
                    # If the item is a group, recursively explore it
                    print(f"{prefix}/{key} (Group)")
                    explore_group(item, prefix=f"{prefix}/{key}")

        # Start exploring from the root group
        explore_group(file)


# def plot_filtered_data(index, dataset):
#     # Clear the current figure
#     plt.clf()
#     # Plot the true input
#     plt.plot(dataset['original_input'][index, :, 0], label='True, input')
#
#     # Masked True input (show these points in a scatter plot)
#     mask = dataset['mask'][index, :, 0] == 0
#     plt.scatter(np.where(mask)[0], dataset['original_input'][index, mask, 0], color='gray', label='True, masked',
#                 alpha=0.6)
#
#     # Prediction only where mask is True (zero in this context)
#     plt.scatter(np.where(mask)[0], dataset['predicted_output'][index, mask, 0], color='orange', label='Prediction',
#                 alpha=0.6)
#
#     # Add labels and legend
#     plt.xlabel('Time Step')
#     plt.ylabel('Value')
#     plt.title(f"Sample Index: {index}")
#     plt.legend()
#     plt.show()
#
#
# def visualize_h5_filtered_data(file_path):
#     # Open the .h5 file
#     with h5py.File(file_path, 'r') as file:
#         # Store datasets in a dictionary
#         data = {
#             'mask': file['/mask'][:],
#             'original_input': file['/original_input'][:],
#             'predicted_output': file['/predicted_output'][:]
#         }
#         # Create interactive plot with slider
#         interact(plot_filtered_data, index=IntSlider(min=0, max=data['original_input'].shape[0] - 1, step=1, value=0),
#                  dataset=fixed(data))


def plot_filtered_data_both_dimensions(index, dataset):
    # Clear the current figure
    plt.clf()

    # Create a figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for i in range(2):  # Loop over each dimension
        # Masked True input (show these points in a scatter plot where mask is 0)
        mask = dataset['mask'][index, :, i] == 0

        # Plot the true input and masked points on subplot
        axes[i].plot(dataset['original_input'][index, :, i], label='True, input')
        axes[i].scatter(np.where(mask)[0], dataset['original_input'][index, mask, i], color='gray',
                        label='True, masked', alpha=0.6)

        # Prediction only where mask is True (zero in this context)
        axes[i].scatter(np.where(mask)[0], dataset['predicted_output'][index, mask, i], color='orange',
                        label='Prediction', alpha=0.6)

        # Set labels and titles for subplots
        if i == 0:
            axes[i].set_title(f'Gaze X Coordinate')
        elif i == 1:
            axes[i].set_title(f'Gaze Y Coordinate')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


def visualize_h5_filtered_data_both_dimensions(file_path):
    # Open the .h5 file
    with h5py.File(file_path, 'r') as file:
        # Store datasets in a dictionary
        data = {
            'mask': file['/mask'][:],
            'original_input': file['/original_input'][:],
            'predicted_output': file['/predicted_output'][:]
        }
        # Create interactive plot with slider
        interact(plot_filtered_data_both_dimensions,
                 index=IntSlider(min=0, max=data['original_input'].shape[0] - 1, step=1, value=0), dataset=fixed(data))