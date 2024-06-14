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
