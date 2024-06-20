def train_autoencoder(autoencoder_manager, data, mse_threshold, initial_size, step_size, incremental_search, epochs):
    print(f"[train_autoencoder] Initial size: {initial_size}")
    print(f"[train_autoencoder] Data shape: {data.shape}")
    print(f"[train_autoencoder] MSE threshold: {mse_threshold}")
    print(f"[train_autoencoder] Step size: {step_size}")
    print(f"[train_autoencoder] Incremental search: {incremental_search}")
    print(f"[train_autoencoder] Epochs: {epochs}")

    current_size = initial_size
    current_mse = float('inf')
    print(f"[train_autoencoder] Training autoencoder with initial size {current_size}...")

    while current_size > 0 and ((current_mse > mse_threshold) if not incremental_search else (current_mse < mse_threshold)):
        print(f"[train_autoencoder] Current size: {current_size}")
        autoencoder_manager.build_autoencoder()
        print(f"[train_autoencoder] Built autoencoder for size: {current_size}")

        autoencoder_manager.train_autoencoder(data, epochs=epochs, batch_size=256)
        print(f"[train_autoencoder] Trained autoencoder for size: {current_size}")

        encoded_data = autoencoder_manager.encode_data(data)
        decoded_data = autoencoder_manager.decode_data(encoded_data)
        current_mse = autoencoder_manager.calculate_mse(data, decoded_data)
        print(f"[train_autoencoder] Current MSE: {current_mse} at size: {current_size}")

        if (incremental_search and current_mse >= mse_threshold) or (not incremental_search and current_mse <= mse_threshold):
            print("[train_autoencoder] Desired MSE reached. Stopping training.")
            break

        if incremental_search:
            current_size += step_size
            if current_size >= data.shape[1]:
                break
        else:
            current_size -= step_size

    print(f"[train_autoencoder] Final autoencoder model: {autoencoder_manager.autoencoder_model}")
    return autoencoder_manager

def process_data(config):
    print(f"[process_data] Loading data from: {config['csv_file']}")
    data = load_csv(config['csv_file'], headers=config['headers'])
    print(f"[process_data] Data loaded: {data.shape[0]} rows and {data.shape[1]} columns.")

    if config['force_date']:
        print("[process_data] Forcing date conversion on index.")
        data.index = pd.to_datetime(data.index)

    print(f"[process_data] Data types:\n{data.dtypes}")

    debug_info = {}
    processed_data = {}

    for column in data.columns:
        print(f"[process_data] Processing column: {column}")
        column_data = data[[column]].values.astype(np.float64)
        print(f"[process_data] Applying sliding window of size: {config['window_size']}")
        windowed_data = sliding_window(column_data, config['window_size'])
        windowed_data = windowed_data.squeeze()  # Ensure correct shape for training
        print(f"[process_data] Windowed data shape: {windowed_data.shape}")
        processed_data[column] = windowed_data

    return processed_data, debug_info
