from app.data_handler import load_csv, write_csv

def process_data(config, plugin):
    """Process the data using the specified plugin."""
    data = load_csv(config['csv_file'], headers=config['headers'])

    # Debugging: Print loaded data
    print("Loaded data:\n", data.head())

    processed_data = plugin.process(data, method=config['method'], save_params=config['save_config'], load_params=config['load_config'], single=config['single'], multi=config['multi'])

    # Debugging: Print processed data
    print("Processed data:\n", processed_data.head())

    include_date = config['force_date'] or not (config['method'] in ['select_single', 'select_multi'])

    if not config['quiet_mode']:
        print("Processing complete. Writing output...")

    write_csv(config['output_file'], processed_data, include_date=include_date, headers=config['headers'])

    if not config['quiet_mode']:
        print(f"Output written to {config['output_file']}")
    return processed_data
