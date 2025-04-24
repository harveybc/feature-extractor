# main.py

import sys
import json
import pandas as pd
from app.config_handler import load_config, save_config, remote_load_config, remote_save_config, remote_log
from app.cli import parse_args
from app.data_processor import process_data, run_autoencoder_pipeline, load_and_evaluate_encoder, load_and_evaluate_decoder
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args
from typing import Any, Dict


def main():
    """
    Orquesta la ejecución completa del sistema, incluyendo la optimización (si se configura)
    y la ejecución del pipeline completo (preprocesamiento, entrenamiento, predicción y evaluación).
    """
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()
    cli_args: Dict[str, Any] = vars(args)

    print("Loading default configuration...")
    config: Dict[str, Any] = DEFAULT_VALUES.copy()

    file_config: Dict[str, Any] = {}
    # Carga remota de configuración si se solicita
    if args.remote_load_config:
        try:
            file_config = remote_load_config(args.remote_load_config, args.username, args.password)
            print(f"Loaded remote config: {file_config}")
        except Exception as e:
            print(f"Failed to load remote configuration: {e}")
            sys.exit(1)

    # Carga local de configuración si se solicita
    if args.load_config:
        try:
            file_config = load_config(args.load_config)
            print(f"Loaded local config: {file_config}")
        except Exception as e:
            print(f"Failed to load local configuration: {e}")
            sys.exit(1)

    # Primera fusión de la configuración (sin parámetros específicos de plugins)
    print("Merging configuration with CLI arguments and unknown args (first pass, no plugin params)...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, {}, {}, file_config, cli_args, unknown_args_dict)
    
    # --- CARGA DE PLUGINS ---
    print("Merging configuration with CLI arguments and unknown args without plugin-specific parameters...")
    if config['load_encoder']:
        print("Loading and evaluating encoder...")
        load_and_evaluate_encoder(config)
    elif config['load_decoder']:
        print("Loading and evaluating decoder...")
        load_and_evaluate_decoder(config)
    else:
        encoder_plugin_name = config['encoder_plugin']
        decoder_plugin_name = config['decoder_plugin']

        print(f"Loading encoder plugin: {encoder_plugin_name}")
        encoder_plugin_class, _ = load_plugin('feature_extractor.encoders', encoder_plugin_name)
        print(f"Loading decoder plugin: {decoder_plugin_name}")
        decoder_plugin_class, _ = load_plugin('feature_extractor.decoders', decoder_plugin_name)

        encoder_plugin = encoder_plugin_class()
        decoder_plugin = decoder_plugin_class()

        # Carga del Preprocessor Plugin (para process_data, ventanas deslizantes y STL)
        plugin_name = config.get('preprocessor_plugin', 'stl_preprocessor')
        print(f"Loading Plugin ..{plugin_name}")
        try:
            preprocessor_class, _ = load_plugin('preprocessor.plugins', plugin_name)
            preprocessor_plugin = preprocessor_class()
            preprocessor_plugin.set_params(**config)
        except Exception as e:
            print(f"Failed to load or initialize Preprocessor Plugin: {e}")
            sys.exit(1)


        # fusión de configuración, integrando parámetros específicos de plugin predictor
        print("Merging configuration with CLI arguments and unknown args (second pass, with plugin params)...")
        config = merge_config(config, encoder_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
        # fusión de configuración, integrando parámetros específicos de plugin optimizer
        config = merge_config(config, decoder_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
        # fusión de configuración, integrando parámetros específicos de plugin pipeline
        config = merge_config(config, preprocessor_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
        

        #encoder_plugin.set_params(**config)
        #decoder_plugin.set_params(**config)
        #preprocessor_plugin.set_params(**config)

        print("Processing and running autoencoder pipeline...")
        run_autoencoder_pipeline(config, encoder_plugin, decoder_plugin, preprocessor_plugin)

        if 'save_config' in config:
            if config['save_config'] != None:
                save_config(config, config['save_config'])
                print(f"Configuration saved to {config['save_config']}.")

        if 'remote_save_config' in config:
            if config['remote_save_config'] != None:
                print(f"Remote saving configuration to {config['remote_save_config']}")
                remote_save_config(config, config['remote_save_config'], config['username'], config['password'])
                print(f"Remote configuration saved.")

if __name__ == "__main__":
    main()
