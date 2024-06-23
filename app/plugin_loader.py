from importlib.metadata import entry_points, EntryPoint

def load_plugin(plugin_group, plugin_name):
    print(f"Attempting to load plugin: {plugin_name} from group: {plugin_group}")
    try:
        group_entries = entry_points().get(plugin_group, [])
        entry_point = next(ep for ep in group_entries if ep.name == plugin_name)
        plugin_class = entry_point.load()
        required_params = list(plugin_class.plugin_params.keys())
        print(f"Successfully loaded plugin: {plugin_name} with params: {plugin_class.plugin_params}")
        return plugin_class, required_params
    except StopIteration:
        print(f"Failed to find plugin {plugin_name} in group {plugin_group}")
        raise ImportError(f"Plugin {plugin_name} not found in group {plugin_group}.")
    except Exception as e:
        print(f"Failed to load plugin {plugin_name} from group {plugin_group}, Error: {e}")
        raise

def load_encoder_decoder_plugins(encoder_name, decoder_name):
    encoder_plugin, encoder_params = load_plugin('feature_extractor.encoders', encoder_name)
    decoder_plugin, decoder_params = load_plugin('feature_extractor.decoders', decoder_name)
    return encoder_plugin, encoder_params, decoder_plugin, decoder_params

def get_plugin_params(plugin_group, plugin_name):
    print(f"Getting plugin parameters for: {plugin_name} from group: {plugin_group}")
    try:
        group_entries = entry_points().get(plugin_group, [])
        entry_point = next(ep for ep in group_entries if ep.name == plugin_name)
        plugin_class = entry_point.load()
        print(f"Retrieved plugin params: {plugin_class.plugin_params}")
        return plugin_class.plugin_params
    except StopIteration:
        print(f"Failed to find plugin {plugin_name} in group {plugin_group}")
        raise ImportError(f"Plugin {plugin_name} not found in group {plugin_group}.")
    except Exception as e:
        print(f"Failed to get plugin params for {plugin_name} from group {plugin_group}, Error: {e}")
        raise ImportError(f"Failed to get plugin params for {plugin_name} from group {plugin_group}, Error: {e}")
