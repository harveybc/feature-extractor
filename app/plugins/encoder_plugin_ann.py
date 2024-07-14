import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Dropout
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal

class Plugin:
    """
    An encoder plugin using a simple neural network based on Keras, with dynamically configurable size.
    """

    plugin_params = {
        'dropout_rate': 0.1,
        'intermediate_layers': 1,
        'layer_size_divisor': 2,
        'learning_rate': 0.01,
        'activation': 'tanh'
    }

    plugin_debug_vars = ['input_dim', 'encoding_dim']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.encoder_model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_size(self, input_dim, encoding_dim):
        self.params['input_dim'] = input_dim
        self.params['encoding_dim'] = encoding_dim

        layers = []
        current_size = input_dim
        layer_size_divisor = self.params['layer_size_divisor'] 
        current_location = input_dim
        int_layers = 0
        while (current_size > encoding_dim) and (int_layers < (self.params['intermediate_layers']+1)):
            layers.append(current_location)
            current_size = max(current_size // layer_size_divisor, encoding_dim)
            current_location = encoding_dim + current_size
            int_layers += 1
        layers.append(encoding_dim)
        # Debugging message
        print(f"Encoder Layer sizes: {layers}")

        # Encoder: set input layer
        inputs = Input(shape=(input_dim,), name="encoder_input")
        x = inputs

        # add dense and dropout layers
        layers_index = 0
        for size in layers:
            layers_index += 1
            # add the conv and maxpooling layers
            x = Dense(encoding_dim, activation='relu', kernel_initializer=HeNormal(), name="encoder_intermediate_layer" + str(layers_index))(x)
            # add dropout layer
            x = Dropout(self.params['dropout_rate'])(x)

        # Encoder: set output layer        
        outputs = Dense(encoding_dim, activation=self.params['activation'], kernel_initializer=GlorotUniform(), name="encoder_output" )(x)
        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder_ANN")


        # Define the Adam optimizer with custom parameters
        adam_optimizer = Adam(
            learning_rate= self.params['learning_rate'],   # Set the learning rate
            beta_1=0.9,            # Default value
            beta_2=0.999,          # Default value
            epsilon=1e-7,          # Default value
            amsgrad=False          # Default value
        )

        self.encoder_model.compile(optimizer=adam_optimizer, loss='mean_squared_error')



        # Debugging messages to trace the model configuration
        print("Encoder Model Summary:")
        self.encoder_model.summary()

    def train(self, data):
        print(f"Training encoder with data shape: {data.shape}")
        self.encoder_model.fit(data, data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1)
        print("Training completed.")

    def encode(self, data):
        print(f"Encoding data with shape: {data.shape}")
        encoded_data = self.encoder_model.predict(data)
        print(f"Encoded data shape: {encoded_data.shape}")
        return encoded_data

    def save(self, file_path):
        save_model(self.encoder_model, file_path)
        print(f"Encoder model saved to {file_path}")

    def load(self, file_path):
        self.encoder_model = load_model(file_path)
        print(f"Encoder model loaded from {file_path}")

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(input_dim=128, encoding_dim=4)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
