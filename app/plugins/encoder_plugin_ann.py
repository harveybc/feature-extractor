import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Dropout, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2

class Plugin:
    """
    An encoder plugin using a simple neural network based on Keras, with dynamically configurable size.
    """

    plugin_params = {
        'intermediate_layers': 2,
        'learning_rate': 0.000001,
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
        input_shape = input_dim
        interface_size = encoding_dim
        # Calculate the sizes of the intermediate layers
        num_intermediate_layers = self.params['intermediate_layers']
        layers = [input_shape]
        step_size = (input_shape - interface_size) / (num_intermediate_layers + 1)
        
        for i in range(1, num_intermediate_layers + 1):
            layer_size = input_shape - i * step_size
            layers.append(int(layer_size))

        layers.append(interface_size)
        # Debugging message
        print(f"Encoder Layer sizes: {layers}")
        # Encoder: set input layer
        inputs = Input(shape=(input_dim,), name="encoder_input")
        x = inputs

        # add dense and dropout layers
        layers_index = 0
        for size in layers[1:-1]:
            layers_index += 1
            # add the conv and maxpooling layers
            x = Dense(size, activation=LeakyReLU(alpha=0.1), kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001), name="encoder_intermediate_layer" + str(layers_index))(x)
            # add batch normalization layer
            x = BatchNormalization(name="encoder_batch_norm" + str(layers_index))(x)



        # Encoder: set output layer        
        x = Dense(encoding_dim, activation=LeakyReLU(alpha=0.1), kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001), name="encoder_last_dense_layer" )(x)
        # Encoder: Last batch normalization layer
        outputs = BatchNormalization(name="encoder_last_batch_norm")(x)



        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder_ANN")


        # Define the Adam optimizer with custom parameters
        adam_optimizer = Adam(
            learning_rate= self.params['learning_rate'],   # Set the learning rate
            beta_1=0.9,            # Default value
            beta_2=0.999,          # Default value
            epsilon=1e-7,          # Default value
            amsgrad=False
        )

        self.encoder_model.compile(optimizer=adam_optimizer, loss='mae')



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
