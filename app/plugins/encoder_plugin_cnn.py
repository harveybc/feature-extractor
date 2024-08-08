import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input ,Dropout
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal

class Plugin:
    """
    An encoder plugin using a convolutional neural network (CNN) based on Keras, with dynamically configurable size.
    """

    plugin_params = {

        'intermediate_layers': 5, 
        'learning_rate': 0.000001,
        'dropout_rate': 0.1,
    }

    plugin_debug_vars = ['input_shape', 'intermediate_layers']

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

    def configure_size(self, input_shape, interface_size):
        self.params['input_shape'] = input_shape

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

        # set input layer
        inputs = Input(shape=(input_shape, 1))
        x = inputs

        # add conv and maxpooling layers, calculating their kernel and pool sizes
        layers_index = 0
        for size in layers:
            layers_index += 1
            # pool size calculation
            if layers_index >= len(layers):
                pool_size = round(size/interface_size)
            else:
                pool_size = round(size/layers[layers_index])
            # kernel size configuration based on the layer's size
            kernel_size = 3 
            if size > 64:
                kernel_size = 5
            if size > 512:
                kernel_size = 7
            # add the conv and maxpooling layers
            x = Conv1D(filters=size, kernel_size=kernel_size, activation='relu', kernel_initializer=HeNormal(), padding='same')(x)
            if pool_size < 2:
                pool_size = 2
            x = MaxPooling1D(pool_size=pool_size)(x)
            x = Dropout(self.params['dropout_rate'])(x) 

        x = Flatten()(x)
        
        outputs = Dense(interface_size, activation='tanh', kernel_initializer=GlorotUniform())(x)
        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder")

                # Define the Adam optimizer with custom parameters
        adam_optimizer = Adam(
            learning_rate= self.params['learning_rate'],   # Set the learning rate
            beta_1=0.9,            # Default value
            beta_2=0.999,          # Default value
            epsilon=1e-7,          # Default value
            amsgrad=False          # Default value
        )

        self.encoder_model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

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
    plugin.configure_size(input_shape=128, interface_size=4)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
