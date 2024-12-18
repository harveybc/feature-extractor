import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input ,Dropout
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal

from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, LeakyReLU, Reshape

class Plugin:
    """
    An encoder plugin using a convolutional neural network (CNN) based on Keras, with dynamically configurable size.
    """

    plugin_params = {

        'intermediate_layers': 3, 
        'learning_rate': 0.0001,
        'dropout_rate': 0.001,
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

    def configure_size(self, input_shape, interface_size, num_channels):
        print(f"[DEBUG] Starting encoder configuration with input_shape={input_shape}, interface_size={interface_size}, num_channels={num_channels}")
        
        self.params['input_shape'] = input_shape

        # Initialize layers array with input_shape
        layers = [input_shape*2]
        num_intermediate_layers = self.params['intermediate_layers']
        
        # Calculate sizes of intermediate layers based on downscaling by 2
        current_size = input_shape*2
        for i in range(num_intermediate_layers):
            next_size = current_size // 2  # Scale down by half
            if next_size < interface_size:
                next_size = interface_size  # Ensure we don't go below the interface_size
            layers.append(next_size)
            current_size = next_size

        # Append the final layer which is the interface size
        layers.append(interface_size)
        print(f"[DEBUG] Encoder Layer sizes: {layers}")

        # Input layer
        inputs = Input(shape=(input_shape, num_channels))
        x = inputs
        print(f"[DEBUG] Input shape: {x.shape}")

        # Initial Conv1D layer
        x = Conv1D(filters=layers[0], kernel_size=3, strides=1, activation=LeakyReLU(alpha=0.1),
                kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001), padding='same')(x)
        print(f"[DEBUG] After Conv1D (filters={layers[0]}) shape: {x.shape}")
        x = BatchNormalization()(x)
        print(f"[DEBUG] After BatchNormalization shape: {x.shape}")

        # Add intermediate layers with stride=2
        for i, size in enumerate(layers[1:-1]):
            x = Conv1D(filters=size, kernel_size=3, strides=2, activation=LeakyReLU(alpha=0.1),
                    kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001), padding='same')(x)
            print(f"[DEBUG] After Conv1D (filters={size}, strides=2) shape: {x.shape}")
            x = BatchNormalization()(x)
            print(f"[DEBUG] After BatchNormalization shape: {x.shape}")

        # Final Conv1D layer to match the interface size
        x = Conv1D(filters=interface_size, kernel_size=1, strides=1, activation=LeakyReLU(alpha=0.1),
                kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001), padding='same')(x)
        print(f"[DEBUG] After Final Conv1D (interface_size) shape: {x.shape}")

        # Output batch normalization layer
        outputs = BatchNormalization()(x)
        print(f"[DEBUG] Final Output shape: {outputs.shape}")

        # Build the encoder model
        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder")

        # Define the Adam optimizer with custom parameters
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],  # Set the learning rate
            beta_1=0.9,  # Default value
            beta_2=0.999,  # Default value
            epsilon=1e-7,  # Default value
            amsgrad=False  # Default value
        )

        self.encoder_model.compile(optimizer=adam_optimizer, loss='mae')
        print(f"[DEBUG] Encoder model compiled successfully.")


    def train(self, data):
        num_channels = data.shape[-1]  # Get number of channels from the data shape
        input_shape = data.shape[1]  # Get the input sequence length
        interface_size = self.params.get('interface_size', 4)  # Assuming interface size is in params
        
        # Rebuild the model with dynamic channel size
        self.configure_size(input_shape, interface_size, num_channels)
        
        # Now proceed with training
        print(f"Training encoder with data shape: {data.shape}")
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.encoder_model.fit(data, data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1, callbacks=[early_stopping])
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
