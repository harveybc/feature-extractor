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
        'learning_rate': 0.00008,
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

        # Set input layer
        inputs = Input(shape=(input_shape,1))
        x = inputs
        print(f"Input shape: {x.shape}")
        # flatten the input
        x = Flatten()(x)
        print(f"Flattened shape: {x.shape}")
        # add the first dense layer of size input_shape
        x = Dense(input_shape, input_shape=(1,input_shape), activation='tanh', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.001))(x)
        print(f"First dense layer shape: {x.shape}")
        x = BatchNormalization()(x)
        print(f"Batch Normalization shape: {x.shape}")
        # Perform reshaping if needed to adjust for the expected input shape
        x = Reshape((input_shape,1))(x)
        print(f"Reshaped shape: {x.shape}")
        # Add Conv1D and MaxPooling1D layers, using channels as features
        layers_index = 0
        for size in layers[1:-1]:  # Use the layers except the first and the last one
            layers_index += 1
            
            # Calculate pool size
            if layers_index >= len(layers):
                pool_size = round(size / interface_size)
            else:
                pool_size = round(size / layers[layers_index])
            
            # Configure kernel size based on the layer's size
            kernel_size = 3
            if size > 64:
                kernel_size = 5
            if size > 512:
                kernel_size = 7
            # if the size is half of less of the current sequence lengthn (second componenty of the output shape of the last layer) use trides of 2
            last_shape = x.shape
            # Extract the sequence length from the output shape
            sequence_length = int(last_shape[1])  # This is the sequence length
            if sequence_length >= 2*size:
                strides = 2  # Reduce sequence length
            else:
                strides = 1  # Keep sequence length the same

            # Add Conv1D and BatchNormalization layers
            x = Conv1D(filters=size, kernel_size=kernel_size, strides=strides, activation='tanh', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.001), padding='valid' )(x)
            print(f"After Conv1D (filters={size}): {x.shape}")
            x = BatchNormalization()(x)
            print(f"After BatchNormalization: {x.shape}")
            #x = Dropout(self.params['dropout_rate'])(x)
            
            # Add MaxPooling1D layer if necessary
            #if pool_size < 2:
            #    pool_size = 2
            #x = MaxPooling1D(pool_size=pool_size)(x)

        # Flatten the output to prepare for the Dense layer
        x = Flatten()(x)

        # Add the last dense layer
        outputs = Dense(interface_size, input_shape=(interface_size,), activation='tanh', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.001))(x)
        # Add the output reshape layer
        #outputs = Reshape((1, interface_size))(x)
        # Build the encoder model
        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder")
                # Define the Adam optimizer with custom parameters
        adam_optimizer = Adam(
            learning_rate= self.params['learning_rate'],   # Set the learning rate
            beta_1=0.9,            # Default value
            beta_2=0.999,          # Default value
            epsilon=1e-7,          # Default value
            amsgrad=False          # Default value
        )

        self.encoder_model.compile(optimizer=adam_optimizer, loss='mae')

    def train(self, data):
        print(f"Training encoder with data shape: {data.shape}")
         # early stopping
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
