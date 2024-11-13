import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, UpSampling1D, Reshape, Flatten, Conv1DTranspose,Dropout
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal

from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, MaxPooling1D, Cropping1D, LeakyReLU,Input
import math

class Plugin:
    plugin_params = {
        'intermediate_layers': 3, 
        'learning_rate': 0.00008,
        'dropout_rate': 0.001,
    }

    plugin_debug_vars = ['interface_size', 'output_shape', 'intermediate_layers']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value 

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_size(self, interface_size, output_shape):
        self.params['interface_size'] = interface_size
        self.params['output_shape'] = output_shape

        layer_sizes = []

        # Calculate the sizes of the intermediate layers
        num_intermediate_layers = self.params['intermediate_layers']
        layers = [output_shape]
        step_size = (output_shape - interface_size) / (num_intermediate_layers + 1)
        
        for i in range(1, num_intermediate_layers + 1):
            layer_size = output_shape - i * step_size
            layers.append(int(layer_size))

        layers.append(interface_size)

        # For the decoder, reverses the order of the generted layers.
        layer_sizes=layers
        layer_sizes.reverse()
        
        # Debugging message
        print(f"Decoder Layer sizes: {layer_sizes}")
       




        self.model = Sequential(name="decoder")

        # Step 1: Add the Dense layer corresponding to the encoder's output layer
        self.model.add(Dense(interface_size, 
                        activation='tanh', 
                        kernel_initializer=GlorotUniform(), 
                        kernel_regularizer=l2(0.001),
                        input_shape=(interface_size,)))  # input_shape must match the encoder's output size

        # Step 2: Reshape the Dense output to (sequence_length, 1)
        # Since we don't have sequence length or channels, treat the entire interface_size as sequence_length with 1 channel
        sequence_length = interface_size  # Treat interface_size as sequence length
        num_channels = 1  # Single channel, as no other information is available

        # Add the Reshape layer
        self.model.add(Reshape((sequence_length, num_channels)))

        # Step 3: Apply BatchNormalization
        self.model.add(BatchNormalization())

        for size in layer_sizes[1:-1]:


            kernel_size = 3 if size <= 64 else 5 if size <= 512 else 7
            
            last_shape =self.model.layers[-1].output_shape
            sequence_length = int(last_shape[1])  # This is the sequence length
            if sequence_length <= 2*size:
                strides = 2  # Reduce sequence length
            else:
                strides = 1  # Keep sequence length the same

            self.model.add(Conv1DTranspose(filters=size, kernel_size=kernel_size, strides=strides, padding='valid', activation='tanh', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.001)))
            print(f"After Conv1DTranspose (filters={size}): {self.model.layers[-1].output_shape}")
            self.model.add(BatchNormalization())
            print(f"After BatchNormalization: {self.model.layers[-1].output_shape}")
            #self.model.add(Dropout(self.params['dropout_rate'] / 2))
            #print(f"After Dropout: {self.model.layers[-1].output_shape}")

        # add a flatten layer
        self.model.add(Flatten())
        print(f"After Flatten: {self.model.layers[-1].output_shape}")
        # 2. UpSampling1D as the inverse of MaxPooling1D in the encoder
        kernel_size = 3 if output_shape <= 64 else 5 if output_shape <= 512 else 7
        #self.model.add(Conv1DTranspose(filters=output_shape, kernel_size=kernel_size, padding='valid', activation=LeakyReLU(alpha=0.1), kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001), name="decoder_output"))
        #print(f"After Final Conv1DTranspose: {self.model.layers[-1].output_shape}")
        #Final Dense lLayer
        self.model.add(Dense(output_shape, activation='tanh', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.001)))
        print(f"After Final Dense: {self.model.layers[-1].output_shape}")

        # 5. Reshape the output to ensure the final output is (None, output_shape, 1)
        self.model.add(Reshape((output_shape, 1)))
        print(f"Final Output Shape: {self.model.layers[-1].output_shape}")







                # Define the Adam optimizer with custom parameters
        adam_optimizer = Adam(
            learning_rate= self.params['learning_rate'],   # Set the learning rate
            beta_1=0.9,            # Default value
            beta_2=0.999,          # Default value
            epsilon=1e-7,          # Default value
            amsgrad=False          # Default value
        )

        self.model.compile(optimizer=adam_optimizer, loss='mae')

    def train(self, encoded_data, original_data):
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))
        original_data = original_data.reshape((original_data.shape[0], -1))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(encoded_data, original_data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1, callbacks=[early_stopping])

    def decode(self, encoded_data):
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))
        decoded_data = self.model.predict(encoded_data)
        decoded_data = decoded_data.reshape((decoded_data.shape[0], -1))
        return decoded_data

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        self.model = load_model(file_path)

    def calculate_mse(self, original_data, reconstructed_data):
        original_data = original_data.reshape((original_data.shape[0], -1))
        reconstructed_data = reconstructed_data.reshape((original_data.shape[0], -1))
        mse = np.mean(np.square(original_data - reconstructed_data))
        return mse

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(interface_size=4, output_shape=128)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
