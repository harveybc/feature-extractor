def configure_size(self, interface_size, input_shape, num_channels=None, encoder_output_shape=None, use_sliding_windows=False):
    # Force use_sliding_windows to Boolean even if not used.
    use_sliding_windows = str(use_sliding_windows).lower() == 'true'
    self.params['interface_size'] = interface_size
    self.params['output_shape'] = input_shape  # number of time steps to reconstruct
    if num_channels is None:
        num_channels = 1

    # Compute encoder layer sizes as in the encoder.
    initial_layer_size = self.params.get('initial_layer_size', 32)
    intermediate_layers = self.params.get('intermediate_layers', 3)
    layer_size_divisor = self.params.get('layer_size_divisor', 2)
    encoder_layers = []
    current_size = initial_layer_size
    for i in range(intermediate_layers):
        encoder_layers.append(current_size)
        current_size = max(current_size // layer_size_divisor, 1)
    encoder_layers.append(interface_size)
    print(f"[configure_size] Encoder layer sizes (from encoder): {encoder_layers}")

    # Mirror the encoder LSTM sizes (excluding the latent interface itself).
    decoder_lstm_sizes = list(reversed(encoder_layers[:-1]))
    print(f"[configure_size] Decoder LSTM sizes (mirrored): {decoder_lstm_sizes}")

    # Build the decoder model using Sequential API.
    from keras.models import Sequential
    self.model = Sequential(name="decoder_lstm")
    # First, expand the latent vector via Dense to match the last LSTM unit of the encoder.
    latent_dense_units = encoder_layers[-2]
    self.model.add(Dense(
        latent_dense_units,
        input_shape=(interface_size,),
        activation='relu',
        kernel_initializer=HeNormal(),
        kernel_regularizer=l2(self.params.get('l2_reg', 1e-2)),
        name="decoder_dense_expand"
    ))
    # Optional dropout
    dropout_rate = self.params.get('dropout_rate', 0.1)
    if dropout_rate > 0:
        from keras.layers import Dropout
        self.model.add(Dropout(dropout_rate, name="decoder_dropout_after_dense"))
    # Repeat the vector to form a sequence of length equal to output sequence length.
    from keras.layers import RepeatVector
    self.model.add(RepeatVector(input_shape))
    print(f"[configure_size] Added RepeatVector layer with output length: {input_shape}")
    # Add mirrored LSTM layers.
    for idx, units in enumerate(decoder_lstm_sizes, start=1):
        from keras.layers import LSTM
        self.model.add(LSTM(
            units=units,
            activation='tanh',
            recurrent_activation='sigmoid',
            return_sequences=True,
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(self.params.get('l2_reg', 1e-2)),
            name=f"decoder_lstm_{idx}"
        ))
        if dropout_rate > 0:
            from keras.layers import Dropout
            self.model.add(Dropout(dropout_rate, name=f"decoder_dropout_after_lstm_{idx}"))
    # Final TimeDistributed Dense layer to reconstruct the original features.
    from keras.layers import TimeDistributed, Dense
    self.model.add(TimeDistributed(
        Dense(num_channels,
              activation='linear',
              kernel_initializer=GlorotUniform(),
              kernel_regularizer=l2(self.params.get('l2_reg', 1e-2))),
        name="decoder_output"
    ))
    self.model.compile(
        optimizer=Adam(learning_rate=self.params.get('learning_rate', 0.0001),
                       beta_1=0.9, beta_2=0.999, epsilon=1e-7),
        loss='mean_squared_error'
    )
    print("[configure_size] Decoder Model Summary:")
    self.model.summary()
