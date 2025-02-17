def configure_size(self, interface_size, input_shape, num_channels, encoder_output_shape, use_sliding_windows):
    # Ensure use_sliding_windows is a proper Boolean.
    use_sliding_windows = str(use_sliding_windows).lower() == 'true'
    self.params['interface_size'] = interface_size
    self.params['input_shape'] = input_shape

    # Retrieve architecture parameters
    intermediate_layers = self.params.get('intermediate_layers', 3)
    initial_layer_size = self.params.get('initial_layer_size', 128)
    layer_size_divisor = self.params.get('layer_size_divisor', 2)
    l2_reg = self.params.get('l2_reg', 1e-5)
    learning_rate = self.params.get('learning_rate', 0.00002)

    # Recompute encoder's layer sizes using the same logic as in the encoder.
    encoder_layers = []
    current_size = initial_layer_size
    for i in range(intermediate_layers):
        encoder_layers.append(current_size)
        current_size = max(current_size // layer_size_divisor, 1)
    encoder_layers.append(interface_size)

    # For the decoder, mirror the encoder's intermediate layers (excluding the latent layer).
    decoder_intermediate_layers = encoder_layers[:-1][::-1]
    print(f"[configure_size] Encoder layer sizes: {encoder_layers}")
    print(f"[configure_size] Decoder intermediate layer sizes: {decoder_intermediate_layers}")
    print(f"[configure_size] Original input shape: {input_shape}")

    # Determine final output units: if sliding windows are used and input_shape is a tuple,
    # the final output units equals the product of its dimensions; otherwise, it's input_shape.
    if use_sliding_windows and isinstance(input_shape, tuple):
        final_output_units = int(np.prod(input_shape))
    else:
        final_output_units = input_shape

    # Extract sequence length and number of filters from encoder_output_shape.
    sequence_length, num_filters = encoder_output_shape
    print(f"[configure_size] Extracted sequence_length={sequence_length}, num_filters={num_filters}")

    # Build the decoder model using Conv1DTranspose layers.
    from keras.models import Sequential
    self.model = Sequential(name="decoder_cnn")
    # First Conv1DTranspose layer: input shape must match the encoder's output shape.
    self.model.add(Conv1DTranspose(
        filters=decoder_intermediate_layers[0],
        kernel_size=3,
        strides=1,
        activation='tanh',
        kernel_initializer=GlorotUniform(),
        kernel_regularizer=l2(l2_reg),
        padding='same',
        input_shape=(sequence_length, num_filters)
    ))
    # Add subsequent Conv1DTranspose layers
    layer_sizes = decoder_intermediate_layers[1:]
    for idx, size in enumerate(layer_sizes, start=1):
        # Use stride 2 for all but the final layer
        strides = 2 if idx < len(layer_sizes) else 1
        self.model.add(Conv1DTranspose(
            filters=size,
            kernel_size=3,
            strides=strides,
            padding='same',
            activation='tanh',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg)
        ))
    # Final projection: if sliding windows are used, output shape equals final_output_units; otherwise, flatten and use Dense.
    if use_sliding_windows:
        self.model.add(Conv1DTranspose(
            filters=num_channels,
            kernel_size=3,
            padding='same',
            activation='tanh',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="decoder_output"
        ))
    else:
        from keras.layers import Flatten
        self.model.add(Flatten(name="decoder_flatten"))
        self.model.add(Dense(
            units=final_output_units,
            activation='linear',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="decoder_dense_output"
        ))
    final_output_shape = self.model.output_shape
    print(f"[configure_size] Final Output Shape: {final_output_shape}")

    adam_optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False
    )
    self.model.compile(
        optimizer=adam_optimizer,
        loss='mean_squared_error'
    )
    print("[configure_size] Decoder Model Summary:")
    self.model.summary()
