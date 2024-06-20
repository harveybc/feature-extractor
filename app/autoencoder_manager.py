class AutoencoderManager:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder_model = None
        self.encoder_model = None
        self.decoder_model = None

    def build_autoencoder(self):
        print("Building encoder...")
        encoder_input = Input(shape=(self.input_dim,), name="encoder_input")
        encoder_output = Dense(self.encoding_dim, activation='relu', name="encoder_output")(encoder_input)
        self.encoder_model = Model(inputs=encoder_input, outputs=encoder_output, name="encoder")
        print(f"Encoder model built: {self.encoder_model}")

        print("Building decoder...")
        decoder_input = Input(shape=(self.encoding_dim,), name="decoder_input")
        decoder_output = Dense(self.input_dim, activation='tanh', name="decoder_output")(decoder_input)
        self.decoder_model = Model(inputs=decoder_input, outputs=decoder_output, name="decoder")
        print(f"Decoder model built: {self.decoder_model}")

        print("Building autoencoder...")
        autoencoder_output = self.decoder_model(encoder_output)
        self.autoencoder_model = Model(inputs=encoder_input, outputs=autoencoder_output, name="autoencoder")
        self.autoencoder_model.compile(optimizer=Adam(), loss='mean_squared_error')
        print(f"Autoencoder model built: {self.autoencoder_model}")

        # Debugging messages to trace the model configuration
        print("Encoder Model Summary:")
        self.encoder_model.summary()
        print("Decoder Model Summary:")
        self.decoder_model.summary()
        print("Full Autoencoder Model Summary:")
        self.autoencoder_model.summary()
