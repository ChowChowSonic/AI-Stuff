import keras
import os
import pickle
import tensorflow as tf
import numpy as np
import audioread
import keras.backend as kerasUtilities
from keras import layers
from tensorflow.keras.optimizers import Adam

class MidiGenerator:
    def __init__(self, input_shape, latent_space_dim,ltsm_neurons=64):
        self.isCompiled = False
        self.input_shape = input_shape # [28, 28, 1]
        self.latent_space_dim = latent_space_dim # 2
        self.ltsm_neurons = ltsm_neurons
        self.reconstruction_loss_weight = 1000000
        self._model_input = None
        self._model_output = None
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self._model_output = self.decoder(self.encoder(self._model_input)) 
            # Don't even ask what's going on with the line above this, I don't know either.
        self.model = keras.Model(self._model_input, self._model_output, name="MusicGenerator")

        self._shape_before_bottleneck = None
        self._ltsm_state = None
        
        # Beginning of encoder stuff
    def _build_encoder(self):
        input_layer = keras.Input(shape = self.input_shape, name="Input_layer")
        embedding = layers.Embedding(np.prod(self.input_shape), self.latent_space_dim)(input_layer)
            # What in the hot crispy kentucky fried fuck is this?
        layer = layers.LSTM(self.latent_space_dim, return_sequences=False, name="LTSM_Layer_1")(embedding)
        #output_layer = self._add_bottleneck(layer)
        #self._ltsm_state = [state_h, state_c]
        print(np.shape(input_layer), np.shape(embedding), np.shape(layer))
        
        self._model_input = input_layer
        return keras.Model(input_layer, layer, name="encoder")

    '''def _add_bottleneck(self, x):
        """Flatten data and add bottleneck (Dense layer)"""
        self._shape_before_bottleneck = kerasUtilities.int_shape(x)[1:] 
        x = layers.Flatten()(x)
        x = layers.Dense(self.latent_space_dim, name="encoder_output")(x)
        return x'''

        # End of encoder stuff
        # Beginning of decoder stuff
    def _build_decoder(self):
        input_layer = keras.Input(self.latent_space_dim, name="decoder_input")
        #postmultiplier = self._add_postmultiplier(input_layer) 
            # Postmultiply the data out of the latent space and back into the spectrogram-esque encoding it once had
        #reshape_layer = layers.Reshape(np.expand_dims(np.shape(input_layer), axis=2).shape)(input_layer)
            #reshape the encoded 1-Dimensional array back into a faux-3-D array
        embedding = layers.Embedding(self.input_shape, self.latent_space_dim)(input_layer)
        memory = layers.LSTM(self.ltsm_neurons, return_sequences=False, name="LTSM_Decoder_2")(embedding)
        output = layers.Activation("sigmoid", name="sigmoid_layer")(memory)
        print(np.shape(input_layer), np.shape(embedding),np.shape(memory), np.shape(output))
        return keras.Model(input_layer, output, name="Decoder")

    '''def _add_postmultiplier(self, input):
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = layers.Dense(num_neurons, name="Postmultiplier")(input)
        return dense_layer'''
    
        # End of decoder stuff
        # Beginning of object utilities
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()
    
    def train(self, x_train, learning_rate=0.0001, batch_size=32, num_epochs=20):
        self.model.fit(x_train,x_train, validation_data=(x_train,x_train), batch_size=batch_size, epochs=num_epochs, shuffle=True)

    def compile(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=self._calculate_combined_loss)

    def save(self, save_folder="."):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self._save_parameters(save_folder)
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = MusicGenerator(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.model.load_weights(weights_path)
        return autoencoder
    
    # End of utility functions
    # Beginning of internal functions

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = kl_loss = -0.5 * kerasUtilities.sum(1 + self.log_variance - kerasUtilities.square(self.mu) - kerasUtilities.exp(self.log_variance), axis=1)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = kerasUtilities.mean(kerasUtilities.square(error), axis=[1, 2, 3])
        return reconstruction_loss

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    
if __name__ == "__main__":
    arr = []
    for _, _, files in os.walk("C:\\Users\\joeya\\Documents\\GitHub\\AI Stuff\\spectrograms"):
        for i in range(len(files)):
            with open(os.path.join("C:\\Users\\joeya\\Documents\\GitHub\\AI Stuff\\spectrograms", files[i]), "rb") as f:
                arr.append(np.load(f))
    '''lengths = []
    for _, _, files in os.walk("C:\\Users\\joeya\\Documents\\GitHub\\AI Stuff\\recordings"):
        for i in range(len(files)):
            with audioread.audio_open(os.path.join("C:\\Users\\joeya\\Documents\\GitHub\\AI Stuff\\recordings", files[i])) as f:
                lengths.append(f.duration)'''
    print(np.shape(arr),np.shape(arr[1]))
    gen = MidiGenerator(256, 64, ltsm_neurons=256)
    gen.compile(learning_rate=0.0001)
    gen.train(*arr[0])
    gen.reconstruct([arr])

    '''first two inputs represent notes number 1 and 2
    The next two represent the octaves
    Then two are duration
    After that is position in the song
    Finally we have song length
    Outputs the next two notes in a sequence'''