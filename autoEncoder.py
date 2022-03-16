import os
import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras.initializers as initializers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Conv2DTranspose, Activation, Reshape, Lambda
from tensorflow.keras import backend as K

class AutoEncoder:
    """Autoencoder represents a deep convolutional autoencoder architecture with mirrored encoder/decoder components"""
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape = input_shape # [Width, height, channels]
        self.conv_filters = conv_filters # [2,4,8] -> The first layer has 2 filters, the second 4 the third 8
        self.conv_kernels = conv_kernels # [3,5,3] -> The first layer has a 3x3 kernel size the second 5x5, third 3x3 
        self.conv_strides = conv_strides # [1,2,2] 
        self.latent_space_dim = latent_space_dim # Number of dimensions
        self.encoder = None
        self.decoder = None
        self._shape_before_bottleneck = None
        self.model = None
        self._model_input = None
        self._num_conv_layers = len(conv_filters)
        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()
    
    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer, mse_loss)

    def train(self, x_train, batch_size=32, num_epochs=20):
        """Train this autoencoder using the dataset given by x_train. 
        The expected output is supposed to be exactly the same as the input so don't worry about needing the right answers here"""
        self.model.fit(x_train, x_train, batch_size=batch_size, epochs=num_epochs, shuffle=True)

    def save(self, path="."):
        if(not os.path.exists(path)): 
            os.makedirs(path)
        
        parampath = os.path.join(path, "Params.pkl")
        with open(parampath, "w+b") as file:
            params = [self.input_shape, self.conv_filters, self.conv_kernels,self.conv_strides, self.latent_space_dim]
            pickle.dump(params, file)
        weightpath = os.path.join(path, "weights.h5")
        self.model.save_weights(weightpath)
            
    def load(save_folder="."):
        parampath = os.path.join(save_folder, "Params.pkl")
        with open(parampath, "rb") as f:
            params = pickle.load(f)
            newEncoder = AutoEncoder(*params)
        weightpath = os.path.join(save_folder, "weights.h5")
        newEncoder.model.load_weights(weightpath)
        return newEncoder

    #Autoencoder stuff
    def _build_autoencoder(self):
        input = self._model_input
        #self.encoder(input) acts like a function - it returns the encoded data given via the input
        #self.decoder(^^^) is the same way - it returns the decoded output of the file
        output = self.decoder(self.encoder(input)) #You'd never be able to recognise that considering the way this is set up though
        self.model = Model(input, output, name="autoencoder")

    #End of autoencoder stuff
    #Decoder stuff
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")
    
    def _add_dense_layer(self, input):
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name="decoder_dense")(input) # Apply the dense layer to the decoder input
        return dense_layer # What in the hot crispy kentucky fried fuck is this syntax though

    def _add_reshape_layer(self, dense):
        return Reshape(self._shape_before_bottleneck)(dense)
    
    def _add_conv_transpose_layers(self, x):
        """Add all conv transpose blocks. First loops throguh all the conv layers in reverse order and stops at the first layer"""
        for index in reversed(range(1,self._num_conv_layers)):
            x = self._add_conv_transpose_layer(index, x)
        return x
    def _add_conv_transpose_layer(self, index, x):
        convTransposeLayer = Conv2DTranspose(filters = self.conv_filters[index], 
        kernel_size=self.conv_kernels[index],
        strides=self.conv_strides[index],
        padding="same", name = f"decoder_conv_transpose_layer_{self._num_conv_layers-index}")
        x= convTransposeLayer(x)
        x = ReLU(name=f"decoder_relu_{self._num_conv_layers-index}")(x)
        x = BatchNormalization(name=f"decoder_bn_{self._num_conv_layers-index}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(filters = 1,
        kernel_size= self.conv_kernels[0], 
        strides = self.conv_strides[0],
        padding = "same", name = f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    #End of Decoder stuff
    #Encoder stuff
    def  _build_encoder(self):
        encoder_input = Input(self.input_shape, name="encoder_input")
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_conv_layers(self, encoderInput):
        """Creates all convolutional blocks in encoder"""
        x = encoderInput
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x
    
    def _add_conv_layer(self, index, x):
        """Adds a single convolutional block to a graph of layers consisting of conv2d + RELU + batch normalization """
        
        #Initialize the conv2d layer with all parameters
        conv_layer = Conv2D(filters = self.conv_filters[index], 
        kernel_size=self.conv_kernels[index], 
        strides=self.conv_strides[index], 
        padding="same", 
        name=f"encoder_conv_layer_{index+1}")

        x = conv_layer(x) # add the CONV2d layer to x
        x = ReLU(name=f"encoder_relu_layer_{index+1}")(x) # add the RELU to x
        x= BatchNormalization(name=f"encoder_bn_{index+1}")(x) # add the Normalization to x
        return x

    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck (Dense layer)"""
        self._shape_before_bottleneck = K.int_shape(x)[1:] # [batch size, width, height, # of channels] - Batch size is unimportant here, also equal to None
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name="encoder_output")(x)
        return x
    # End of encoder stuff
    def reconstruct(self, images):
        latent_reps = self.encoder.predict(images)
        reconstructions = self.decoder.predict(latent_reps)
        return reconstructions, latent_reps

#<--------------------------------------- VAR ENCODER START --------------------------------------->

class VarEncoder:
    """
    VarEncoder represents a Deep Convolutional variational autoencoder architecture
    with mirrored encoder and decoder components.
    """
    tf.compat.v1.disable_eager_execution()
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape = input_shape # [28, 28, 1]
        self.conv_filters = conv_filters # [2, 4, 8]
        self.conv_kernels = conv_kernels # [3, 5, 3]
        self.conv_strides = conv_strides # [1, 2, 2]
        self.latent_space_dim = latent_space_dim # 2
        self.reconstruction_loss_weight = 1000000

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.model = self._build()

    def _build(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input)) # What in the hot, crispy, kentucky fried fuck is going on here
        return Model(model_input, model_output, name="autoencoder")

# Beginning of Encoder functions

    def _build_encoder(self):
        encoder_input = Input(shape=self.input_shape, name="encoder_input")
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        return Model(encoder_input, bottleneck, name="encoder")

    def _add_conv_layers(self, encoder_input):
        """Create all convolutional blocks in encoder."""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Add a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(filters=self.conv_filters[layer_index], kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index], padding="same", name=f"encoder_conv_layer_{layer_number}")
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck with Guassian sampling (Dense
        layer).
        """
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        self.mu = Dense(self.latent_space_dim, name="mu")(x) # The average of a standard 3-D normalized variable
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x) # The log_variance of the above

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0.,
                                      stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

        x = Lambda(sample_point_from_normal_distribution, name="encoder_output")([self.mu, self.log_variance])
        return x

# End of Encoder Functions
# Beginning of Decoder functions  

    def _build_decoder(self):
        decoder_input = Input(shape=self.latent_space_dim, name="decoder_input")
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        return Model(decoder_input, decoder_output, name="decoder")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> 8
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_conv_transpose_layers(self, x):
        """Add conv transpose blocks."""
        # loop through all the conv layers in reverse order and stop at the
        # first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    # End of Decoder functions
    # Beginning of utility functions

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
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
    
    def generatefromLatentReps(self, latent):
        reconstructed_images = self.decoder.predict(latent)
        return reconstructed_images

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VarEncoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.model.load_weights(weights_path)
        return autoencoder
    
    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, x_train, batch_size=batch_size, epochs=num_epochs, shuffle=True)
    
    # End of utility functions
    # Beginning of internal functions

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis=1)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
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
    # End of internal functions
    # End of class
    
#<--------------------------------------- VAR ENCODER START --------------------------------------->

class ImageEncoder:
    """
    imageEncoder is an attempt at a Deep Convolutional variational autoencoder architecture
    with mirrored encoder and decoder components, intended for image generaton. 
    THIS IS EXPERIMENTAL. IT MAY WORK, IT MAY DESTROY YOUR PC. USE WITH CAUTION. 
    """
    tf.compat.v1.disable_eager_execution()
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape = input_shape # [28, 28, 1]
        self.conv_filters = conv_filters # [2, 4, 8]
        self.conv_kernels = conv_kernels # [3, 5, 3]
        self.conv_strides = conv_strides # [1, 2, 2]
        self.latent_space_dim = latent_space_dim # 2
        self.reconstruction_loss_weight = 1000000

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None
        self.n = 0; 

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.model = self._build()

    def _build(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input)) # What in the hot, crispy, kentucky fried fuck is going on here
        return Model(model_input, model_output, name="autoencoder")

# Beginning of Encoder functions

    def _build_encoder(self):
        #shape=(self.input_shape[0], self.input_shape[1],1); 
        #input = []
        #for i in range(self.input_shape[-1]):
        encoder_input = Input(shape=self.input_shape, name="encoder_input")
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        return Model(encoder_input, bottleneck, name="encoder")

    def _add_conv_layers(self, encoder_input):
        """Create all convolutional blocks in encoder."""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Add a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        self.n = self.n+1; 
        layer_number = self.n; 

        conv_layer = Conv2D(filters=self.conv_filters[layer_index], kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index], padding="same", name=f"encoder_conv_layer_{layer_number}")
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck with Guassian sampling (Dense
        layer).
        """
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        self.mu = Dense(self.latent_space_dim, name="mu")(x) # The average of a standard 3-D normalized variable
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x) # The log_variance of the above

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0.,
                                      stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

        x = Lambda(sample_point_from_normal_distribution, name="encoder_output")([self.mu, self.log_variance])
        return x

# End of Encoder Functions
# Beginning of Decoder functions  

    def _build_decoder(self):
        decoder_input = Input(shape=self.latent_space_dim, name="decoder_input")
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        return Model(decoder_input, decoder_output, name="decoder")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> 1*2*4 -> 8
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_conv_transpose_layers(self, x):
        """Add conv transpose blocks."""
        # loop through all the conv layers in reverse order and stop at the
        # first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = 1-Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    # End of Decoder functions
    # Beginning of utility functions

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
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

    def generatefromLatentReps(self, latent):
        reconstructed_images = self.decoder.predict(latent)
        return reconstructed_images

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VarEncoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.model.load_weights(weights_path)
        return autoencoder
    
    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, x_train, batch_size=batch_size, epochs=num_epochs, shuffle=True)
    
    # End of utility functions
    # Beginning of internal functions

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis=1)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
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
    # End of internal functions
    # End of class

if(__name__ == "__main__"):
    auto = AutoEncoder((28,28,1),(32,64,64,64),(3,3,3,3),(1,2,2,1),2)
    auto.summary()

'''first two inputs represent notes number 1 and 2
    The next two represent the octaves
    Then two are duration
    After that is position in the song
    Finally we have song length
    Outputs the next two notes in a sequence'''
#This is turning out to be just as difficult as I thought it would be....