#code/DLEPS/models/model_zinc.py
import copy
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Lambda, Activation, Flatten,
                                     RepeatVector, TimeDistributed, GRU, Conv1D)
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
import zinc_grammar as G

# Helper variables in Keras format for parsing the grammar
masks_K = K.variable(G.masks)
ind_of_ind_K = K.variable(G.ind_of_ind)

MAX_LEN = 277
DIM = G.D

class MoleculeVAE:
    autoencoder = None

    def create(self, charset, max_length=MAX_LEN, latent_rep_size=2, weights_file=None):
        charset_length = len(charset)

        x = Input(shape=(max_length, charset_length))
        _, z = self._buildEncoder(x, latent_rep_size, max_length)
        self.encoder = Model(x, z)

        encoded_input = Input(shape=(latent_rep_size,))
        decoder_output = self._buildDecoder(encoded_input, latent_rep_size, max_length, charset_length)
        self.decoder = Model(encoded_input, decoder_output)

        x1 = Input(shape=(max_length, charset_length))
        vae_loss, z1 = self._buildEncoder(x1, latent_rep_size, max_length)
        decoder_output = self._buildDecoder(z1, latent_rep_size, max_length, charset_length)
        self.autoencoder = Model(x1, decoder_output)

        # For obtaining mean and log variance of encoding distribution
        x2 = Input(shape=(max_length, charset_length))
        z_m, z_l_v = self._encoderMeanVar(x2, latent_rep_size, max_length)
        self.encoderMV = Model(inputs=x2, outputs=[z_m, z_l_v])

        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name=True)
            self.decoder.load_weights(weights_file, by_name=True)
            self.encoderMV.load_weights(weights_file, by_name=True)

        self.autoencoder.compile(optimizer='Adam', loss=vae_loss, metrics=['accuracy'])

    def _encoderMeanVar(self, x, latent_rep_size, max_length, epsilon_std=0.01):
        h = Conv1D(9, 9, activation='relu', name='conv_1')(x)
        h = Conv1D(9, 9, activation='relu', name='conv_2')(h)
        h = Conv1D(10, 11, activation='relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        return z_mean, z_log_var

    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std=0.01):
        h = Conv1D(9, 9, activation='relu', name='conv_1')(x)
        h = Conv1D(9, 9, activation='relu', name='conv_2')(h)
        h = Conv1D(10, 11, activation='relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        # This function is the main change.
        # We mask the training data to allow only future rules based on the current non-terminal
        def conditional(x_true, x_pred):
            most_likely = K.argmax(x_true)
            most_likely = tf.reshape(most_likely, [-1])  # Flatten most_likely
            ix2 = tf.expand_dims(tf.gather(ind_of_ind_K, most_likely), 1)  # Index ind_of_ind with res
            ix2 = tf.cast(ix2, tf.int32)  # Cast indices as ints
            M2 = tf.gather_nd(masks_K, ix2)  # Get slices of masks_K with indices
            M3 = tf.reshape(M2, [-1, MAX_LEN, DIM])  # Reshape them
            P2 = tf.multiply(tf.exp(x_pred), M3)  # Apply them to the exp-predictions
            P2 = tf.divide(P2, K.sum(P2, axis=-1, keepdims=True))  # Normalize predictions
            return P2

        def vae_loss(x, x_decoded_mean):
            x_decoded_mean = conditional(x, x_decoded_mean)  # Apply masking
            x_flat = K.flatten(x)
            x_decoded_flat = K.flatten(x_decoded_mean)
            xent_loss = max_length * binary_crossentropy(x_flat, x_decoded_flat)
            kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        z = Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var])
        return vae_loss, z

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
        h = Dense(latent_rep_size, name='latent_input', activation='relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences=True, name='gru_1')(h)
        h = GRU(501, return_sequences=True, name='gru_2')(h)
        h = GRU(501, return_sequences=True, name='gru_3')(h)
        return TimeDistributed(Dense(charset_length), name='decoded_mean')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)

    def load(self, charset, weights_file, latent_rep_size=2, max_length=MAX_LEN):
        self.create(charset, max_length=max_length, weights_file=weights_file, latent_rep_size=latent_rep_size)
