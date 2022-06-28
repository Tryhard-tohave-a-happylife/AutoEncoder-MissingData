import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.metrics import Mean
from tensorflow.keras import optimizers

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var
    
    @staticmethod
    def kl_divergence(mean, logvar):
        return -0.5 * tf.reduce_sum(
                   1 + logvar - tf.square(mean) -
                   tf.exp(logvar), axis=1)
    
    @staticmethod
    def binary_cross_entropy_with_logits(logits, labels):
        logits = tf.math.log(logits)
        ret = 0 - tf.reduce_sum(labels * logits + (1-labels) * tf.math.log(- tf.math.expm1(logits)), axis=1)
        tf.print("ret: ", logits)
        return ret
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            # z_mean, z_log_var, z = self.encoder(data)
            # reconstruction = self.decoder(z)
            reconstruction, z_mean, z_log_var = self(data)
            # kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(self.kl_divergence(z_mean, z_log_var))
            #print(reconstruction.shape[1])
            #reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(data, reconstruction), axis=1)) 
            reconstruction_loss = tf.reduce_sum(tf.pow(reconstruction - data , 2)) / reconstruction.shape[1]
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
def create_encoder_decoder(input_size, layer_units, latent_dim, activation_func):
    encoder_inputs = Input(shape=(input_size, ))
    #Encoder Block
    for idx, each in enumerate(layer_units):
        if idx == 0:
            x = Dense(units=each, activation=activation_func, kernel_initializer="he_normal")(encoder_inputs)
            continue
        x = Dense(units=each, activation=activation_func, kernel_initializer="he_normal")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    
    #Decoder Block
    latent_inputs = Input(shape=(latent_dim, ))
    for idx, each in enumerate(layer_units[::-1]):
        if idx == 0:
            x = Dense(units=each, activation=activation_func, kernel_initializer="he_normal")(latent_inputs)
            continue
        x = Dense(units=each, activation=activation_func, kernel_initializer="he_normal")(x)
    decoder_outputs = Dense(units=input_size, kernel_initializer="he_normal")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    return encoder, decoder

def create_model(input_size, layer_units=[16, 8], latent_dim=5, activation_func="relu"):
    encoder, decoder = create_encoder_decoder(input_size, layer_units, latent_dim, activation_func)
    vae_model = VAE(encoder, decoder)
    
    return vae_model

def compile_model(model, opt="adam", lr=0.008):
    if opt.lower() == "adam":
        model.compile(optimizer=optimizers.Adam(learning_rate=lr))
    else:
        model.compile(optimizer=optimizers.Adam(learning_rate=lr))
    return model

