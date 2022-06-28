import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.metrics import Mean
from tensorflow.keras import optimizers

class CAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.contractive_loss_tracker = Mean(name="contractive_loss")
          
    def call(self, inputs):
        x_latent = self.encoder(inputs)
        x = self.decoder(x_latent)
        return x, x_latent
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.contractive_loss_tracker,
        ]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction, latent_output = self(data)
            latent_layer = self.encoder.get_layer("latent_layer")
            total_loss, reconstruction_loss, contractive_loss = \
                        tf.function(calculate_loss).get_concrete_function(data, reconstruction, latent_output, latent_layer)
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.contractive_loss_tracker.update_state(contractive_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "contractive_loss": self.contractive_loss_tracker.result()
        }
    
def calculate_loss(x, x_bar, h, latent_layer):
    lambda_ = 100
    reconstruction_loss = tf.reduce_mean( 
                tf.keras.losses.mse(x, x_bar) 
            ) 
    W = tf.Variable(latent_layer.weights[0])
    dh = h * (1 - h)  # N_batch x N_hidden
    W = tf.transpose(W)
    contractive = lambda_ * tf.reduce_sum(tf.linalg.matmul(dh**2 ,tf.square(W)), axis=1)
    total_loss = reconstruction_loss + contractive
    return total_loss, reconstruction_loss, contractive

def create_encoder_decoder(input_size, layer_units, latent_dim, activation_func):
    encoder_inputs = Input(shape=(input_size, ))
    #Encoder Block
    for idx, each in enumerate(layer_units):
        if idx == 0:
            x = Dense(units=each, activation=activation_func, kernel_initializer="he_normal")(encoder_inputs)
            continue
        x = Dense(units=each, activation=activation_func, kernel_initializer="he_normal")(x)
    encoder_output = Dense(latent_dim, activation=activation_func, kernel_initializer="he_normal", name="latent_layer")(x)
    encoder = Model(encoder_inputs, encoder_output, name="encoder")
    
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
    cae_model = CAE(encoder, decoder)
    
    return cae_model

def compile_model(model, opt="adam", lr=0.0005):
    if opt.lower() == "adam":
        model.compile(optimizer=optimizers.Adam(learning_rate=lr))
    else:
        model.compile(optimizer=optimizers.Adam(learning_rate=lr))
    return model

def grad(model, inputs):
    with tf.GradientTape() as tape:
        reconstruction, inputs_reshaped,hidden = model(inputs)
        loss_value = loss(inputs_reshaped, reconstruction, hidden, model)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), inputs_reshaped, reconstruction

def custom_train(model, train_data, epochs=10, batch_size=32):
    for epoch in range(epochs): 
        print("Epoch: ", epoch)
        for x in range(0, len(train_data), batch_size):
            x_inp = x_train[x : x + batch_size]
            loss_value, grads, inputs_reshaped, reconstruction = grad(model, x_inp)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print("Step: {}, Loss: {}".format(global_step.numpy(),tf.reduce_sum(loss_value)))
