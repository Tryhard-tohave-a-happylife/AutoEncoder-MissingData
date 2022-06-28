import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.metrics import Mean
from tensorflow.keras import optimizers

class AEClassification(Model):
    def __init__(self, model_dae, model_clf, alpha, **kwargs):
        super(AEClassification, self).__init__(**kwargs)
        self.model_dae = model_dae
        self.model_clf = model_clf
        self.alpha = alpha
        self.total_loss = Mean(name="total_loss")
        self.reconstruction_loss = Mean(name="reconstruction_loss")
        self.clf_loss = Mean(name="clf_loss")
    
    def call(self, inputs):
        res = self.model_dae(inputs)
        final = self.model_clf(res)
        return res, final
    
    @property
    def metrics(self):
        return [
            self.total_loss,
            self.reconstruction_loss,
            self.clf_loss,
        ]
    
    @tf.function
    def train_step(self, data):
        X, y = data
        with tf.GradientTape() as tape:
            res, final = self(X)
            reconstruction_loss = tf.reduce_sum(tf.pow(res - X , 2)) / res.shape[1]
            _loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
            clf_loss = _loss(y, final)
            total_loss = (1 - self.alpha) * reconstruction_loss + self.alpha * clf_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss.update_state(total_loss)
        self.reconstruction_loss.update_state(reconstruction_loss)
        self.clf_loss.update_state(clf_loss)
        return {
            "total_loss": self.total_loss.result(),
            "reconstruction_loss": self.reconstruction_loss.result(),
            "clf_loss": self.clf_loss.result(),
        }
    
def create_model_dae(input_size, num_layer, drop_out_rate, theta, activation_func):
    input_ = Input(shape = (input_size, ))
    x = Dropout(rate=drop_out_rate)(input_)
    #Encoder Block
    for i in range(1, num_layer + 1):
        x = Dense(units=input_size + theta * i, activation=activation_func, kernel_initializer="he_normal")(x)
    #Encoder Block
    for i in range(1, num_layer):
        x  = Dense(units=input_size + theta * (num_layer - i), activation=activation_func, kernel_initializer="he_normal")(x)
     
    output = Dense(units=input_size, kernel_initializer="he_normal")(x)
    model = Model(input_, output)
    return model

def model_for_classification(input_size, num_class):
    
    input_ = Input(shape = (input_size, ))
    x = Dense(units=128, activation="relu", kernel_initializer="he_normal")(input_)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=64, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    output_ = Dense(num_class, activation="softmax")(x)
    return Model(input_, output_)

def create_model(num_class, input_size, num_layer, drop_out_rate=0.5, theta=7, activation_func="tanh", alpha=0.5):
    model_dae = create_model_dae(input_size, num_layer, drop_out_rate, theta, activation_func)
    model_clf = model_for_classification(input_size, num_class)
    final_model = AEClassification(model_dae, model_clf, alpha)
    
    return final_model

def compile_model(model, opt="adam", lr=0.01):
    if opt.lower() == "adam":
        model.compile(optimizer=optimizers.Adam(learning_rate=lr))
    else:
        model.compile(optimizer=optimizers.Adam(learning_rate=lr))
    return model

