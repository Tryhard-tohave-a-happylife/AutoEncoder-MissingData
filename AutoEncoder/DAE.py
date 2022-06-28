from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input

def create_model(input_size, num_layer, drop_out_rate=0.5, theta=7, activation_func="tanh"):
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