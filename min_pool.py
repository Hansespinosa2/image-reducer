
from keras.layers import Layer, MaxPooling2D
import keras.backend as K

class MinPooling2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', **kwargs):
        super(MinPooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs):
        # Negate the inputs to turn them into a scenario where max pooling will act like min pooling
        inputs_neg = -inputs
        # Use MaxPooling2D on the negated inputs
        output_neg = MaxPooling2D(self.pool_size, self.strides, self.padding)(inputs_neg)
        # Negate the output to turn the values back
        output = -output_neg
        return output

    def compute_output_shape(self, input_shape):
        # You may need to adjust this calculation depending on your exact padding and strides
        rows = input_shape[1]
        cols = input_shape[2]
        out_rows = (rows + self.pool_size[0] - 1) // self.pool_size[0]
        out_cols = (cols + self.pool_size[1] - 1) // self.pool_size[1]
        return (input_shape[0], out_rows, out_cols, input_shape[3])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config
