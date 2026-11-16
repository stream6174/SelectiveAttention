import tensorflow as tf
from keras.layers import Layer, Conv1D, Softmax

class ImageLinearSelectiveAttention(Layer):
    def __init__(self, chan, chan_out = None, kernel_size = 1,padding='valid', stride = 1, key_dim = 64, value_dim = 64, heads = 8, dropout = 0.2, norm_queries = True, band_size=None):
        super().__init__()
        self.chan = chan
        chan_out = chan if chan_out is None else chan_out

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.heads = heads

        self.dropout = dropout
        self.norm_queries = norm_queries
        self.band_size = band_size

        conv_kwargs = {'padding': padding, 'strides': stride}
        self.to_q = Conv1D(key_dim, kernel_size, activation="relu", **conv_kwargs)
        self.to_k = Conv1D(key_dim, kernel_size, activation="relu", **conv_kwargs)
        self.to_v = Conv1D(value_dim, kernel_size, activation="relu",  **conv_kwargs)
    
        out_conv_kwargs = {'padding': padding}
        self.to_out = Conv1D(value_dim, kernel_size, activation="relu", **out_conv_kwargs)
        self._dense_layer = tf.keras.layers.Dense(key_dim)
        self._dropout_layer = tf.keras.layers.Dropout(rate=self.dropout)

    def _band_form(self, x, max_forward=None, max_backward=None):

        if max_forward is None:
            max_forward = self.band_size
        if max_backward is None:
            max_backward = self.band_size

        band = tf.linalg.band_part(x, max_forward, max_backward)        
        band = tf.cast(band, tf.float32)
        return band

    def call(self, x, context = None):
        k_dim, heads = self.key_dim, self.heads
        b, n, c = x.shape

        if self.band_size is None:
            self.band_size = n // 4

        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

        q= tf.keras.activations.elu(q) + 1
        k = tf.keras.activations.elu(k) + 1
        v = tf.keras.activations.elu(v) + 1

        q, k = map(lambda x: x * (k_dim ** -0.25), (q, k))

        if context is not None:
            ck, cv = self.to_k(context), self.to_v(context)
            ck = tf.keras.activations.elu(ck) + 1
            cv = tf.keras.activations.elu(cv) + 1
            ck, cv = map(lambda t: t * (k_dim ** -0.25), (ck, cv))

            k = tf.concat([k, ck], axis=-1)
            v = tf.concat([v, cv], axis=-1)

        k = Softmax(axis=-1)(k)

        if self.norm_queries:
            q = Softmax(axis=-2)(q)

        q = self._band_form(q)
        filter_input = tf.multiply(q, q)

        context_input = tf.matmul(k, v, transpose_b=True)
        filtered_context = tf.matmul(context_input, filter_input)
        filtered_context = self._dropout_layer(filtered_context)
        _input = tf.matmul(q, self._dense_layer(k), transpose_a=True)
        out = tf.matmul(filtered_context, _input)
        out = tf.multiply(out, filtered_context)
        out = self.to_out(out)
        
        return out