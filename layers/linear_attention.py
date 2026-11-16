import tensorflow as tf
from keras.layers import Layer, Conv1D, Softmax

class ImageLinearAttention(Layer):
    def __init__(self, chan, chan_out = None, kernel_size = 1,padding='valid', stride = 1, key_dim = 64, value_dim = 64, heads = 8, norm_queries = True):
        super().__init__()
        self.chan = chan
        chan_out = chan if chan_out is None else chan_out

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.heads = heads

        self.norm_queries = norm_queries

        conv_kwargs = {'padding': padding, 'strides': stride}
        self.to_q = Conv1D(key_dim * heads, kernel_size, activation="relu", **conv_kwargs)
        self.to_k = Conv1D(key_dim * heads, kernel_size, activation="relu", **conv_kwargs)
        self.to_v = Conv1D(value_dim * heads, kernel_size, activation="relu",  **conv_kwargs)
    
        out_conv_kwargs = {'padding': padding}
        self.to_out = Conv1D(value_dim, kernel_size, activation="relu", **out_conv_kwargs)

    def call(self, x, context = None):
        k_dim, heads = self.key_dim, self.heads
        b, n, c = x.shape

        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

        q = tf.keras.activations.elu(q) + 1
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

        context_einsum_equation = 'hdn,hen->hde'
        output_einsum_equation = 'hdn,hde->hen'

        context = tf.einsum(context_einsum_equation, k, v)
        out = tf.einsum(output_einsum_equation, q, context)
        out = self.to_out(out)
        
        return out