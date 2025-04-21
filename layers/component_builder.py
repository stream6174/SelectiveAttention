import tensorflow as tf

from keras import constraints
from keras import initializers
from keras import regularizers

from keras.layers import core
from keras.engine.base_layer import Layer
import string


# isort: off
from tensorflow.python.platform import tf_logging as logging

_CHR_IDX = string.ascii_lowercase



#the key_dense and value_dense should output a tensor with the same shape as the query.shape,
#the output_dense generates a tensor with the same shape as value.shape

def _build_proj_equation(source_dims,target_dims, bound_dims):
    """Builds an einsum equation for projections inside multi-head attention."""
    input_str = "a" #set the dimension for batch_size
    kernel_str = ""
    output_str = "a"
    bias_axes = ""
    letter_offset = 1
    for i in range(source_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        kernel_str += char
    

    letter_offset += source_dims
    for i in range(target_dims):
        char = _CHR_IDX[i + letter_offset]
        kernel_str += char
        output_str += char
        

    letter_offset += target_dims
    for i in range(bound_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char # complete joint the input notations, bound dimensions is kim_dim
        kernel_str += char
        output_str += char
        bias_axes += char

    equation = f"{input_str},{kernel_str}->{output_str}"

    return equation, bias_axes
             


class InputDense(Layer):

    def __init__(self,
                 name,
                 query,
                 key,
                 value,
                 key_dim,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 ):
        super(InputDense, self).__init__(name=name)

        self._key_dim = key_dim
        self._use_bias = use_bias
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)

        self._built_from_signature = False


        if hasattr(query, "shape"):
            self._query_shape = tf.TensorShape(query.shape)
        else:
            self._query_shape = tf.TensorShape(query)
        if hasattr(value, "shape"):
            self._value_shape = tf.TensorShape(value.shape)
        else:
            self._value_shape = tf.TensorShape(value)
        if key is None:
            self._key_shape = self._value_shape
        elif hasattr(key, "shape"):
            self._key_shape = tf.TensorShape(key.shape)
        else:
            self._key_shape = tf.TensorShape(key)

        
        free_dims = self._value_shape.rank - 2
        output_dims = self._query_shape.rank - 2
            
        #key_dense and value_dense are with shape [B,T,dim], which is the same as the query.shape
            
        self._einsum_equation, self._bias_axes = _build_proj_equation(
            free_dims, output_dims, 1
        )


    def get_config(self):
        config = {
            "key_dim": self._key_dim,
            "use_bias": self._use_bias,
            "kernel_initializer": initializers.serialize(
                self._kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self._bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self._kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self._bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self._activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self._kernel_constraint),
            "bias_constraint": constraints.serialize(self._bias_constraint),
            "query_shape": self._query_shape,
            "key_shape": self._key_shape,
            "value_shape": self._value_shape,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):

        query_shape = config.pop("query_shape")
        key_shape = config.pop("key_shape")
        value_shape = config.pop("value_shape")
        layer = cls(**config)
        if None in query_shape or value_shape or key_shape:
            logging.warning(
                "One of dimensions of the input shape is missing. It "
                "should have been memorized when the layer was serialized. "
                "%s is created without weights.",
                str(cls),
            )
        else:
            layer._build_from_signature(query_shape, value_shape, key_shape)
        return layer
        
    
    def _get_common_kwargs_for_sublayer(self):
        common_kwargs = dict(
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )
        # Create new clone of kernel/bias initializer, so that we don't reuse
        # the initializer instance, which could lead to same init value since
        # initializer is stateless.
        kernel_initializer = self._kernel_initializer.__class__.from_config(
            self._kernel_initializer.get_config()
        )
        bias_initializer = self._bias_initializer.__class__.from_config(
            self._bias_initializer.get_config()
        )
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return common_kwargs
    


    def call(self, input):
        return core.EinsumDense(
            self._einsum_equation,
            output_shape= list(self._value_shape[1:-1])+[self._key_dim],
            bias_axes=self._bias_axes if self._use_bias else None,
            activation="relu",
            name="qkv",
            **self._get_common_kwargs_for_sublayer()
        )(input)
    
class OutputDense(Layer):

    def __init__(self,
                 name,
                 query,
                 key,
                 value,
                 key_dim,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None ,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 ):
        super(OutputDense, self).__init__(name=name)
        self._key_dim = key_dim
        self._use_bias = use_bias
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)

        self._built_from_signature = False

        if hasattr(query, "shape"):
            self._query_shape = tf.TensorShape(query.shape)
        else:
            self._query_shape = tf.TensorShape(query)
        if hasattr(value, "shape"):
            self._value_shape = tf.TensorShape(value.shape)
        else:
            self._value_shape = tf.TensorShape(value)
        if key is None:
            self._key_shape = self._value_shape
        elif hasattr(key, "shape"):
            self._key_shape = tf.TensorShape(key.shape)
        else:
            self._key_shape = tf.TensorShape(key)

        self._einsum_equation, self._bias_axes= _build_proj_equation(
            self._value_shape.rank - 2, self._query_shape.rank - 2, 1
        )


    def get_config(self):
        config = {
            "key_dim": self._key_dim,
            "use_bias": self._use_bias,
            "kernel_initializer": initializers.serialize(
                self._kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self._bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self._kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self._bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self._activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self._kernel_constraint),
            "bias_constraint": constraints.serialize(self._bias_constraint),
            "query_shape": self._query_shape,
            "key_shape": self._key_shape,
            "value_shape": self._value_shape,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):

        query_shape = config.pop("query_shape")
        key_shape = config.pop("key_shape")
        value_shape = config.pop("value_shape")
        layer = cls(**config)
        if None in [query_shape, key_shape, value_shape]:
            logging.warning(
                "One of dimensions of the input shape is missing. It "
                "should have been memorized when the layer was serialized. "
                "%s is created without weights.",
                str(cls),
            )
        else:
            layer._build_from_signature(query_shape, value_shape, key_shape)
        
        return layer 


    def _get_common_kwargs_for_sublayer(self):
        common_kwargs = dict(
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )

        kernel_initializer = self._kernel_initializer.__class__.from_config(
            self._kernel_initializer.get_config()
        )
        bias_initializer = self._bias_initializer.__class__.from_config(
            self._bias_initializer.get_config()
        )
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return common_kwargs
    
    @property
    def clear(self):
        del self._output_dense
    
    def call(self, output):
        return core.EinsumDense(
            self._einsum_equation,
            output_shape= list(self._query_shape)[1:-1] + [self._key_dim],
            bias_axes=self._bias_axes if self._use_bias else None,
            activation="relu",
            name="output",
            **self._get_common_kwargs_for_sublayer(),
        )(output)
    
class ComponentBuilder(Layer):

    def __init__(self,
                 name,
                 band_size,
                 use_bias=True,
                 ):
        super(ComponentBuilder, self).__init__(name=name)
        
        self._band_size = band_size
        self._use_bias = use_bias

    
    def _band_form(self,filter_input, max_backward=None, max_forward=None):

        if max_forward is None:
            max_forward = -self._band_size
        if max_backward is None:
            max_backward = self._band_size

        band = tf.linalg.band_part(filter_input, max_forward, max_backward)
        band = tf.cast(band, tf.float32)
        return band
    
    def call(self, query, key, value, **kwargs):
        #filter_input using "*" to maximum captrued information 
        #then build filter with Hines Matrix to simujlate inhibitions
        filter_input = tf.multiply(query, query)
        filter_input = self._band_form(filter_input)
        
        #context_input and input using "@" to keep the projected information
        context_input = tf.matmul(query, key, transpose_a=True)
        _input = tf.matmul(key, value, transpose_a=True)

        return filter_input, context_input, _input



