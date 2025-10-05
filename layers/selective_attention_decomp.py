
import collections
import string

import tensorflow as tf

from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.base_layer import Layer
from keras.layers import core
from keras.layers import regularization
from keras.utils import tf_utils


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

class ComponentBuilder(Layer):

    def __init__(self,
                 name,
                 num_heads,
                 band_size,
                 use_bias=True,
                 ):
        super(ComponentBuilder, self).__init__(name=name)
        
        self._num_heads = num_heads
        self._band_size = band_size
        self._use_bias = use_bias


    
    def _band_form(self, filter_input, max_backward=None, max_forward=None):
        band_limit = filter_input.shape.as_list()[-1]

        if self._band_size is None:
            self._band_size = band_limit //(self._num_heads * 2)
        elif self._band_size > band_limit:
            raise ValueError("The band size exceeding the largest limitation.")

        if max_forward is None:
            max_forward = self._band_size
        if max_backward is None:
            max_backward = self._band_size

        band = tf.linalg.band_part(filter_input, max_forward, max_backward)
        band = tf.cast(band, tf.float32)
        return band
    
    def call(self, query, key, value, **kwargs):

        query_decomposition = tf.keras.layers.Reshape(tuple(query.shape.as_list()[1:-1]+
                                                              [self._num_heads]+ [query.shape.as_list()[-1]//self._num_heads]),
                                                              input_shape=query.shape)(query)
        
        key_decomposition = tf.keras.layers.Reshape(tuple(key.shape.as_list()[1: -1]+
                                                          [self._num_heads]+[key.shape.as_list()[-1]//self._num_heads]),
                                                          input_shape=key.shape)(key)
        
        value_decomposition = tf.keras.layers.Reshape(tuple(value.shape.as_list()[1: -1]+
                                                              [self._num_heads]+ [value.shape.as_list()[-1]//self._num_heads]), 
                                                              input_shape=value.shape)(value)

        #filter_input using "*" to maximum captrued information 
        #then build filter with Hines Matrix to simulate inhibitions

        filter_query = self._band_form(query_decomposition)
        filter_input = tf.multiply(filter_query, filter_query)

        #filter_input = self._band_form(filter_input)
    
        filter_value = self._band_form(value_decomposition)

       
        context_input = tf.matmul(key_decomposition, filter_query, transpose_a=True)

        _input = tf.matmul(key_decomposition, filter_value, transpose_a=True)
        
        
        filter_input = tf.keras.layers.Reshape( tuple(filter_input.shape.as_list()[1:-2] +
                                                               [filter_input.shape.as_list()[-2] * filter_input.shape.as_list()[-1]]),
                                                               input_shape=filter_input.shape)(filter_input)
        context_input = tf.keras.layers.Reshape(tuple(context_input.shape.as_list()[1:-2] + 
                                                                [context_input.shape.as_list()[-2]*context_input.shape.as_list()[-1]]),
                                                                input_shape=context_input.shape)(context_input)
        _input = tf.keras.layers.Reshape( tuple(_input.shape.as_list()[1:-2] +
                                                  [_input.shape.as_list()[-2] * _input.shape.as_list()[-1]]),
                                                  input_shape=_input.shape)(_input)
        
        filter_input = tf.concat([filter_input, filter_input], axis=-1)
        
        return filter_input, context_input, _input


class SelectiveAttention(Layer):

    def __init__(self,
                 key_dim,
                 num_heads=4,
                 band_size=None,
                 value_dim=None,
                 dropout=0.2,
                 use_bias=True,
                 output_shape=None,
                 attention_axes=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="he_normal",
                 kernel_regularizer=None,

                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs,
                 ):
        super(SelectiveAttention, self).__init__()

        self._key_dim = key_dim
        self._num_heads = num_heads
        self._band_size = band_size
        self._value_dim = value_dim if value_dim else key_dim
        self._dropout = dropout
        self._use_bias = use_bias
        self._output_shape = output_shape
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)
        if attention_axes is not None and not isinstance(
            attention_axes, collections.abc.Sized
        ):
            self._attention_axes = (attention_axes,)
        else:
            self._attention_axes = attention_axes
        self._built_from_signature = False
        self._query_shape, self._key_shape, self._value_shape = None, None, None


    def get_config(self):
        self._clean_up()

        config = {
            "num_heads":self._num_heads,
            "band_size": self._band_size,
            "key_dim": self._key_dim,
            "value_dim": self._value_dim,
            "dropout": self._dropout,
            "use_bias": self._use_bias,
            "output_shape": self._output_shape,
            "attention_axes": self._attention_axes,
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
        
        
    def _build_from_signature(self, query, value, key=None):


        self._built_from_signature = True
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

        if self._band_size is None:
            self._band_size = round(self._key_dim //(self._num_heads * 2))

        # Verify shapes are valid
        if self._query_shape.rank is None or self._value_shape.rank is None:
            raise ValueError("Input shapes must have known rank")

        free_dims = self._value_shape.rank - 2
        output_dims = self._query_shape.rank - 2
            
        #key_dense and value_dense are with shape [B,T,dim], which is the same as the query.shape
            
        
        # Any setup work performed only once should happen in an `init_scope`
        # to avoid creating symbolic Tensors that will later pollute any eager
        # operations.
        with tf_utils.maybe_init_scope(self):
            _einsum_equation, _bias_axes = _build_proj_equation(
                        free_dims, output_dims, 1
                    )
            self._dense = tf.keras.layers.Dense(self._key_dim)

            self._input_dense = core.EinsumDense(
                                _einsum_equation,
                                output_shape= list(self._value_shape[1:-1])+[self._key_dim],
                                bias_axes=_bias_axes if self._use_bias else None,
                                activation="relu",
                                name="qkv",
                                **self._get_common_kwargs_for_sublayer()
                                )
            _einsum_equation, _bias_axes= _build_proj_equation(
                        self._value_shape.rank - 2, self._query_shape.rank - 2, 1
                    )
            self._output_dense =core.EinsumDense(
                                _einsum_equation,
                                output_shape= list(self._query_shape)[1:-1] + [self._key_dim],
                                bias_axes=_bias_axes if self._use_bias else None,
                                activation="relu",
                                name="output",
                                **self._get_common_kwargs_for_sublayer(),
                            )
            
            self._component_builder = ComponentBuilder(name="components",
                                                        num_heads=self._num_heads, 
                                                        band_size=self._band_size,
                                                        use_bias=self._use_bias)
            

            self._build_attention(self._value_shape.rank)

    """
    the att_scores_rank should be the same as the rank of the output_dense layer output object.
    """
    def _build_attention(self, rank):
        
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 1))
            norm_axes = -1
        else:
            self._attention_axes = tuple(self._attention_axes)
            norm_axes = tuple(
                range(
                    rank - len(self._attention_axes), rank
                )
            )
        self._softmax = tf.keras.layers.Softmax(axis=norm_axes)
        self._dropout_layer = regularization.Dropout(rate=self._dropout)


    def _compute_attention(
        self, filter_input, context_input, _input, training=None
    ):
        
        filtered_context = tf.matmul(filter_input, context_input, transpose_a=True)
        filtered_context = self._dropout_layer(
            filtered_context, training=training
        )
        
        attended_info = tf.matmul(_input, filtered_context)
        #reuse the object level information to reinforce the feature-category mapping
        attended_info = tf.matmul(attended_info, filtered_context)

        return attended_info, filtered_context
    
    
    def call(
        self,
        query,
        value,
        key=None,
        return_attention_scores=False,
        training=None,
    ):
        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)

        if key is None:
            key = value

        # Convert RaggedTensor to Tensor.
        query_is_ragged = isinstance(query, tf.RaggedTensor)
        if query_is_ragged:
            query_lengths = query.nested_row_lengths()
            query = query.to_tensor()
        key_is_ragged = isinstance(key, tf.RaggedTensor)
        value_is_ragged = isinstance(value, tf.RaggedTensor)
        if key_is_ragged and value_is_ragged:
            # Ensure they have the same shape.
            bounding_shape = tf.math.maximum(
                key.bounding_shape(), value.bounding_shape()
            )
            key = key.to_tensor(shape=bounding_shape)
            value = value.to_tensor(shape=bounding_shape)
        elif key_is_ragged:
            key = key.to_tensor(shape=tf.shape(value))
        elif value_is_ragged:
            value = value.to_tensor(shape=tf.shape(key))
    


        # Verify shapes are valid
        if self._query_shape.rank is None or self._value_shape.rank is None:
            raise ValueError("Input shapes of EinsumDense must have known rank")
        
        query = self._input_dense(query)
        key = self._input_dense(key)
        value = self._input_dense(value)

        filter_input, context_input, _input = self._component_builder(query, key, value)

        attention_output, saliency_map = self._compute_attention(
            filter_input, context_input, _input, training
        )

        attention_output = self._dense(attention_output)
        saliency_map = self._dense(saliency_map)
        
        attention_output = self._output_dense(attention_output)
        #saliency_map = self._output_dense(saliency_map)

        if query_is_ragged:
            attention_output = tf.RaggedTensor.from_tensor(
                attention_output, lengths=query_lengths
            )


        if return_attention_scores:
            return attention_output, saliency_map
        return attention_output
