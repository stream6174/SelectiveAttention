
import collections
import math
import string

import numpy as np
import tensorflow as tf

from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.base_layer import Layer
from keras.layers import activation
from keras.layers import core
from keras.layers import regularization
from keras.utils import tf_utils

from layers.component_builder import ComponentBuilder, InputDense, OutputDense

# isort: off
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export



class SelectiveAttention(Layer):

    def __init__(self,
                 key_dim,
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

    
    
    def _build_dense(self, query, value, key=None):

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
            self._band_size = round(self._key_dim / 2)


        # Verify shapes are valid
        if self._query_shape.rank is None or self._value_shape.rank is None:
            raise ValueError("Input shapes must have known rank")
        

        self._input_dense = InputDense(name="input_dense",
                                        query=query,
                                        key=key,
                                        value=value,
                                        key_dim=self._key_dim)
        
        self._output_dense = OutputDense(name="output_dense",
                                         query=query,
                                         key=key,
                                         value=value,
                                         key_dim=self._key_dim)
        
        
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
            self._band_size = round(self._key_dim /2)

        # Verify shapes are valid
        if self._query_shape.rank is None or self._value_shape.rank is None:
            raise ValueError("Input shapes must have known rank")


        # Any setup work performed only once should happen in an `init_scope`
        # to avoid creating symbolic Tensors that will later pollute any eager
        # operations.
        with tf_utils.maybe_init_scope(self):

            self._input_dense = InputDense(name="input_dense",
                                            query=query,
                                            key=key,
                                            value=value,
                                            key_dim=self._key_dim)
            
            self._output_dense = OutputDense(name="output_dense",
                                            query=query,
                                            key=key,
                                            value=value,
                                            key_dim=self._key_dim)
            
            self._component_builder = ComponentBuilder(name="components",
                                                        band_size=self._band_size, 
                                                        use_bias=self._use_bias)
            

            self._build_attention(self._value_shape.rank)

    """
    the att_scores_rank should be the same as the rank of the output_dense layer output object.
    """
    def _build_attention(self, rank):
        
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 1))
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
        
        filtered_context = tf.matmul(filter_input, context_input)
        filtered_context = self._dropout_layer(
            filtered_context, training=training
        )
        attended_info = tf.matmul(filtered_context, _input)


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
        
        attention_output = self._output_dense(attention_output)
        saliency_map = self._output_dense(attention_output)

        if query_is_ragged:
            attention_output = tf.RaggedTensor.from_tensor(
                attention_output, lengths=query_lengths
            )


        if return_attention_scores:
            return attention_output, saliency_map
        return attention_output