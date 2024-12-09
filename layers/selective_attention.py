# author: GAO Chenxi
# date: 2024/11/30 10:18
# -*- python version：3.8.10 -*-
# -*- coding: utf-8 -*-
# modified version of selective attention based on keras multi-head attention layer


import collections
import numpy as np
import string
from layers.component_builder import ComponentBuilder

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Softmax
from tensorflow.keras import initializers, regularizers, constraints

class SelectiveAttention(Layer):

    def __init__(self,
                 num_heads,
                 key_dim,
                 value_dim=None,
                 dropout=0.0,
                 use_bias=True,
                 output_shape=None,
                 attention_axes=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self._num_heads = num_heads
        self._key_dim = key_dim
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
        config = {
            "num_heads": self._num_heads,
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

    def from_config(cls, config):
        # If the layer has a different build() function from the Keras default,
        # we need to trigger the customized build to create weights.
        query_shape = config.pop("query_shape")
        key_shape = config.pop("key_shape")
        value_shape = config.pop("value_shape")
        layer = cls(**config)
        if not None in [query_shape, key_shape, value_shape]:
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

    def _build_attention(self, rank):

        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)

        self._softmax = Softmax()
        self._dropout_layer = Dropout(rate=self._dropout)

    def _build_from_signature(self, query, value,key=None, is_cross_attention=False):

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

        output_rank = len(self._query_shape)

        with tf.init_scope():
            self._filter_input = ComponentBuilder(name="filter", transpose_a=False, transpose_b=False, is_filter=True)
            self._content_input = ComponentBuilder(name="content", transpose_a=True, transpose_b=False)
            self._input = ComponentBuilder(name="input", transpose_a=True, transpose_b=False)
            self._build_attention(output_rank)


    def _compute_components(self, query, key, value):

        filter_input = self._filter_input(query, query)
        content_input = self._content_input(query, key)
        x_input = self._input(key, value)

        return filter_input, content_input, x_input


    def _compute_attention_scores(self, filter_input, content_input):
        attention_scores = tf.matmul(filter_input, content_input)
        #attention_scores = tf.reshape(attention_scores, self._query_shape)
        attention_scores = self._softmax(attention_scores)
        return attention_scores

    def _compute_attention_dropout(self, attention_scores):
        attention_dropout = Dropout(rate=self._dropout)(attention_scores)
        return attention_dropout
    def _compute_attention_output(self, attention_dropout, input):

        attention_output = tf.matmul(attention_dropout, input)

        return attention_output


    def _compute_attention(self,
                           query,
                           key,
                           value,
                           training=None):

        filter_input, content_input, x_input = self._compute_components(query, key, value)
        attention_scores = self._compute_attention_scores(filter_input, content_input)


        if training is not None:
            attention_probs = self._compute_attention_dropout(attention_scores)
            attention_output = self._compute_attention_output(attention_probs, x_input)
        else:
            attention_output = self._compute_attention_output(attention_scores, x_input)

        return attention_output, attention_scores

    def call(self,
             query,
             key,
             value,
             attention_mask=None,
             return_attention_scores=False,
             training=None,
             use_causal_mask=False,
             is_cross_attention=False
             ):
        if key is None:
            key = value
        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key, is_cross_attention=is_cross_attention)


        attention_output, attention_scores = self._compute_attention(
            query, key, value, training
        )

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output
