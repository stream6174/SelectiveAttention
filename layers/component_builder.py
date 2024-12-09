# author: GAO Chenxi
# date: 2024/12/7 16:38
# -*- python version：3.8.10 -*-
# -*- coding: utf-8 -*-

import tensorflow as tf

class ComponentBuilder(tf.keras.layers.Layer):

    def __init__(self,
                 name,
                 transpose_a,
                 transpose_b,
                 is_filter=False,
                 ):
        super(ComponentBuilder, self).__init__(name=name)
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.is_filter = is_filter

    def _band_form(self,filter_input, max_backward=None, max_forward=None):

        band_size = round(filter_input.shape.as_list()[1] / 2)

        if max_forward is None:
            max_forward = -band_size
        if max_backward is None:
            max_backward = band_size

        band = tf.linalg.band_part(filter_input, max_forward, max_backward)
        band = tf.cast(band, tf.float32)
        return band
    def call(self, input_a, input_b, **kwargs):

        if self.is_filter:
            component = tf.multiply(input_a, input_b)
            component = self._band_form(component)
        else:
            component = tf.matmul(input_a, input_b, transpose_a=self.transpose_a, transpose_b=self.transpose_b)

        return component




