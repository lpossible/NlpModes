"""
@author:Luo Ping
@date:2020-8-4
"""
import tensorflow as tf
import tensorflow.keras.backend as kb
from keras.layers import Layer


class LocalAttention(Layer):
    """
    LocalAttention Layer: Two type:@1 using the same kernel @2 using  the different kernel
    """

    def __init__(self, **kwargs):
        super(LocalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.wk = self.add_weight(shape=(input_shape[-1], 100), initializer="glorot_uniform")
        self.wq = self.add_weight(shape=(input_shape[-1], 100), initializer="glorot_uniform")
        self.wv = self.add_weight(shape=(input_shape[-1], 100), initializer="glorot_uniform")
        # there is a question:because the shape[0] equals None so that it can not as the value of first dim
        # so must set a specific value according to data.at present,the max value is 2 that the number of data is 40526
        # to accelerate,i change the data num to 50000,set the batch=16
        self.forwardpad = self.add_weight(shape=(16, 1, input_shape[-1]), initializer="glorot_uniform")
        self.backwardpad = self.add_weight(shape=(16, 1, input_shape[-1]), initializer="glorot_uniform")

    def call(self, inputs, **kwargs):
        # forward pad
        inputs = tf.concat((self.forwardpad, inputs), axis=1)
        # backward pad
        inputs = tf.concat((inputs, self.backwardpad), axis=1)
        # local attention
        window_inputs = inputs[:, :3, :]
        window_attentions = self.attention(window_inputs)
        for line in range(1, 100):
            window_inputs = inputs[:, line:line + 3, :]
            window_attention = self.attention(window_inputs)
            window_attentions = tf.concat((window_attentions, window_attention), axis=1)
        # get local attention
        local_attention = window_attentions[:, 1::3, :]
        return local_attention

    def attention(self, inputs):
        # x->fx
        q = kb.dot(inputs, self.wq)
        k = kb.dot(inputs, self.wk)
        v = kb.dot(inputs, self.wv)
        # compute the attention score
        qk = kb.batch_dot(q, kb.permute_dimensions(k, [0, 2, 1]))
        attention_score = kb.softmax(qk)
        attention_result = kb.batch_dot(attention_score, v)
        return attention_result

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[1], 100)
        return output_shape
