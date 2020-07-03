"""
@author:luo ping
@date:2020-7-1
"""
import math
import numpy as np
from keras.layers import Layer
import keras.backend as kb
import tensorflow as tf


class RelativePositionMultiAttention(Layer):
    """"
    默认分为两个头
    """

    def __init__(self, **kwargs):
        super(RelativePositionMultiAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 相对位置编码
        batch_size, seq_len, embedding_dim = input_shape[0]
        half_dim = embedding_dim // 2
        self.half_dim = half_dim
        emb = math.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * -emb)
        emb = np.expand_dims(np.arange(-seq_len // 2, seq_len // 2), 1) * np.expand_dims(emb, 0)
        emb = np.concatenate((np.sin(emb), np.cos(emb)), axis=-1).reshape(seq_len, -1)
        self.emb = tf.convert_to_tensor(emb, dtype='float32')
        # 初始化u和v
        self.u = self.add_weight(name='u_weight', shape=(2, half_dim), initializer='glorot_uniform')
        self.v = self.add_weight(name='v_weight', shape=(2, half_dim), initializer='glorot_uniform')
        # 初始化Wq和Wv
        self.Wq = self.add_weight(name='Wq', shape=(input_shape[0][-1], half_dim), initializer='glorot_uniform')
        self.Wv = self.add_weight(name='Wv', shape=(input_shape[0][-1], half_dim), initializer='glorot_uniform')
        self.built = True

    def call(self, inputs, **kwargs):
        inputs, masks = inputs
        K1, K2 = tf.split(inputs, 2, axis=-1)
        pos_emb1, pos_emb2 = tf.split(self.emb, 2, axis=-1)
        Q = kb.dot(inputs, self.Wq)
        V = kb.dot(inputs, self.Wv)
        # 计算attention1
        attention1_1part = kb.batch_dot(Q, kb.permute_dimensions(K1, [0, 2, 1]))
        attention1_2part = kb.dot(Q, kb.permute_dimensions(pos_emb1, [1, 0]))
        attention1_3part = kb.dot(K1, tf.reshape(self.u[0], shape=(self.half_dim, 1)))
        attention1_3part = kb.permute_dimensions(attention1_3part, [0, 2, 1])
        attention1_4part = kb.dot(pos_emb1, tf.reshape(self.v[0], shape=(self.half_dim, 1)))
        attention1_4part = kb.permute_dimensions(attention1_4part, [1, 0])
        attention1 = attention1_1part + attention1_2part + attention1_3part + attention1_4part
        # 计算attention2
        attention2_1part = kb.batch_dot(Q, kb.permute_dimensions(K2, [0, 2, 1]))
        attention2_2part = kb.dot(Q, kb.permute_dimensions(pos_emb2, [1, 0]))
        attention2_3part = kb.dot(K2, tf.reshape(self.u[1], shape=(self.half_dim, 1)))
        attention2_3part = kb.permute_dimensions(attention2_3part, [0, 2, 1])
        attention2_4part = kb.dot(pos_emb2, tf.reshape(self.v[1], shape=(self.half_dim, 1)))
        attention2_4part = kb.permute_dimensions(attention2_4part, [1, 0])
        attention2 = attention2_1part + attention2_2part + attention2_3part + attention2_4part
        attention = attention1 + attention2
        # 计算masks矩阵
        masks = kb.cast(masks, 'float32')
        masks = masks * (-2 ** 32 + 1)
        masks = kb.expand_dims(masks, 1)
        masks = kb.tile(masks, (1, 100, 1))
        qk = attention + masks
        qk = kb.softmax(qk)
        result = kb.batch_dot(qk, V)
        # 链接
        result = kb.concatenate((inputs, result), axis=-1)
        return result

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][1], input_shape[0][-1] + self.half_dim)
        return output_shape
