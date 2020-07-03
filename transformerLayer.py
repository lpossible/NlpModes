"""
@Author:luo ping
@NetworkLayers:Embedding,PositionEncoding,Attention,Add&Normalization,FeedForward
"""
from keras.layers import Layer
import pickle
import tensorflow as tf
import keras.backend as kb
import numpy as np


class Embedding(Layer):
    """
    @layer:Embedding
    """

    def __init__(self, **kwargs):
        super(Embedding, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        @using: init weight
        """
        with open('./data/word_vec.pkl', 'rb') as fr:
            self.embeddings = pickle.load(fr)
        self.embeddings = tf.convert_to_tensor(self.embeddings, dtype='float32')
        self.pad_special = self.add_weight(shape=(2, 100), initializer='glorot_uniform')
        self.embeddings = tf.concat((self.pad_special, self.embeddings), axis=0)
        self.built = True

    def call(self, inputs, **kwargs):
        """
        @using: compute
        :return: result
        """
        inputs = tf.gather(self.embeddings, inputs)
        return inputs

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[1], 100)
        return output_shape


class PositionEncoding(Layer):
    """
    位置编码层
    """

    def __init__(self, **kwargs):
        self.length = 0
        self.dim = 0
        super(PositionEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.length = input_shape[1]
        self.dim = input_shape[-1]

    def call(self, inputs, **kwargs):
        position_encoding = np.zeros((self.length, self.dim))
        for pos in range(self.length):
            for i in range(self.dim):
                position_encoding[pos, i] = pos / np.power(10000, (i - i % 2) / self.dim)
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = tf.convert_to_tensor(position_encoding, 'float32')
        inputs = inputs + position_encoding
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class Attention(Layer):
    """
    自注意力机制和残差连接
    """

    def __init__(self, attention_dim, encoder=None, masking=None, **kwargs):
        self._masking = masking
        self._attention_dim = attention_dim
        self.encoder = encoder
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.encoder:
            self.kernel = self.add_weight(name="kernel", shape=(2, input_shape[0][-1], self._attention_dim),
                                          initializer='glorot_uniform')
            self.v = self.add_weight(name='v', shape=(input_shape[0][-1], input_shape[0][-1]),
                                     initializer='glorot_uniform')
        else:
            self.kernel = self.add_weight(name="kernel", shape=(3, input_shape[0][-1], self._attention_dim),
                                          initializer='glorot_uniform')
        self.built = True

    def call(self, inputs, **kwargs):
        # Attention作为transformer编码层使用
        if self.encoder:
            if self._masking:
                assert len(inputs) == 2, 'input should be [inputs,masks]'
                inputs, masks = inputs
                wq = kb.dot(inputs, self.kernel[0])
                wk = kb.dot(inputs, self.kernel[1])
                wv = kb.dot(inputs, self.v)
                qk = kb.batch_dot(wq, kb.permute_dimensions(wk, [0, 2, 1]))
                qk /= np.sqrt(self._attention_dim)
                # 计算masks矩阵
                masks = kb.cast(masks, 'float32')
                masks = masks * (-2 ** 32 + 1)
                masks = kb.expand_dims(masks, 1)
                masks = kb.tile(masks, (1, 100, 1))
                qk = qk + masks
                qk = kb.softmax(qk)
                result = kb.batch_dot(qk, wv)
                # 进行残差连接
                output = inputs + result
                return output
            else:
                inputs, masks = inputs
                wq = kb.dot(inputs, self.kernel[0])
                wk = kb.dot(inputs, self.kernel[1])
                wv = kb.dot(inputs, self.v)
                qk = kb.batch_dot(wq, kb.permute_dimensions(wk, [0, 2, 1]))
                qk /= np.sqrt(self._attention_dim)
                qk = kb.softmax(qk)
                result = kb.batch_dot(qk, wv)
                output = inputs + result
                return output
        # Attention层作为经典BLSTM后接层使用
        else:
            if self._masking:
                assert len(inputs) == 2, 'input should be [inputs,masks]'
                inputs, masks = inputs
                wq = kb.dot(inputs, self.kernel[0])
                wk = kb.dot(inputs, self.kernel[1])
                wv = kb.dot(inputs, self.kernel[-1])
                qk = kb.batch_dot(wq, kb.permute_dimensions(wk, [0, 2, 1]))
                qk /= np.sqrt(self._attention_dim)
                # 计算masks矩阵
                masks = kb.cast(masks, 'float32')
                masks = masks * (-2 ** 32 + 1)
                masks = kb.expand_dims(masks, 1)
                masks = kb.tile(masks, (1, 100, 1))
                qk = qk + masks
                qk = kb.softmax(qk)
                output = kb.batch_dot(qk, wv)
                return output
            else:
                inputs, masks = inputs
                wq = kb.dot(inputs, self.kernel[0])
                wk = kb.dot(inputs, self.kernel[1])
                wv = kb.dot(inputs, self.kernel[2])
                qk = kb.batch_dot(wq, kb.permute_dimensions(wk, [0, 2, 1]))
                qk /= np.sqrt(self._attention_dim)
                qk = kb.softmax(qk)
                output = kb.batch_dot(qk, wv)
                return output

    def get_config(self):
        base_config = super(Attention, self).get_config()
        base_config['attention_dim'] = self._attention_dim
        base_config['masking'] = self._masking
        base_config['encoder'] = self.encoder
        return base_config

    def compute_output_shape(self, input_shape):
        if self.encoder:
            return input_shape
        else:
            output_shape = (input_shape[0][0], input_shape[0][1], self._attention_dim)
            return output_shape


class LayerNormalization(Layer):
    """
    对每个样本数据进行归一化处理
    """

    def __init__(self, **kwargs):
        self.eps = 1e-8
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=(input_shape[-1],), initializer='ones')
        self.beta = self.add_weight(name='beta', shape=(input_shape[-1],), initializer='zeros')
        self.built = True

    def call(self, inputs, **kwargs):
        mean, variance = tf.nn.moments(inputs, [1, 2], keepdims=True)
        # 归一化第一步
        inputs = (inputs - mean) / (variance + self.eps) ** 0.5
        # 归一化第二步
        output = self.gamma * inputs + self.beta
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


class PositionWiseFeedForward(Layer):
    """
    编码层前馈神经网络
    """

    def __init__(self, inner_dim, out_dim, **kwargs):
        self.inner_dim = inner_dim
        self.out_dim = out_dim
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w1 = self.add_weight(name='w1', shape=(input_shape[-1], self.inner_dim), initializer='glorot_uniform')
        self.bias1 = self.add_weight(name='bias1', shape=(self.inner_dim,), initializer='glorot_uniform')
        self.w2 = self.add_weight(name='w2', shape=(self.inner_dim, self.out_dim), initializer='glorot_uniform')
        self.bias2 = self.add_weight(name='bias2', shape=(self.out_dim,), initializer='glorot_uniform')
        self.built = True

    def call(self, inputs, **kwargs):
        inputs = kb.relu(kb.dot(inputs, self.w1) + self.bias1)
        inputs = kb.dot(inputs, self.w2) + self.bias2
        return inputs

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[1], self.out_dim)
        return output_shape

    def get_config(self):
        base_config = super(PositionWiseFeedForward, self).get_config()
        base_config['inner_dim'] = self.inner_dim
        base_config['out_dim'] = self.out_dim
        return base_config
