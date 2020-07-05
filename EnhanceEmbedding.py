"""
@Author:luo ping
对位置编码进行单独的双向lstm
"""
from keras.layers import Layer, Bidirectional, LSTM
import pickle
import tensorflow as tf
import keras.backend as kb
import numpy as np


class EEmbedding(Layer):
    """
    @layer:EEmbedding
    """

    def __init__(self, **kwargs):
        super(EEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        @using: init weight
        """
        self.length = 100
        self.dim = 100
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
        pos_tensor = tf.zeros_like(inputs)
        position_encoding = np.zeros((self.length, self.dim))
        for pos in range(self.length):
            for i in range(self.dim):
                position_encoding[pos, i] = pos / np.power(10000, (i - i % 2) / self.dim)
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = tf.convert_to_tensor(position_encoding, 'float32')
        pos_tensor = pos_tensor + position_encoding
        # pos_tensor = Bidirectional(LSTM(50, return_sequences=True))(posm)
        inputs = kb.concatenate((inputs, pos_tensor), axis=-1)
        return inputs

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[1], 200)
        return output_shape
