"""
@author:luo ping
@date:2020-6-7
@using:The encoder of transformer model
"""
from keras.layers import Layer
from transformerLayer import PositionWiseFeedForward, Attention, LayerNormalization


class Encoder(Layer):
    """
    transformer的编码层
    """

    def __init__(self, attention_dim, inner_dim, out_dim, **kwargs):
        self.attention_dim = attention_dim
        self.inner_dim = inner_dim
        self.out_dim = out_dim
        super(Encoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.stack = 1
        self.built = True

    def call(self, inputs, **kwargs):
        x, masks = inputs
        for i in range(self.stack):
            x = Attention(self.attention_dim, masking=True, encoder=True)([x, masks])
            x = LayerNormalization()(x)
            x = PositionWiseFeedForward(self.inner_dim, self.out_dim)(x)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][1], self.out_dim)
        return output_shape

    def get_config(self):
        base_config = super(Encoder, self).get_config()
        base_config['attention_dim'] = self.attention_dim
        base_config['inner_dim'] = self.inner_dim
        base_config['out_dim'] = self.out_dim
        return base_config
