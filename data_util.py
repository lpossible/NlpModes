"""
@author:luo Ping
@date:2020-5-30
@school:LzJt university
@using:date processing and train model
"""
import pickle
import numpy as np
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM, TimeDistributed, Dense, Dropout
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from transformerLayer import Embedding, PositionEncoding, Attention, LayerNormalization, PositionWiseFeedForward
from keras.callbacks import EarlyStopping
from EncoderLayer import Encoder
from PartMaskAttention import PartAttention
from RelativePositionEncoding import RelativePositionMultiAttention
from EnhanceEmbedding import EEmbedding
import os
import tensorflow as tf

tf.set_random_seed(4)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 测试数据路径
test_data_path = './data/bme_test_data.txt'
# 训练数据路径
train_data_path = './data/bme_train_data.txt'
# 词典数据路径
vocab_path = './data/vocab.txt'
# 词向量数据路径
word2vec_data_path = './data/word2vec.txt'


# 开始数据处理
def get_data(data_path, word_dict=None, label_dict=None, mode=None):
    # 处理词向量数据
    if mode == 'vec':
        words_vec = []
        with open(data_path, encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                if line != '\n':
                    word_vec = line.strip().split()[1:]
                    words_vec.append(word_vec)
        all_vec = list()
        for each in words_vec:
            each_vec = []
            for char in each:
                each_vec.append(eval(char))
            all_vec.append(each_vec)
        all_vec = np.asarray(all_vec)
        with open('./data/word_vec.pkl', 'wb') as fw:
            pickle.dump(all_vec, fw)
        return True
    # 建立词典
    elif mode == 'vocab':
        word_list = list()
        with open(data_path, encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                if line != '\n':
                    word_list.append(line.strip())
        # 特殊词
        special_word = ['pad', 'unknown']
        # 对词典表进行汇总
        word_list = special_word + word_list
        word_dict = dict()
        for key, value in enumerate(word_list):
            word_dict[value] = key
        return word_dict
    # 处理训练和测试数据
    else:
        data, labels = [], []
        with open(data_path, encoding='utf-8') as fr:
            lines = fr.readlines()
        sequence, tag = [], []
        for line in lines:
            if line != '\n':
                [char, label] = line.strip().split()
                sequence.append(char)
                tag.append(label)
            else:
                sequence_ids = [word_dict[char] if char in word_dict else word_dict['unknown'] for char in sequence]
                tag_ids = [label_dict[label] for label in tag]
                data.append(sequence_ids)
                labels.append(tag_ids)
                sequence, tag = [], []
        return data, labels


# 处理词向量数据
vec_data_processing = get_data(word2vec_data_path, mode='vec')
# 建立中文字词典
words_dict = get_data(vocab_path, mode='vocab')
# 建立标签词典
labels_dict = {"O": 0,
               "B_nsl": 1, "M_nsl": 2,
               "E_nsl": 3
               }
# 得到训练数据和测试数据
train_data, train_labels = get_data(train_data_path, word_dict=words_dict, label_dict=labels_dict)
test_data, test_labels = get_data(test_data_path, word_dict=words_dict, label_dict=labels_dict)
# padding 数据
train_data = sequence.pad_sequences(train_data, maxlen=100, padding='post')
train_labels = sequence.pad_sequences(train_labels, maxlen=100, padding='post')
test_data = sequence.pad_sequences(test_data, maxlen=100, padding='post')
test_labels = sequence.pad_sequences(test_labels, maxlen=100, padding='post')
# 将标签转换为one-hot编码
train_labels = to_categorical(train_labels, len(labels_dict))
test_labels = to_categorical(test_labels, len(labels_dict))
# 建立mask矩阵
train_mask_data = np.equal(train_data, 0)
test_mask_data = np.equal(test_data, 0)
train_mask_data = train_mask_data.astype(np.float32)
test_mask_data = test_mask_data.astype(np.float32)


# 建立部分mask数据矩阵
def process_mask(mask_data):
    """
    将一维的mask向量根据pad的长度转化为二维的mask矩阵
    :return:
    """
    # 记录位置
    pos = 0
    # 所有数据
    data = list()
    for each in mask_data:
        for i in range(100):
            if each[i] == 1:
                pos = i
                break
        each_data = list()
        for j in range(pos):
            each_data.append(each)
        for k in range(100 - pos):
            each_data.append(1 - each)
        each_data = np.asarray(each_data)
        data.append(each_data)
    data = np.asarray(data)
    return data


# 部分mask数据
part_mask_train_data = process_mask(train_mask_data)
part_mask_test_data = process_mask(test_mask_data)
# 建立监听
early_stop = EarlyStopping(patience=5)
# 建立模型
inputs = Input(shape=(100,), dtype='int32', name='inputs')
masks = Input(shape=(100,), dtype='float32', name='masks')
x = EEmbedding()(inputs)
# x = Embedding()(inputs)
x = Dropout(rate=0.2)(x)
# x = PositionEncoding()(x)
# x = Encoder(attention_dim=128, inner_dim=128, out_dim=128)([x, masks])
x = Bidirectional(LSTM(128, return_sequences=True))(x)
# x = Encoder(attention_dim=128, inner_dim=128, out_dim=128)([x, masks])
# x = RelativePositionMultiAttention()([x, masks])
x = Dropout(rate=0.2)(x)
# x = PartAttention(attention_dim=128)([x, masks])
# x = Attention(attention_dim=128, masking=True, encoder=False)([x, masks])
for i in range(6):
    x = Attention(attention_dim=128, masking=True, encoder=True)([x, masks])
    x = LayerNormalization()(x)
    x = PositionWiseFeedForward(inner_dim=128, out_dim=128)(x)
    x = LayerNormalization()(x)
# x = Encoder(attention_dim=128, inner_dim=128, out_dim=128)([x, masks])
x = Dense(len(labels_dict))(x)
outputs = CRF(len(labels_dict))(x)
model = Model(inputs=[inputs, masks], outputs=outputs)
model.compile(loss=crf_loss, optimizer='adam', metrics=[crf_viterbi_accuracy])
model.summary()
model.fit(x=[train_data, train_mask_data], y=train_labels, epochs=100, validation_split=0.2, callbacks=[early_stop])

# 模型评估
score = model.evaluate(x=[test_data, test_mask_data], y=test_labels, batch_size=32)
print(score)
# 模型保存
model.save('./model/ee_dp_bilstm_dp_tr.h5')
