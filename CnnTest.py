"""
@author:Luo Ping
@date:2020-7-28
"""
import os
import tensorflow as tf
import pickle
import numpy as np
from keras.layers import Input, Conv1D, Dense, Bidirectional, LSTM
from keras.preprocessing import sequence
from keras.utils import to_categorical
from transformerLayer import Embedding
from keras_contrib.layers import CRF
from keras.models import Model
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from LocalAttention import LocalAttention

tf.set_random_seed(4)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 测试数据路径
test_data_path = './data/people_test_data.txt'
# 训练数据路径
train_data_path = './data/people_train_data.txt'
# 词典数据路径
vocab_path = './data/people_vocab.txt'
# 词向量数据路径
word2vec_data_path = './data/people_wordvec.txt'
# 词向量pkl路径
word_vec_pkl = './data/people_word_vec.pkl'


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
        with open(word_vec_pkl, 'wb') as fw:
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
               "B-ORG": 1, "I-ORG": 2,
               "B-PER": 3, 'I-PER': 4,
               'B-LOC': 5, 'I-LOC': 6
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
# 建立模型
inputs = Input(shape=(100,), dtype='int32')
x = Embedding()(inputs)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = LocalAttention()(x)
x = Dense(len(labels_dict))(x)
outputs = CRF(len(labels_dict))(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss=crf_loss, optimizer='adam', metrics=[crf_viterbi_accuracy])
model.fit(x=[train_data[:50000]], y=train_labels[:50000], epochs=1, validation_split=0.2, batch_size=16)
# 模型评估
score = model.evaluate(x=[test_data[:4624]], y=test_labels[:4624], batch_size=16)
print(score)
model.save('./model/people_em_bilstm_localatt.h5')
