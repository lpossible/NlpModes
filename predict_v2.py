"""
new version predict
"""
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.models import load_model
import numpy as np
from keras.preprocessing import sequence
from transformerLayer import Embedding, PositionEncoding, Attention, PositionWiseFeedForward, LayerNormalization
from EncoderLayer import Encoder
from PartMaskAttention import PartAttention
import os
from RelativePositionEncoding import RelativePositionMultiAttention

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
model = load_model("./model/encoder_6encoder.h5", custom_objects={"CRF": CRF, "crf_loss": crf_loss,
                                                                  "crf_viterbi_accuracy": crf_viterbi_accuracy,
                                                                  "Attention": Attention,
                                                                  "PositionEncoding": PositionEncoding,
                                                                  "Embedding": Embedding,
                                                                  'PositionWiseFeedForward': PositionWiseFeedForward,
                                                                  'LayerNormalization': LayerNormalization,
                                                                  'Encoder': Encoder,
                                                                  'PartAttention': PartAttention,
                                                                  'RelativePositionMultiAttention': RelativePositionMultiAttention})
special_words = ['<PAD>', '<UNK>']  # 特殊词表示
# 读取字符词典文件
with open('./data/vocab.txt', encoding="utf8") as fo:
    char_vocabs = [line.strip() for line in fo]
char_vocabs = special_words + char_vocabs

# 字符和索引编号对应,enumerate从0开始
id_to_vocab = {idx: char for idx, char in enumerate(char_vocabs)}
vocab_to_id = {char: idx for idx, char in id_to_vocab.items()}
# "BIO"标记的标签
label2idx = {"O": 0,
             "B_nsl": 1, "M_nsl": 2,
             "E_nsl": 3
             }
# 索引和BIO标签对应
idx2label = {idx: label for label, idx in label2idx.items()}


def read_corpus(corpus_path, vocab2idx):
    """read_corpus"""
    sentence = []
    data_sen = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    # 句子id列表
    data = []
    # 句子列表
    each_sen = []
    for line in lines:
        if line != '\n':
            char = line.strip().split()[0]
            # 数据字符，需要传入模型的
            data.append(char)
            # 用作匹配的字符序列
            each_sen.append(char)
        else:
            sent_ids = [vocab2idx[char] if char in vocab2idx else vocab2idx['<UNK>'] for char in data]
            sentence.append(sent_ids)
            data_sen.append(each_sen)
            data = []
            each_sen = []
    return sentence, data_sen


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


# 提取所有实体
entity = ''
test_entity = []
with open('./data/bme_test_data.txt', encoding='utf-8') as fo:
    for line in fo.readlines():
        if line != '\n':
            content = line.strip().split()
            if content[-1] != 'O':
                entity += content[0]
            else:
                if len(entity):
                    test_entity.append(entity)
                    entity = ''
                else:
                    continue
# 数据路径
data_path = './data/bme_test_data.txt'
# 数据转为id序列,并保存数据序列
id_data, sen_data = read_corpus(data_path, vocab_to_id)

# padding数据
MAX_LEN = 100
id_data = sequence.pad_sequences(id_data, MAX_LEN, padding='post')
mask_data = np.equal(id_data, 0)
mask_data = mask_data.astype(np.float)
# mask_data = process_mask(mask_data)
output = model.predict([id_data, mask_data])
# 提取标签
labels = []
for each in output:
    labels.append(np.argmax(each, axis=1))
# 识别实体数
entities = []
entity = ''
for i in range(len(labels)):
    for j in range(len(labels[0])):
        if labels[i][j] == 0:
            if len(entity):
                entities.append(entity)
                entity = ''
            else:
                continue
        elif idx2label[labels[i][j]] == 'B_nsl':
            if len(entity) == 0:
                entity += id_to_vocab[id_data[i][j]]
            else:
                entities.append(entity)
                entity = ''
                entity += id_to_vocab[id_data[i][j]]
        elif idx2label[labels[i][j]] == 'M_nsl':
            if len(entity) == 0:
                continue
            else:
                entity += id_to_vocab[id_data[i][j]]
        else:
            if len(entity) == 0:
                continue
            else:
                entity += id_to_vocab[id_data[i][j]]
                entities.append(entity)
                entity = ''
# 统计识别的实体数
model_entity = entities
# 统计准确率和召回率,F值
precision_num = 0
for each in model_entity:
    if each in test_entity:
        precision_num += 1
accuracy = float(precision_num) / len(model_entity)
recall = float(precision_num) / len(test_entity)
print('准确率为', accuracy)
print('召回率为', recall)
print('F值为', (2 * accuracy * recall) / (accuracy + recall))
