"""
@author:luo ping
@date:2020-7-6
@using:word vec training
"""
from gensim.models import Word2Vec

# 语料路径
train_data_path = './data/people_train_data.txt'
test_data_path = './data/people_test_data.txt'
# 数据序列
data = []
# 单句
sentence = []
# 对于已分词且标注的语料，需要形成句子序列
# 提取训练语料数据
with open(train_data_path, encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        if line != '\n':
            sentence.append(line.strip().split()[0])
        else:
            data.append(sentence)
            sentence = []
# 提取测试语料数据
sentence = []
with open(test_data_path, encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        if line != '\n':
            sentence.append(line.strip().split()[0])
        else:
            data.append(sentence)
            sentence = []
model = Word2Vec(data, min_count=2)
model.train(data, total_examples=len(data), epochs=32)
model.wv.save_word2vec_format(fname='./data/people_wordvec.txt', binary=False)
model.save('./model/wordvec.w2v')
