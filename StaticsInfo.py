"""
统计MSRA语料数据
"""
# 创建字典
data_dict = dict()
data_dict['LOC'] = 0
data_dict['PER'] = 0
data_dict['ORG'] = 0
# 路径
data_dir = '../data/people_test_data.txt'
# 提取所有实体
entity = ''
test_entity = []
with open(data_dir, encoding='utf-8') as fo:
    for line in fo.readlines():
        if line != '\n':
            content = line.strip().split()
            content_label = content[-1]
            if content_label == 'O':
                if len(entity):
                    test_entity.append(entity)
                    entity = ''
                else:
                    continue
            elif content_label in ['B-ORG', 'B-PER', 'B-LOC']:
                if len(entity):
                    test_entity.append(entity)
                    entity = ''
                    entity += content[0]
                else:
                    entity += content[0]
            else:
                entity += content[0]
print("测试集实体总数", len(test_entity))
# 提取相应的实体
with open(data_dir, encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        if line != '\n':
            word, label = line.strip().split()
            if label == 'B-LOC':
                data_dict['LOC'] += 1
            elif label == 'B-ORG':
                data_dict['ORG'] += 1
            elif label == 'B-PER':
                data_dict['PER'] += 1
            else:
                continue
print(data_dict)
