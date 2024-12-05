########################################################
# All rights reserved. 
# Author: XIE Zhengwei @ Beijing Gigaceuticals Tech Co., Ltd 
#                      @ Peking University International Cancer Institute
# Contact: xiezhengwei@gmail.com
#
#
########################################################

import sys
sys.path.append('/mnt/d/Research/PHD/DLEPS/code/DLEPS')
import molecule_vae
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Draw

import numpy as np  
import pandas as pd
from collections import Counter
import h5py
import zinc_grammar
import nltk
from functools import reduce

# 1. 读取训练和测试SMILES数据
dt1 = pd.read_csv('/mnt/d/Research/PHD/DLEPS/results/train_SMILES_demo.csv')
dt2 = pd.read_csv('/mnt/d/Research/PHD/DLEPS/results/test_SMILES_demo.csv')

# 合并SMILES数组
smiles_array = np.concatenate([dt1['SMILES'].values, dt2['SMILES'].values], axis=0)
print("SMILES数组的总数:", len(smiles_array))

# 2. 读取L1000.csv数据
L1000 = pd.read_csv('/mnt/d/Research/PHD/DLEPS/results/L1000.csv')
print("L1000.csv中的SMILES总数:", len(L1000))

# 3. 检查并处理重复的SMILES
# 检查SMILES数组中的重复项
smiles_counts = Counter(smiles_array)
duplicates = [smile for smile, count in smiles_counts.items() if count > 1]
if duplicates:
    print("在SMILES数组中发现重复项:")
    print("重复的SMILES数量:", len(duplicates))
    print("重复的SMILES:", duplicates)
else:
    print("SMILES数组中没有重复项。")

# 检查L1000.csv中的重复项
L1000_smiles_counts = Counter(L1000['smiles'])
L1000_duplicates = [smile for smile, count in L1000_smiles_counts.items() if count > 1]
if L1000_duplicates:
    print("在L1000.csv的'smiles'列中发现重复项:")
    print("重复的SMILES数量:", len(L1000_duplicates))
    print("重复的SMILES:", L1000_duplicates)
else:
    print("L1000.csv的'smiles'列中没有重复项。")

# 4. 创建'occurrence'列，确保每个重复的SMILES都有唯一的标识
# 创建SMILES DataFrame，并添加唯一的出现次数索引
smiles_df = pd.DataFrame({'smiles': smiles_array})
smiles_df['occurrence'] = smiles_df.groupby('smiles').cumcount()
smiles_df['order'] = smiles_df.index

# 在L1000中创建'occurrence'列
L1000['occurrence'] = L1000.groupby('smiles').cumcount()

# 5. 合并数据，根据'smiles'和'occurrence'进行匹配
merged_df = pd.merge(smiles_df, L1000, on=['smiles', 'occurrence'], how='left')

# 6. 检查并处理缺失数据
missing_smiles = merged_df[merged_df.isnull().any(axis=1)]
if len(missing_smiles) > 0:
    print("以下SMILES在L1000.csv中未找到，对应的'occurrence'为:")
    print(missing_smiles[['smiles', 'occurrence']])
    # 删除缺失的行
    merged_df = merged_df.dropna()
    print("删除缺失数据后，剩余的行数:", len(merged_df))
else:
    print("所有SMILES都在L1000.csv中找到了匹配。")

# 7. 根据'order'排序，确保顺序与原始的'smiles'数组一致
merged_df = merged_df.sort_values('order')

# 8. 删除不需要的列，保留SMILES和基因表达数据
final_df = merged_df.drop(['occurrence', 'order'], axis=1)

# 更新smiles_array，使用对齐后的SMILES
smiles_array = final_df['smiles'].values

# 9. 使用RDKit处理SMILES字符串
smiles_rdkit = []
iid = []
for i in range(len(smiles_array)):
    try:
        smiles_rdkit.append(MolToSmiles(MolFromSmiles(smiles_array[i])))
        iid.append(i)
    except:
        print("Error at %d" % (i))

print("成功处理的SMILES数量:", len(smiles_rdkit))

# 10. 定义ZINC语法的tokenizer函数
def xlength(y):
    return reduce(lambda sum, element: sum + 1, y, 0)

def get_zinc_tokenizer(cfg):
    long_tokens = [a for a in list(cfg._lexical_index.keys()) if xlength(a) > 1]
    replacements = ['$','%','^']  # 可以根据需要添加更多替换符号
    assert len(long_tokens) == len(replacements)
    for token in replacements: 
        assert token not in cfg._lexical_index
    
    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        tokens = []
        for char in smiles:
            try:
                ix = replacements.index(char)
                tokens.append(long_tokens[ix])
            except:
                tokens.append(char)
        return tokens
    
    return tokenize

# 11. 准备语法解析器和映射
_tokenize = get_zinc_tokenizer(zinc_grammar.GCFG)
_parser = nltk.ChartParser(zinc_grammar.GCFG)
_productions = zinc_grammar.GCFG.productions()
_prod_map = {}
for ix, prod in enumerate(_productions):
    _prod_map[prod] = ix
MAX_LEN = 277
_n_chars = len(_productions)

# 12. 对SMILES进行解析和编码
assert type(smiles_rdkit) == list
tokens = list(map(_tokenize, smiles_rdkit))
parse_trees = []
i = 0
badi = []
for t in tokens:
    try:
        tp = next(_parser.parse(t))
        parse_trees.append(tp)
    except:
        print("Parse tree error at %d" % i)
        badi.append(i)
    i += 1

# 13. 生成one-hot编码
productions_seq = [tree.productions() for tree in parse_trees]
indices = [np.array([_prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
one_hot = np.zeros((len(indices), MAX_LEN, _n_chars), dtype=np.float32)
for i in range(len(indices)):
    num_productions = len(indices[i])
    if num_productions > MAX_LEN:
        print("Too large molecules, out of range at index %d" % i)
        one_hot[i][np.arange(MAX_LEN), indices[i][:MAX_LEN]] = 1.
    else:    
        one_hot[i][np.arange(num_productions), indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN), -1] = 1.

# 14. 获取成功解析的SMILES的索引
iix = np.arange(len(smiles_rdkit))
iidx = [i for i in iix if i not in badi]
print("成功解析的SMILES数量:", len(iidx))
iid2 = np.array(iid)[iidx]

# 15. 过滤one_hot和final_df，只保留成功解析的SMILES
one_hot_filtered = one_hot[iidx]
final_df_filtered = final_df.iloc[iidx].reset_index(drop=True)

# 16. 随机打乱数据
num_examples = one_hot_filtered.shape[0]
perm = np.arange(num_examples)
np.random.shuffle(perm)
one_hot_shuffled = one_hot_filtered[perm]
final_df_shuffled = final_df_filtered.iloc[perm].reset_index(drop=True)

# 17. 划分训练集和测试集
TEST_SIZE = 3000  # 可以根据需要调整测试集大小
smile_train = one_hot_shuffled[TEST_SIZE:]
smile_test = one_hot_shuffled[:TEST_SIZE]

L1000_train = final_df_shuffled.iloc[TEST_SIZE:].reset_index(drop=True)
L1000_test = final_df_shuffled.iloc[:TEST_SIZE].reset_index(drop=True)

# 18. 保存SMILES的训练和测试数据
h5f = h5py.File('SMILE_train_demo.h5', 'w')
h5f.create_dataset('data', data=smile_train)
h5f.close()

h5f = h5py.File('SMILE_test_demo.h5', 'w')
h5f.create_dataset('data', data=smile_test)
h5f.close()

# 19. 保存L1000的训练和测试数据
# 去除'smiles'列，只保留基因表达数据
L1000_train_data = L1000_train.drop('smiles', axis=1)
L1000_test_data = L1000_test.drop('smiles', axis=1)

# 保存为CSV文件
L1000_train_data.to_csv('L1000_train.csv', index=False)
L1000_test_data.to_csv('L1000_test.csv', index=False)

print("数据处理完成，训练和测试数据已保存。")
