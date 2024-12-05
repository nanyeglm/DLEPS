# /mnt/d/Research/PHD/DLEPS/code/DLEPS/DLEPS_tutorial.py

# 导入所需模块
import os
import pandas as pd
import matplotlib.pyplot as plt

from dleps_predictor import DLEPS

# 加载天然产物和 FDA 批准药物的 SMILES 数据
smi = pd.read_csv('/mnt/d/Research/PHD/DLEPS/data/Brief_Targetmol_natural_product_2719')
fda = pd.read_csv('/mnt/d/Research/PHD/DLEPS/data/Brief_FDA-Approved-Drug_961')

# 查看天然产物数据
print("天然产物数据预览：")
print(smi.head())

# 初始化 DLEPS 模型
# 指定模型权重路径为训练后生成的模型权重文件
predictor = DLEPS(
    reverse=False, 
    up_name='/mnt/d/Research/PHD/DLEPS/data/BROWNING_up',
    down_name='/mnt/d/Research/PHD/DLEPS/data/BROWNING_down',
    save_exp=None,
    model_weights_path='/mnt/d/Research/PHD/DLEPS/my_trained_model.h5'  # 指定训练后保存的模型权重文件路径
)

# 查看模型结构
predictor.model[0].summary()

# 模型推理
# 输入：SMILES 数组
# 输出：cs 数组，其中 -2 表示处理失败的小分子
scores = predictor.predict(fda['SMILES'].values)

# 打印 FDA 批准药物数据
print("FDA 批准药物数据预览：")
print(fda.head())

# 可视化 cs 得分
plt.figure(figsize=(10, 6))
plt.hist(scores, bins=50, color='skyblue', edgecolor='black')
plt.title("FDA Approved Drugs CS Scores")
plt.xlabel("CS Scores")
plt.ylabel("Frequency")
plt.show()

# 将预测得分添加到 FDA 数据中
fda['score'] = scores

# 查看更新后的 FDA 数据
print("更新后的 FDA 批准药物数据预览：")
print(fda.head())

# 如果需要保存结果到文件，可以取消以下注释：
# fda = fda.set_index('Unnamed: 0')
# fda.to_csv('../../results/fda_HUA_merge.csv')
# print("FDA 预测结果已保存到 '../../results/fda_HUA_merge.csv'")

# 对天然产物数据进行预测
smi_scores = predictor.predict(smi['SMILES'].values)

# 可视化天然产物的 cs 得分
plt.figure(figsize=(10, 6))
plt.hist(smi_scores, bins=50, color='lightgreen', edgecolor='black')
plt.title("Natural Products CS Scores")
plt.xlabel("CS Scores")
plt.ylabel("Frequency")
plt.show()

# 将预测得分添加到天然产物数据中
smi['score'] = smi_scores

# 设置天然产物数据的索引
# 假设 'Unnamed: 0' 是需要设置为索引的列名，根据实际数据调整
# 如果 'Unnamed: 0' 不存在，请更改为实际的列名或移除此行
smi = smi.set_index('Unnamed: 0')

# 打印更新后的天然产物数据
print("更新后的天然产物数据预览：")
print(smi.head())

# 如果需要保存结果到文件，可以取消以下注释：
# smi.to_csv('../../results/natural_product_scores.csv')
# print("天然产物预测结果已保存到 '../../results/natural_product_scores.csv'")
