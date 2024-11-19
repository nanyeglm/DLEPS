"""解析独一无二的数据"""

import pandas as pd

# 加载 sig_info 文件
sig_info_path = "/home/cpu/study/DLEPS/data/GSE92742/GSE92742_Broad_LINCS_sig_info.txt"
sig_info = pd.read_csv(sig_info_path, sep="\t", dtype=str)


# 查看文件基本信息
print(sig_info.head())
print(sig_info["sig_id"].nunique())  # 唯一 sig_id 数量
print(sig_info["pert_id"].nunique())  # 唯一 pert_id 数量
print(sig_info["pert_iname"].nunique())  # 唯一 pert_iname 数量
