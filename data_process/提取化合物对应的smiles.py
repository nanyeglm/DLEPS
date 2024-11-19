import pandas as pd
import requests
import time

# 文件路径
input_path = "/home/cpu/study/DLEPS/data/GSE92742/GSE92742_Broad_LINCS_sig_info.txt"
output_path = (
    "/home/cpu/study/DLEPS/data/GSE92742/GSE92742_Broad_LINCS_sig_info_with_smiles.csv"
)
batch_size = 100  # 每批次查询的记录数


# 定义函数：通过 PubChem 查询 SMILES
def get_smiles_from_pubchem(pert_iname):
    """
    根据化合物名称查询 PubChem API，获取 SMILES。
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(base_url.format(pert_iname))
        response.raise_for_status()  # 检查响应状态
        data = response.json()
        smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        return smiles
    except Exception as e:
        # 捕获异常并返回 None
        print(f"查询 {pert_iname} 的 SMILES 失败：{e}")
        return None


# 加载 sig_info 文件
data = pd.read_csv(input_path, sep="\t")

# 创建新列，用于存储 SMILES，如果已存在则跳过
if "smiles" not in data.columns:
    data["smiles"] = None

# 遍历每个化合物名称（pert_iname），分批处理
start_time = time.time()
for start_idx in range(0, len(data), batch_size):
    end_idx = start_idx + batch_size
    batch = data.iloc[start_idx:end_idx]
    print(f"正在处理第 {start_idx} - {end_idx} 条记录...")

    for idx, row in batch.iterrows():
        if pd.notna(row["smiles"]):
            # 如果 SMILES 已存在，跳过
            continue

        pert_iname = row["pert_iname"]
        print(f"正在查询 {pert_iname} 的 SMILES...")

        # 查询 SMILES
        smiles = get_smiles_from_pubchem(pert_iname)
        if smiles:
            data.at[idx, "smiles"] = smiles
            print(f"找到 SMILES：{smiles}")
        else:
            print(f"未找到 {pert_iname} 的 SMILES")

        # 添加延时，避免触发 API 限制
        time.sleep(0.2)

    # 定期保存结果以防数据丢失
    print(f"正在保存第 {start_idx} - {end_idx} 条记录的进度...")
    data.to_csv(output_path, index=False)

end_time = time.time()
print(f"处理完成，总耗时 {end_time - start_time:.2f} 秒。结果已保存到 {output_path}")
