# import pandas as pd
# from cmapPy.pandasGEXpress.parse import parse
# import matplotlib.pyplot as plt
# import seaborn as sns


# sig_info = pd.read_csv("/home/cpu/study/DLEPS/data/GSE92742/GSE92742_Broad_LINCS_sig_info.txt", sep="\t")
# print(sig_info.columns)

# gene_info = pd.read_csv(
#     "/home/cpu/study/DLEPS/data/GSE92742/GSE92742_Broad_LINCS_gene_info.txt",
#     sep="\t",
#     dtype=str,
# )
# print(gene_info.columns)

# drugX_ids = sig_info["sig_id"][sig_info["pert_iname"] == "trichostatin-a"]
# landmark_gene_ids = gene_info["pr_gene_id"][gene_info["pr_is_lm"] == "1"]
# print(drugX_ids)

# drugX_gctoo = parse(
#     "/home/cpu/study/DLEPS/data/GSE92742/GSE92742_Level4_ZSPCINF.mlr12k.gctx",
#     cid=drugX_ids,
# )
# print(drugX_gctoo.data_df.shape)  # 输出数据维度


import pandas as pd
from cmapPy.pandasGEXpress import parse

# 文件路径
gctx_file_path = "/home/cpu/study/DLEPS/data/GSE92742/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
sig_info_file_path = (
    "/home/cpu/study/DLEPS/data/GSE92742/GSE92742_Broad_LINCS_sig_info.txt"
)


# 加载 sig_info 文件
def load_sig_info(file_path):
    try:
        sig_info = pd.read_csv(file_path, sep="\t", encoding="utf-8", dtype=str)
        print(f"sig_info 文件加载成功，包含 {sig_info.shape[0]} 行数据")
        return sig_info
    except Exception as e:
        print("加载 sig_info 文件时出错：", e)
        return None


# 筛选 pert_iname = trichostatin-a 的 sig_id
def filter_by_pert_iname(sig_info, pert_iname_value):
    filtered = sig_info[sig_info["pert_iname"] == pert_iname_value]
    if filtered.empty:
        print(f"未找到 pert_iname = {pert_iname_value} 的记录")
        return None
    print(
        f"找到 {filtered.shape[0]} 个与 pert_iname = {pert_iname_value} 相关的 sig_id"
    )
    return filtered["sig_id"].tolist()


# 加载 GCTX 文件的子集
def load_gctx_subset(file_path, sig_ids):
    try:
        print(f"加载 GCTX 文件的子集，sig_id 数量：{len(sig_ids)}")
        gctoo = parse.parse(file_path, cid=sig_ids)
        print(
            f"GCTX 文件子集加载成功，包含 {gctoo.data_df.shape[0]} 行和 {gctoo.data_df.shape[1]} 列"
        )
        return gctoo.data_df
    except Exception as e:
        print("加载 GCTX 文件子集时出错：", e)
        return None


# 主函数
def main():
    # 加载 sig_info 文件
    sig_info = load_sig_info(sig_info_file_path)
    if sig_info is None:
        return

    # 筛选与 trichostatin-a 相关的 sig_id
    pert_iname_value = "trichostatin-a"
    sig_ids = filter_by_pert_iname(sig_info, pert_iname_value)
    if sig_ids is None:
        return

    # 加载 GCTX 数据子集
    gctx_subset = load_gctx_subset(gctx_file_path, sig_ids)
    if gctx_subset is not None:
        # 输出子集数据示例
        print("GCTX 数据子集示例：")
        print(gctx_subset.head())


# 运行主函数
if __name__ == "__main__":
    main()
