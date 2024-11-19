from cmapPy.pandasGEXpress.parse import parse
import pandas as pd

# 设置文件路径
gctx_path = "/home/cpu/study/DLEPS/data/GSE92742/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
sig_info_path = "/home/cpu/study/DLEPS/data/GSE92742/GSE92742_Broad_LINCS_sig_info.txt"  # sig_info 文件路径
output_path = "trichostatin_a_data.csv"


# 加载 sig_info
sig_info = pd.read_csv(sig_info_path, sep="\t")

# 筛选 pert_iname=trichostatin-a 的 sig_id
trichostatin_a_sigs = sig_info[sig_info["pert_iname"] == "trichostatin-a"][
    "sig_id"
].tolist()
print(f"找到 {len(trichostatin_a_sigs)} 个与 trichostatin-a 相关的 sig_id")

# 从 .gctx 文件中加载子集数据
gctx_data = parse(gctx_path, cid=trichostatin_a_sigs)

# 转换为 DataFrame
data_df = gctx_data.data_df

# 保存为 CSV 文件
data_df.to_csv(output_path)
print(f"相关数据已保存至 {output_path}")
