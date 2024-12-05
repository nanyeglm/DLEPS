import os
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from tqdm import tqdm  # 导入 tqdm


def read_sig_info(sig_info_path):
    """
    读取 sig_info 文件，并提取 canonical_smiles 映射关系。
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始读取 sig_info 文件: {sig_info_path}")
    try:
        sig_info = pd.read_csv(sig_info_path, sep="\t")
        # 提取 pert_id 到 canonical_smiles 的唯一映射
        smiles_mapping = sig_info[['pert_id', 'canonical_smiles']].drop_duplicates()
        smiles_mapping = smiles_mapping.set_index('pert_id')['canonical_smiles'].to_dict()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功读取 sig_info 文件，共包含 {len(sig_info)} 条记录。")
        return sig_info, smiles_mapping
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 读取 sig_info 文件时出错: {e}", file=sys.stderr)
        return None, None


def group_by_pert_id(sig_info_df):
    """
    按照 pert_id 分组，获取每个 pert_id 对应的所有 sig_id。
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始按 'pert_id' 分组.")
    grouped = sig_info_df.groupby('pert_id')['sig_id'].apply(list).to_dict()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 共分组 {len(grouped)} 个不同的 'pert_id'。")
    return grouped


def process_pert_id(pert_id, sig_ids, gctx_path):
    """
    处理单个 pert_id 的数据提取和计算任务。
    """
    start_time = time.time()
    try:
        # 提取对应的 sig_id 数据
        gctx_data = parse(gctx_path, cid=sig_ids)
        data_df = gctx_data.data_df  # 行：基因，列：sig_id

        # 计算每行（基因）的平均值
        gene_means = data_df.mean(axis=1)
        gene_means.name = pert_id  # 设置 Series 名称为 pert_id

        # 打印一些统计信息
        elapsed = time.time() - start_time
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 完成 'pert_id' {pert_id} 的处理。"
            f" 数据维度: {data_df.shape}，耗时: {elapsed:.2f} 秒。"
        )
        return gene_means

    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 错误: 处理 'pert_id' {pert_id} 时出错: {e}", file=sys.stderr)
        return None


def extract_and_compute_averages_parallel(gctx_path, grouped_data, smiles_mapping, max_workers=4):
    """
    使用并行处理提取和计算平均值，并一次性合并所有结果。
    集成 tqdm 进度条以显示处理进度。
    """
    combined_df = None
    total_pert_ids = len(grouped_data)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始并行提取表达数据并计算平均值，共 {total_pert_ids} 个 'pert_id'。")

    gene_means_list = []  # 用于收集所有 gene_means
    smiles_list = []      # 用于收集对应的 smiles

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_pert_id, pert_id, sig_ids, gctx_path): pert_id
            for pert_id, sig_ids in grouped_data.items()
        }

        # 使用 tqdm 包装 as_completed 生成器
        for future in tqdm(as_completed(futures), total=total_pert_ids, desc="Processing pert_ids"):
            pert_id = futures[future]
            try:
                result = future.result()
                if result is not None:
                    smiles = smiles_mapping.get(pert_id, pert_id)  # 获取对应的 canonical_smiles
                    gene_means_list.append(result)
                    smiles_list.append(smiles)
            except Exception as e:
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 错误: 并行任务处理 'pert_id' {pert_id} 时出错: {e}",
                    file=sys.stderr,
                )

    if gene_means_list:
        # 将所有 gene_means 合并为一个 DataFrame
        combined_df = pd.concat(gene_means_list, axis=1)
        combined_df.columns = smiles_list  # 替换列名为 canonical_smiles
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 并行处理完成。")
        print(f"    综合 DataFrame 的维度: {combined_df.shape}")
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 没有有效的 gene_means 结果。", file=sys.stderr)

    return combined_df


def save_combined_data(combined_df, output_path):
    """
    将综合 DataFrame 保存为 CSV 文件，并转置。
    """
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始保存综合数据到 CSV 文件: {output_path}")
    try:
        # 转置 DataFrame，使 canonical_smiles 成为行，基因 ID 成为列
        combined_df = combined_df.T
        # 重置索引，将 canonical_smiles 作为一列
        combined_df.reset_index(inplace=True)
        # 重命名列名
        combined_df.rename(columns={'index': 'smiles'}, inplace=True)
        # 保存为 CSV 文件，不保存索引
        combined_df.to_csv(output_path, index=False)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 综合数据已成功保存至: {output_path}")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 保存综合数据时出错: {e}", file=sys.stderr)


def main():
    # 设置文件路径
    sig_info_path = "/mnt/d/Research/PHD/DLEPS/results/GSE92742_Broad_LINCS_sig_info_final.txt"
    gctx_path = "/mnt/d/Research/PHD/DLEPS/data/GSE92742/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
    output_combined_path = "/mnt/d/Research/PHD/DLEPS/results/combined_pert_id_averages.csv"

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 脚本开始运行.")
    print(f"    sig_info 文件路径: {sig_info_path}")
    print(f"    .gctx 文件路径: {gctx_path}")
    print(f"    输出文件路径: {output_combined_path}")

    # 1. 读取 sig_info 文件
    sig_info_df, smiles_mapping = read_sig_info(sig_info_path)
    if sig_info_df is None or smiles_mapping is None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 无法读取 sig_info 文件，脚本终止。", file=sys.stderr)
        return

    # 2. 按 pert_id 分组
    grouped_data = group_by_pert_id(sig_info_df)

    # 3. 使用并行提取表达数据并计算平均值
    combined_df = extract_and_compute_averages_parallel(gctx_path, grouped_data, smiles_mapping, max_workers=32)  # 根据CPU核数调整 max_workers

    if combined_df is not None and not combined_df.empty:
        # 4. 保存综合数据
        save_combined_data(combined_df, output_combined_path)
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 没有数据可保存。", file=sys.stderr)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 脚本运行结束.")


if __name__ == "__main__":
    main()
