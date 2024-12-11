# make_zinc_dataset_grammar.py
import nltk
import numpy as np
import h5py
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import molecule_vae
import zinc_grammar

# 读取SMILES数据
with open('data/250k_rndm_zinc_drugs_clean.smi', 'r') as f:
    L = [line.strip() for line in f]

MAX_LEN = 277
NCHARS = len(zinc_grammar.GCFG.productions())
output_dir = "temp_chunks"
os.makedirs(output_dir, exist_ok=True)  # 创建存放临时文件的目录

# 将单条SMILES转换为One-hot编码
def to_one_hot_chunk(smiles_chunk, idx):
    prod_map = {prod: ix for ix, prod in enumerate(zinc_grammar.GCFG.productions())}
    tokenize = molecule_vae.get_zinc_tokenizer(zinc_grammar.GCFG)
    parser = nltk.ChartParser(zinc_grammar.GCFG)
    one_hot = np.zeros((len(smiles_chunk), MAX_LEN, NCHARS), dtype=np.float32)
    failed_smiles = []

    for i, smiles in enumerate(smiles_chunk):
        try:
            tokens = tokenize(smiles)
            parse_tree = next(parser.parse(tokens))
            productions_seq = parse_tree.productions()
            indices = np.array([prod_map[p] for p in productions_seq], dtype=int)
            num_productions = len(indices)
            one_hot[i][np.arange(num_productions), indices] = 1.
            one_hot[i][np.arange(num_productions, MAX_LEN), -1] = 1.
        except Exception:
            failed_smiles.append(smiles)

    # 保存结果到临时文件
    temp_file = os.path.join(output_dir, f"chunk_{idx}.h5")
    with h5py.File(temp_file, 'w') as temp_h5:
        temp_h5.create_dataset('data', data=one_hot)
    return temp_file, failed_smiles

# 分块策略
batch_size = 50
num_batches = len(L) // batch_size + int(len(L) % batch_size != 0)
smiles_chunks = [L[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

failed_smiles_all = []
temp_files = []

# 并行处理并保存临时文件
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(to_one_hot_chunk, chunk, idx): idx for idx, chunk in enumerate(smiles_chunks)}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing SMILES"):
        idx = futures[future]
        try:
            temp_file, failed_smiles = future.result()
            temp_files.append(temp_file)
            failed_smiles_all.extend(failed_smiles)
        except Exception as e:
            print(f"Error processing batch {idx}: {e}")

# 合并所有临时文件到最终HDF5文件
final_output = 'zinc_grammar_dataset.h5'
with h5py.File(final_output, 'w') as h5f:
    full_dataset = h5f.create_dataset('data', shape=(len(L), MAX_LEN, NCHARS), dtype=np.float32)
    offset = 0
    for temp_file in temp_files:
        with h5py.File(temp_file, 'r') as temp_h5:
            chunk_data = temp_h5['data'][:]
            full_dataset[offset:offset + chunk_data.shape[0], :, :] = chunk_data
            offset += chunk_data.shape[0]
        os.remove(temp_file)  # 删除临时文件

# 保存失败的SMILES
if failed_smiles_all:
    with open('failed_smiles.log', 'w') as f:
        f.write("\n".join(failed_smiles_all))
    print(f"Failed SMILES recorded in 'failed_smiles.log'")
else:
    print("All SMILES processed successfully.")

print(f"Final dataset saved to {final_output}")
