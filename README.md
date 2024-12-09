# DLEPS

A Deep Learning based Efficacy Prediction System for Drug Discovery

# 环境配置(复现之前务必阅读)

**请注意**:
**1. DLEPS项目的分子编码部分基于[grammarVAE](https://github.com/mkusner/grammarVAE)项目.**

**2. 整个想最核心的文件是dleps_predictor.py,本部分代码包含了整个模型的训练,预测的核心框架**

**3. [grammarVAE](https://github.com/mkusner/grammarVAE)项目的深度学习环境非常老,基于python2.7和tensorflow0.12开发,非常不建议使用该环境**

**4. DLEPS项目的原始开发环境为python3.7,tensorflow1.15,不支持GPU加速**

**5. 原项目的keras版本为2.X,截至本项目发布之前,keras版本已更新至3.X,两代版本在API层存在较大差异,请务必注意**

**6. 建议按照下面步骤创建虚拟环境,并配置特定版本下的python和tensorflow版本,否则不保证能复现该项目**

   ```bash
   conda create -n DLEPS python==3.7.12
   conda activate DLEPS
   conda install numpy pandas nltk h5py requests  rdkit matplotlib
   pip install tensorflow==1.15
   pip install keras==2.3.0
   pip install h5py==2.10
   ```

# 项目结构及核心文件解读

```
DLEPS
├─ README.md
├─ analysis_plot
│  ├─ Browning_ScoreMap.ipynb
│  ├─ FP_tSNE_TrainingTestFdaNp.ipynb
│  └─ NASH_ScoreMap.ipynb
├─ code
│  ├─ DLEPS
│  │  ├─ DLEPS_tutorial.ipynb
│  │  ├─ Training.ipynb
│  │  ├─ dleps_predictor.py
│  │  ├─ driv_DLEPS.py
│  │  ├─ models
│  │  │  ├─ __init__.py
│  │  │  ├─ model_zinc.py
│  │  │  └─ utils.py
│  │  ├─ molecule_vae.py
│  │  ├─ utils.py
│  │  ├─ vectorized_cmap.py
│  │  └─ zinc_grammar.py
│  └─ Preprocess
│     └─ preprocess.ipynb
└─ my_trained_model.h5
```

- preprocess.ipynb: 分子编码数据预处理
- DLEPS_tutorial.ipynb: DLEPS项目使用教程
- Training.ipynb: DLEPS项目训练
- dleps_predictor.py: DLEPS算法框架

# 项目复现指南

1. 分子编码部分来自[grammarVAE](https://github.com/nanyeglm/grammarVAE)项目,请先根据该项目的指南,进行分子训练和优化,获得基于语法的VAE模型预训练权重文件,grammarVAE项目提供了一个预训练文件:[zinc_vae_grammar_L56_E100_val.hdf5](https://github.com/nanyeglm/grammarVAE/blob/master/pretrained/zinc_vae_grammar_L56_E100_val.hdf5),可直接用于编码和解码SMILES字符串，节省训练时间

2. 从[GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)中下载第一阶段的L1000原始数据

 | 文件名                                                       | 大小     | 描述                                                         |
   | ------------------------------------------------------------ | -------- | ------------------------------------------------------------ |
   | [GSE92742_Broad_LINCS_Level1_LXB_n1403502.tar.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level1_LXB_n1403502.tar.gz) | 1.2 TB   | Level 1 数据：原始流式细胞术数据，每个384孔板孔生成一个LXB文件，记录每个分析物的荧光强度值。 |
   | [GSE92742_Broad_LINCS_Level2_GEX_delta_n49216x978.gctx.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level2_GEX_delta_n49216x978.gctx.gz) | 98.9 MB  | Level 2 数据：从Luminex珠子解卷积后的基因表达值，包含49,216个样本和978个基因。 |
   | [GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx.gz) | 2.3 GB   | Level 2 数据：从Luminex珠子解卷积后的基因表达值，包含1,269,922个样本和978个基因。 |
   | [GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz) | 48.8 GB  | Level 3 数据：包含直接测量的标志性转录本和推断基因的表达谱，经过不变集缩放和分位数归一化处理，包含1,319,138个样本和12,328个基因。 |
   | [GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx.gz) | 49.6 GB  | Level 4 数据：包含相对于对照的差异表达基因的稳健 z 分数。    |
   | [GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz) | 19.9 GB  | Level 5 数据：每种处理的重复样本的加权平均差异表达向量，包含473,647个签名和12,328个基因。 |
   | [GSE92742_Broad_LINCS_README.pdf](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_README.pdf) | 25.8 KB  | 数据集的概述和说明文档。                                     |
   | [GSE92742_Broad_LINCS_auxiliary_datasets.tar.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_auxiliary_datasets.tar.gz) | 1.6 GB   | 辅助数据集，可能包括额外的元数据或支持文件。                 |
   | [GSE92742_Broad_LINCS_cell_info.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_cell_info.txt.gz) | 2.5 KB   | 提供细胞系的元数据信息，如细胞系名称、特性等。               |
   | [GSE92742_Broad_LINCS_gene_info.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_gene_info.txt.gz) | 211.6 KB | 提供基因的注释信息，包括基因ID、基因符号和基因描述。         |
   | [GSE92742_Broad_LINCS_gene_info_delta_landmark.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_gene_info_delta_landmark.txt.gz) | 18.3 KB  | 提供与地标基因相关的注释信息。                               |
   | [GSE92742_Broad_LINCS_inst_info.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_inst_info.txt.gz) | 11.5 MB  | 包含实验详细信息，如处理条件、时间点和剂量。                 |
   | [GSE92742_Broad_LINCS_pert_info.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_pert_info.txt.gz) | 1.1 MB   | 提供关于扰动剂（如药物或基因敲除）的信息。                   |
   | [GSE92742_Broad_LINCS_pert_metrics.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_pert_metrics.txt.gz) | 908.9 KB | 提供扰动剂相关的性能指标信息。                               |
   | [GSE92742_Broad_LINCS_sig_info.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_info.txt.gz) | 10.6 MB  | 提供基因表达签名（signatures）的元数据信息，包括签名ID、处理条件等。 |
   | [GSE92742_Broad_LINCS_sig_metrics.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_metrics.txt.gz) | 11.9 MB  | 提供基因表达签名相关的性能指标信息。                         |
   | [GSE92742_SHA512SUMS.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_SHA512SUMS.txt.gz) | 1.4 KB   | 文件的 SHA512 校验和，用于验证文件完整性。                   |

重点关注以下文件

- [GSE92742_Broad_LINCS_sig_info.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fsig%5Finfo.txt.gz)
- [GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5FLevel5%5FCOMPZ.MODZ%5Fn473647x12328.gctx.gz)
- [GSE92742_Broad_LINCS_gene_info.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fgene%5Finfo.txt.gz)

3. **执行BRD ID_TO_SMILES.ipynb文件**:该脚本根据[GSE92742_Broad_LINCS_sig_info.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fsig%5Finfo.txt.gz)中的化合物唯一[BRD标号](https://clue.io/connectopedia/what_is_a_brd_id),从[LINCS_small_molecules.tsv](https://s3.amazonaws.com/lincs-dcic/sigcom-lincs-metadata/LINCS_small_molecules.tsv)中获取所有化合物的SMILES描述符,得到文件[GSE92742_Broad_LINCS_sig_info_final.txt](/mnt/d/Research/PHD/DLEPS/results/GSE92742_Broad_LINCS_sig_info_final.txt).同时处理过程遵循以下原则

   - 只保留筛选包含BRD的化合物(扰动剂)数据

   - 只保留实验记录出现次数大于5次的化合物数据

   - 使用RDKit库验证每个SMILES是否有效
4. **执行[2.转录谱数据分析.ipynb](/mnt/d/Research/PHD/DLEPS/data_process/ipynb/2.转录谱数据分析.ipynb)**:该脚本根据[GSE92742_Broad_LINCS_sig_info.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fsig%5Finfo.txt.gz)中的sig_id实验条件信息,以相同SMILES进行分组,从[GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5FLevel5%5FCOMPZ.MODZ%5Fn473647x12328.gctx.gz)检索每一个SMILES分组下所有sig_id实验条件的基因表达数据,对同一个SMILES分组下所有sig_id实验条件的基因表达数据进行加权平均,获取所有化合物的全基因扰动数据([combined_smiles_averages.csv](/mnt/d/Research/PHD/DLEPS/results/combined_smiles_averages.csv)),具体思路:
5. **只保留landmark基因的列数据**:得到L1000_landmark.csv
6. **划分训练集和测试集**:train_SMILES_demo.csv,test_SMILES_demo.csv
7. **执行preprocess.ipynb**:调用GVAE和ONE-HOT算法对train_SMILES_demo.csv以及test_SMILES_demo.csv进行数据编码,保存数据编码文件得到SMILE_train_demo.h5以及SMILE_test_demo.h5,同时此过程中完成L1000_landmark.csv基因表达数据与SMILES的编码对齐
8. **执行Training.ipynb**:进行模型训练,并保存预训练模型参数[my_trained_model.h5](/mnt/d/Research/PHD/DLEPS/my_trained_model.h5)
9. 调用语序连模型进行预测:执行DLEPS_tutorial完成预测
