# DLEPS

A Deep Learning based Efficacy Prediction System for Drug Discovery

# 环境配置

由于grammarVAE项目的分子编码执行在tensorflow1.x的环境中,因此需配置

```
pip install tensorflow==1.15.0
pip install keras==2.3.0
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
重点关注以下文件

- [GSE92742_Broad_LINCS_sig_info.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fsig%5Finfo.txt.gz)
- [GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5FLevel5%5FCOMPZ.MODZ%5Fn473647x12328.gctx.gz)
- [GSE92742_Broad_LINCS_gene_info.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fgene%5Finfo.txt.gz)

3. **执行[1.BRD ID_TO_SMILES.ipynb](/mnt/d/Research/PHD/DLEPS/data_process/ipynb/1.BRD ID_TO_SMILES.ipynb)文件**:该脚本根据[GSE92742_Broad_LINCS_sig_info.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fsig%5Finfo.txt.gz)中的化合物唯一[BRD标号](https://clue.io/connectopedia/what_is_a_brd_id),从[LINCS_small_molecules.tsv](https://s3.amazonaws.com/lincs-dcic/sigcom-lincs-metadata/LINCS_small_molecules.tsv)中获取所有化合物的SMILES描述符,得到文件[GSE92742_Broad_LINCS_sig_info_final.txt](/mnt/d/Research/PHD/DLEPS/results/GSE92742_Broad_LINCS_sig_info_final.txt).同时处理过程遵循以下原则

   - 只保留筛选包含BRD的化合物(扰动剂)数据

   - 只保留实验记录出现次数大于5次的化合物数据

   - 使用RDKit库验证每个SMILES是否有效
4. **执行[2.转录谱数据分析.ipynb](/mnt/d/Research/PHD/DLEPS/data_process/ipynb/2.转录谱数据分析.ipynb):**该脚本根据[GSE92742_Broad_LINCS_sig_info.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fsig%5Finfo.txt.gz)中的sig_id实验条件信息,以相同SMILES进行分组,从[GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5FLevel5%5FCOMPZ.MODZ%5Fn473647x12328.gctx.gz)检索每一个SMILES分组下所有sig_id实验条件的基因表达数据,对同一个SMILES分组下所有sig_id实验条件的基因表达数据进行加权平均,获取所有化合物的全基因扰动数据([combined_smiles_averages.csv](/mnt/d/Research/PHD/DLEPS/results/combined_smiles_averages.csv)),具体思路:
5. **只保留landmark基因的列数据**:得到L1000_landmark.csv
6. **划分训练集和测试集**:train_SMILES_demo.csv,test_SMILES_demo.csv
7. **执行preprocess.ipynb:**调用GVAE和ONE-HOT算法对train_SMILES_demo.csv以及test_SMILES_demo.csv进行数据编码,保存数据编码文件得到SMILE_train_demo.h5以及SMILE_test_demo.h5,同时此过程中完成L1000_landmark.csv基因表达数据与SMILES的编码对齐
8. **执行Training.ipynb**:进行模型训练,并保存预训练模型参数[my_trained_model.h5](/mnt/d/Research/PHD/DLEPS/my_trained_model.h5)
9. 调用语序连模型进行预测:执行DLEPS_tutorial完成预测
