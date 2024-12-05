# DLEPS

A Deep Learning based Efficacy Prediction System for Drug Discovery

# 环境配置

由于grammarVAE项目的分子编码执行在tensorflow1.x的环境中,因此需配置

`pip install tensorflow==1.15.0`
`pip install keras==2.3.0`

# 项目结构及文件解读

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


