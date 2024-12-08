#code/DLEPS/dleps_predictor.py

from __future__ import print_function, division

import os
import numpy as np
import pandas as pd
import nltk
import h5py
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Lambda
from keras.models import Model
from keras import backend as K
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau

from utils import get_fp, to1hot
import molecule_vae
from molecule_vae import get_zinc_tokenizer
import zinc_grammar
from vectorized_cmap import computecs

def sampling(args):
    z_mean_, z_log_var_ = args
    batch_size = K.shape(z_mean_)[0]
    epsilon = K.random_normal(shape=(batch_size, 56), mean=0., stddev=0.01)
    return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

class DLEPS(object):
    def __init__(self, setmean=False, reverse=True, base=-2, up_name=None, down_name=None, save_exp=None, model_weights_path=None):
        """
        初始化 DLEPS 模型。

        参数:
            setmean (bool): 是否设置基因表达为均值。
            reverse (bool): 是否反转上调/下调基因集的效能评分。
            base (float): 基础效能评分。
            up_name (str): 上调基因集文件路径。
            down_name (str): 下调基因集文件路径。
            save_exp (str): 保存基因表达谱的文件路径。
            model_weights_path (str): 模型权重文件路径。
        """
        self.save_exp = save_exp
        self.reverse = reverse
        self.base = base
        self.setmean = setmean
        self.loaded = False
        self.model_weights_path = model_weights_path

        self.model = []
        self.model.append(self._build_model())
        self.W = self._get_W()
        A3, self.con = self._get_con()
        self.full_con = A3.dot(self.W)

        # 加载基因信息
        self.genes = pd.read_table("/mnt/d/Research/PHD/DLEPS/data/gene_info.txt", header=0)
        self.gene_dict = dict(zip(self.genes["pr_gene_symbol"], self.genes["pr_gene_id"]))

        # 加载上调基因集
        if up_name:
            self.ups_new = self._get_genes(up_name)
        else:
            self.ups_new = None
            self.reverse = False
            print('DLEPS: No input of up files\n')

        # 加载下调基因集
        if down_name:
            self.downs_new = self._get_genes(down_name)
        else:
            self.downs_new = None
            self.reverse = False
            print('DLEPS: No input of down files\n')

        # 加载模型权重
        if self.model_weights_path and os.path.exists(self.model_weights_path):
            self.model[0].load_weights(self.model_weights_path)
            self.loaded = True
            print(f"Loaded model weights from '{self.model_weights_path}'.")
        else:
            print("Model weights not found. Please train the model or provide model weights.")

    def _build_model(self):
        """
        构建 DLEPS 模型。

        返回:
            Keras 模型实例。
        """
        # 变分自编码器权重路径
        grammar_weights = '/mnt/d/Research/PHD/DLEPS/data/zinc_vae_grammar_L56_E100_val.hdf5'
        grammar_model = molecule_vae.ZincGrammarModel(grammar_weights)
        self.grammar_model = grammar_model
        z_mn, z_var = grammar_model.vae.encoderMV.output
        x = Lambda(sampling, output_shape=(56,), name='lambda')([z_mn, z_var])
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(1024, activation='tanh')(x)
        x = Dropout(0.25)(x)
        expression = Dense(978, activation='linear')(x)
        model = Model(inputs=grammar_model.vae.encoderMV.input, outputs=expression)
        return model

    def _get_W(self):
        """
        获取基因映射矩阵 W。

        返回:
            numpy 数组，形状为 (978, 12328)。
        """
        hf = h5py.File('/mnt/d/Research/PHD/DLEPS/data/denseweight.h5', 'r')
        n1 = hf.get('W')
        W = np.array(n1)
        hf.close()
        return W

    def _get_con(self):
        """
        获取 978 个基因的平均表达水平。

        返回:
            A3 (numpy 数组): 形状为 (979,)。
            con (numpy 数组): 形状为 (978,)。
        """
        benchmark = pd.read_csv('/mnt/d/Research/PHD/DLEPS/data/benchmark.csv')
        A3 = np.concatenate((np.array([1]), benchmark['1.0'].values), axis=0)
        con = benchmark['1.0'].values
        return A3, con

    def _get_genes(self, fl_name):
        """
        获取基因签名。

        参数:
            fl_name (str): 基因集文件路径。

        返回:
            list: 筛选后的基因 ID 列表。
        """
        print(fl_name)
        up = pd.read_csv(fl_name, header=None)
        ups = up.values.astype(int)
        print(ups.shape)
        ups = list(np.squeeze(ups))
        ups_new = [i for i in ups if i in list(self.genes["pr_gene_id"])]
        print(ups_new)
        return ups_new

    def train(self, smile_train, rna_train, validation_data, epochs=30000, batch_size=512, shuffle=True):
        """
        训练模型。

        参数:
            smile_train (numpy 数组): 训练集 SMILES 数据。
            rna_train (numpy 数组): 训练集 RNA 表达数据。
            validation_data (tuple): 验证集数据，形式为 (smile_val, rna_val)。
            epochs (int): 训练轮数。
            batch_size (int): 批量大小。
            shuffle (bool): 是否打乱数据。

        返回:
            history: 训练历史。
        """
        assert (not self.loaded), 'Dense Model should not be loaded before training.'

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.2,
                                      patience=10,
                                      min_lr=0.0001)

        # 冻结 VAE 编码器的层
        for layer in self.grammar_model.vae.encoderMV.layers:
            layer.trainable = False

        # 编译模型
        self.model[0].compile(optimizer='adadelta', loss='mean_squared_error')

        # 训练模型
        his = self.model[0].fit(smile_train,
                                rna_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                validation_data=validation_data,
                                callbacks=[reduce_lr])

        # 保存训练后的模型权重
        self.model[0].save_weights('my_trained_model.h5')
        print("Model weights saved to 'my_trained_model.h5'.")

        return his

    def cal_expr(self, smiles, average=True):
        """
        计算基因表达变化，仅在提供了有效的 SMILES 时调用。

        参数:
            smiles (list): SMILES 字符串列表。
            average (bool): 是否计算平均表达。

        返回:
            numpy 数组: 基因表达变化。
        """
        expr = []
        if len(smiles) > 0:
            onehot = to1hot(smiles)
            expr = self.model[0].predict(onehot)
            return expr
        else:
            return []

    def comp_cs(self, expr, save_file):
        """
        计算富集评分。

        参数:
            expr (numpy 数组): 基因表达变化。
            save_file (str): 是否保存文件。

        返回:
            numpy 数组: 富集评分。
        """
        abs_expr = expr + self.con
        A2 = np.hstack([np.ones([expr.shape[0], 1]), abs_expr])
        L12k = A2.dot(self.W)
        if self.setmean:
            L12k_df = pd.DataFrame(L12k, columns=self.genes["pr_gene_id"])
            L12k_df = L12k_df - L12k_df.mean()
        else:
            L12k_delta = L12k - self.full_con
            L12k_df = pd.DataFrame(L12k_delta, columns=self.genes["pr_gene_id"])
        cs = computecs(self.ups_new, self.downs_new, L12k_df.T)
        if save_file:
            L12k_df.T.to_hdf(save_file, key='data')
        return cs[0].values

    def predict(self, smiles, average=True, save_onehot=None, load_onehot=None, save_expr=None):
        """
        预测 SMILES 的效能评分。

        参数:
            smiles (list): SMILES 字符串列表。
            average (bool): 是否计算平均值。
            save_onehot (str): 是否保存 one-hot 编码。
            load_onehot (str): 是否加载预先保存的 one-hot 编码。
            save_expr (str): 是否保存表达谱。

        返回:
            list: 效能评分列表。
        """
        # 检查模型是否已加载
        if not self.loaded:
            raise ValueError("Model weights not loaded. Please train the model or provide model weights.")

        score = [self.base] * len(smiles)
        idx = []
        expr = []

        if load_onehot:
            clean_smiles = smiles
        else:
            fps = get_fp(smiles)
            assert len(smiles) == len(fps)
            clean_smiles = []
            clean_fps = []
            nan_smiles = []
            for i in range(len(fps)):
                if np.isnan(sum(fps[i])):
                    nan_smiles.append(smiles[i])
                else:
                    clean_smiles.append(smiles[i])
                    clean_fps.append(fps[i])
                    idx.append(i)
            clean_fps = np.array(clean_fps)

        if len(clean_smiles) > 0:
            if load_onehot:
                fss = load_onehot.split('.')
                if fss[-1] == 'npz':
                    onehotz = np.load(load_onehot)
                    print(onehotz.files)
                    onehot = onehotz[onehotz.files[0]]
                else:
                    onehot = np.load(load_onehot)
                head = '.'.join(fss[:-1])
                idx = np.load(head + '_idx.npy').astype(int)
            else:
                onehot = to1hot(clean_smiles)
                if save_onehot:
                    np.save(save_onehot, onehot)
                    np.save(save_onehot + '_idx', idx)

            # 使用本地模型进行预测
            expr = self.model[0].predict(onehot)

            print('DLEPS: 978 signatures obtained\n\n')
            step_size = 50000

            # 如果输入的 SMILES 太多，分批处理
            if expr.shape[0] > step_size:
                cs_arr = np.zeros(expr.shape[0])
                num = int(expr.shape[0] / (step_size * 1.0))
                for i in range(num):
                    print(i, num, sep=':')
                    cur_expr = expr[i * step_size:(i + 1) * step_size]
                    if save_expr:
                        cur_cs = self.comp_cs(cur_expr, save_expr + 'SMILES_L12k_' + str(i) + '.h5')
                    else:
                        cur_cs = self.comp_cs(cur_expr, None)
                    cs_arr[i * step_size:(i + 1) * step_size] = cur_cs

                cur_expr = expr[num * step_size:expr.shape[0]]
                if save_expr:
                    cur_cs = self.comp_cs(cur_expr, save_expr + 'SMILES_L12k_' + str(num) + '.h5')
                else:
                    cur_cs = self.comp_cs(cur_expr, None)

                cs_arr[num * step_size:expr.shape[0]] = cur_cs
            else:
                abs_expr = expr + self.con
                A2 = np.hstack([np.ones([expr.shape[0], 1]), abs_expr])
                L12k = A2.dot(self.W)
                if self.setmean:
                    L12k_df = pd.DataFrame(L12k, columns=self.genes["pr_gene_id"])
                    L12k_df = L12k_df - L12k_df.mean()
                else:
                    L12k_delta = L12k - self.full_con
                    L12k_df = pd.DataFrame(L12k_delta, columns=self.genes["pr_gene_id"])
                if self.save_exp:
                    L12k_df.to_csv(self.save_exp)

                print('DLEPS: 12328 gene changes obtained\n\n')
                if save_expr:
                    cs_arr = self.comp_cs(expr, save_expr)
                else:
                    cs = computecs(self.ups_new, self.downs_new, L12k_df.T)
                    cs_arr = cs[0].values
                print('DLEPS: Enrichment scores were calculated\n\n')

            # 反转效能评分（如果需要）
            for i in range(len(idx)):
                if self.reverse:
                    score[idx[i]] = cs_arr[i] * (-1)
                else:
                    score[idx[i]] = cs_arr[i]

        return score
