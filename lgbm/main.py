import pandas as pd
import numpy as np
from funcs import *

if __name__ == '__main__':


    # 样本数量
    n_samples = 1000
    # 特征数量
    n_features = 20
    # 生成 ID 列
    ids = np.arange(1, n_samples + 1)
    # 生成 20 个特征列
    features = np.random.randn(n_samples, n_features)
    # 生成二元干预列
    treatment = np.random.randint(0, 2, n_samples)
    # 生成二元标签列
    label = np.random.randint(0, 2, n_samples)
    # 创建 DataFrame
    columns = ['id'] + [f'feature_{i + 1}' for i in range(n_features)] + ['treatment', 'label']
    data = np.hstack((ids.reshape(-1, 1), features, treatment.reshape(-1, 1), label.reshape(-1, 1)))
    df = pd.DataFrame(data, columns=columns)
    print(df.head())

    # 训练
    features = [f'feature_{i + 1}' for i in range(n_features)]
    target_col = 'label'
    treatment_col = 'treatment'


    params = {
        'objective': 'binary',  # 二分类问题
        'metric': 'binary_logloss', # 评估指标
        'boosting_type': 'gbdt',  # 提升类型，梯度提升决策树
        'num_leaves': 31,  # 树的最大叶子数
        'learning_rate': 0.05,  # 学习率
        'feature_fraction': 0.9,  # 建树时使用特征的比例
        'bagging_fraction': 0.8,  # 建树时使用样本的比例
        'bagging_freq': 5,  # 每 k 次迭代执行bagging
        'verbose': -1  # 不输出详细信息
    }

    models = dict()
    model = get_model(df, features+[treatment_col], [target_col], params)
    predict = get_predict(model, df)


    # def get_auuc_score(data, treatment, normalize):
    #     d = get_treatment_comp_data(data, treatment)
    #     return auuc_score(d, 'label', 'treatment', normalize=normalize)
    #

