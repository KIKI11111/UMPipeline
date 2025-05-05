import pandas as pd
import numpy as np

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





