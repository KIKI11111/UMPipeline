
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr

from tqdm import tqdm
import pickle
from sklearn.model_selection import KFold


def get_user_cv_model(sample_data, marker, target_col, feat_cols, train_params, cv=4):
    """
    :param sample_data:
    :param marker:
    :param target_col:
    :param feat_cols:
    :param train_params:
    :param cv:
    :return:
    """
    model_dict_return = {}
    user_set = np.array(list(set(sample_data.userid.unique())))
    kf = KFold(n_splits=cv, shuffle=True)
    i = 0
    for tr_ind, val_ind in kf.split(user_set):
        i += 1
        tr_user = set(user_set[tr_ind])
        val_user = set(user_set[val_ind])
        print(f'train shape:{len(tr_user)},val shape:{len(val_user)}')
        train_d = sample_data.loc[sample_data.userid.isin(tr_user), [target_col] + feat_cols].copy()
        val_d = sample_data.loc[sample_data.userid.isin(val_user), [target_col] + feat_cols].copy()

        train_d_set = lgb.Dataset(data=train_d[feat_cols], label=train_d[[target_col]])
        val_d_set = lgb.Dataset(data=val_d[feat_cols], label=val_d[[target_col]])
        booster = lgb.train(train_params, train_d_set, valid_sets=[train_d_set, val_d_set])

        model_dict_return[f"{marker}_{i}"] = booster
    return model_dict_return



def fill_mean(df):
    for col in list(df.columns[df.isnull().sum() > 0]):
        mean_v = df[col].mean()
        df[col].fillna(mean_v, inplace=True)
    return df


def build_psm_model(d, feature_ls, model_params, treat_ls, target_col='treatment', cv=3):
    ps_models = dict()
    for treat in tqdm(treat_ls):
        treatment_set = {0, treat}
        d_selected = d.query(f"{target_col}.isin(@treatment_set).values").copy()
        d_selected[target_col] = (d_selected[target_col] > 0) * 1
        ps_models[treat] = get_user_cv_model(d_selected, 'psm', target_col, feature_ls, model_params, cv=cv)
    return ps_models


def get_psm_t_score(ps_models, df_to_predict, treatment):
    features = None
    model_set = ps_models[treatment]
    predict_result = None
    for _, model in model_set.items():
        if features is None:
            features = model.feature_name()
        if predict_result is None:
            predict_result = model.predict(df_to_predict[features])
        else:
            predict_result = predict_result + model.predict(df_to_predict[features])
    return predict_result / len(model_set)


def get_psm_score(df_to_predict, dim_col_list, ps_models, treatment_list):
    data_dim = df_to_predict[dim_col_list].copy()
    for treat in tqdm(treatment_list):
        data_dim[f'ps_score_{treat}'] = get_psm_t_score(ps_models, df_to_predict, treat)
    return data_dim


def get_neighbors(data_dict, base_treat, neighbor_model, buffer_rt=1):
    t_index_dict = dict()
    for treat, d in tqdm(data_dict.items()):
        if treat == base_treat:
            continue
        dist, idx = neighbor_model.kneighbors(d, return_distance=True)
        t_index_dict[treat] = pd.DataFrame(zip(idx.flatten(), dist.flatten()),
                                           columns=['index', 'distance']).sort_values(by='distance',
                                                                                      ascending=True).drop_duplicates(
            subset=['index'], keep='first').iloc[:int(round(d.shape[0] * buffer_rt, 0)), 0].values
        print(d.shape, len(t_index_dict[treat]))
    idx_all = set()
    for _, idx in t_index_dict.items():
        idx_all = idx_all | set(idx)
    return idx_all, t_index_dict





# eg

# treatment_ls = list(range(3, 11))
# score_cols = [f'ps_score_{i}' for i in treatment_ls]
# feature_top_n = 40  # 选取top n 的特征

# treatment_col = 'treatment'
# base_treatment = 0

# # 数据增加标识列，“rct_mark”

# dim_cols = []


# # 处理， 删除缺失率高的特征等
# data_clean = data.copy(deep=True)

# # 计算各个特征与干预特征的相关性，取TOP N个特征进行建模
# correlations = []
# for column in list(data.columns[data.isnull().sum() > 0]):
#     mean_val = data[column].mean()
#     data[column].fillna(mean_val, inplace=True)

# for feature in tqdm(data.columns):
#     if feature == treatment_col:
#         continue
#     correlation_ls = []
#     for i in range(3):
#         d_tmp = data.sample(frac=0.66)
#         correlation, _ = pearsonr(d_tmp[feature], d_tmp[treatment_col])
#         correlation_ls.append(abs(correlation))
#     correlations.append((feature, np.mean(correlation_ls)))

# correlations.sort(key=lambda x: x[1], reverse=True)
# psm_features = [feature[0] for feature in correlations[:feature_top_n]]

# # 观测样本， 随机样本
# train_data_before_psm = data_clean[data_clean['rct_mark'] == 0].copy()
# test_data_before_psm = data_clean[data_clean['rct_mark'] == 1].copy()

# models = build_psm_model(train_data_before_psm, psm_features, params, treatment_ls, target_col=treatment_col, cv=3)
# psm_score = get_psm_score(data_clean, dim_cols, models, treatment_ls)

# # 评估样本， 随机样本中取 干预大于0的 + 同样条数干预=0的样本
# rct_t = psm_score.query(f"rct_mark == 1 and {treatment_col} > 0").copy()
# rct_b = psm_score.query(f"rct_mark == 0 and {treatment_col} == 0").sample(rct_t.shape[0])
# rct_d = pd.concat([rct_b, rct_t], axis=0).sample(frac=1).copy()

# rct_user = set(rct_b.userid) | set(rct_t.userid)

# # 匹配前训练样本， 观测样本 + 随机样本中除去已被选为评估样本的
# obv_t_d  # 匹配前训练样本
# obv_dict = dict()
# for t in tqdm(treatment_ls):
#     obv_dict[t] = obv_t_d.query(f'{treatment_col}=={t}').copy()[score_cols].values
# # 从基准组中筛选相似样本
# obv_dict[base_treatment] = obv_t_d.query(f'{treatment_col}=={base_treatment} and rct_mark == 0'
#                                          ).copy()[score_cols].values
# neigh = NearestNeighbors(n_neighbors=2)
# neigh.fit(obv_dict[base_treatment])
# idx_set, t_idx_dict = get_neighbors(obv_dict, base_treatment, neigh)

# # 匹配后训练样本，观测样本中干预大于0的 + 从随机基准样本中匹配的样本
# obv_t_psm = pd.concat([obv_t_d.query(f'{treatment_col}=={base_treatment}').iloc[list(idx_set), :],
#                        obv_t_d.query(f'{treatment_col}>0')], axis=0).copy()


