# auuc

import gc
import pandas as pd
from causalml.metrics.visualize import auuc_score
from tqdm import tqdm
from multiprocessing import Pool


def get_balance_sample(d_algo,
                       d_base,
                       random_state=None,
                       concat=True):
    """
    平衡测试组和基准组，采样
    :param d_algo:
    :param d_base:
    :param random_stat:
    :param concat:
    :return:
    """
    d_base = d_base.copy(deep=True)
    d_algo = d_algo.copy(deep=True)
    d_base_cnt = d_base.shape[0]
    d_algo_cnt = d_algo.shape[0]
    if d_base_cnt > d_algo_cnt:
        d_base = d_base.sample(n=d_algo_cnt, random_state=random_state)
    else:
        d_algo = d_algo.sample(n=d_base_cnt, random_state=random_state)
    if concat:
        res = pd.concat([d_algo, d_base], axis=0)
        res = res.sample(n=res.shape[0])
        return res
    return d_algo, d_base


def bool_treatment(d, treatment_col):
    """
    将干预列进行0，1处理
    :param d:
    :param treatment_col:
    :return:
    """
    d[treatment_col] = (d[treatment_col] > 0) * 1
    return d


def get_auuc_detail(sample, treatment_col, label_col):
    """
    获取数据集的auuc结果
    :param sample:
    :param treatment_col:
    :param label_col:
    :return:
    """
    # res = {}
    sample = bool_treatment(sample, treatment_col)
    s = auuc_score(sample, outcome_col=label_col, treatment_col=treatment_col, normalize=True)
    # treatment_effect_col=treatment_effect_col)

    # s_lift = s.values[0]/s.values[1] - 1
    s_lift = s.values[0] - s.values[1]
    # res['auuc_lift'] = s

    #     res['auuc_lift'] = s_lift
    #     res['test_auuc'] = s.values[0]
    #     res['rand_auuc'] = s.values[1]
    #    res['sample_cnt'] = (sample[treatment_col] > 0).sum()
    #    return res
    return s_lift


# 前N个评估样本的累积增益
def get_topN_cumlift(sample, treatment_col, label_col, n):
    sample = bool_treatment(sample, treatment_col)
    s = top_n_cumlift(sample, label_col, treatment_col, n)
    return s


def round_test_auuc(d_algo, d_base, treatment_col, label_col, treatment_effect_col, sample_times, n):
    """
    对某一组进行随机采样，进行多次auuc计算
    :param d_algo:
    :param d_base:
    :param treatment_col:
    :param label_col:
    :param sample_times:
    :return:
    """
    res = []
    for i, t in enumerate(range(sample_times)):
        sample = get_balance_sample(d_algo, d_base, random_state=t, concat=True)
        sample = sample[[treatment_col, label_col, treatment_effect_col]]
        # res.append(pd.DataFrame({i: get_auuc_detail(sample, treatment_col, label_col, treatment_effect_col)}))
        # AUUC
        res.append(get_auuc_detail(sample, treatment_col, label_col))
        # 前N个评估样本的累积增益
        # res.append(get_topN_cumlift(sample, treatment_col, label_col,n = n))
    # res_df = pd.concat(res, axis=0)
    # return res_df
    return res


def calc_all_auuc(d, treatment_col, label_col, treatment_list, treatment_effect_col_list, sample_times=50, n=1000):
    res = {}
    for (treatment, treatment_effect_col) in zip(treatment_list, treatment_effect_col_list):
        d_algo, d_base = get_treatment_data(d,
                                            treatment_col,
                                            label_col,
                                            treatment,
                                            treatment_effect_col=treatment_effect_col,
                                            control_treatment=0,
                                            split=True)

        res[treatment] = round_test_auuc(d_algo, d_base, treatment_col, label_col,
                                         treatment_effect_col=treatment_effect_col, sample_times=sample_times, n=1000)
    return res


def get_treat_sample(df, treatment_col, treatment, balance=False, random_state=None):
    """
    获取某个折扣的样本
    :param df:
    :param treatment_col:
    :param treatment:
    :param balance:
    :param random_state:
    :return:
    """
    cond = (df[treatment_col] == 0) | (df[treatment_col] == treatment)
    ret_df = df.loc[cond, :].copy(deep=True)
    ret_df[treatment_col] = ret_df[treatment_col].map(lambda x: '0' if x == 0 else '1')
    if balance:
        ret_df = get_balance_sample(ret_df.loc[ret_df[treatment_col] == '1', :],
                                    ret_df.loc[ret_df[treatment_col] == '0', :],
                                    random_state=random_state,
                                    concat=True)
    return ret_df


def get_treatment_data(d,
                       treatment_col,
                       label_col,
                       treatment,
                       treatment_effect_col,
                       control_treatment=0,
                       split=True):
    """
    获取某个treatment的预测结果及标签
    :param d:
    :param treatment_col:
    :param label_col:
    :param treatment:
    :param control_treatment:
    :param split:
    :return:
    """
    cond = d[treatment_col].isin({control_treatment, treatment})
    return_cols = [treatment_col, label_col, treatment_effect_col]
    d_result = d.loc[cond, return_cols].copy(deep=True)
    if not split:
        return bool_treatment(d_result, treatment_col)
    d_base = d_result.loc[d_result[treatment_col] == 0, :].copy(deep=True)
    d_algo = d_result.loc[d_result[treatment_col] == treatment, :].copy(deep=True)
    return bool_treatment(d_algo, treatment_col), bool_treatment(d_base, treatment_col)

# eg:
# 根据feature_importance 计算各折扣下的auuc
# N = [5,10,15,20,40,60,80,100,120,140,160,180]
# final_res = {}
# for n in N:
#     features = []
#     predict_sample_result = get_predict(model, sample)
#     # 计算AUUC指标
#     treatment_list = [3,4,5,6,7,8,9,10]
#     treatment_effect_col_list = [f'uplift_{i}' for i in treatment_list]
#     sample_times = 2
#     res = calc_all_auuc(predict_sample_result, treatment_col='treatment', label_col='label',
#                   treatment_list = treatment_list, treatment_effect_col_list = treatment_effect_col_list,
#                   sample_times=sample_times,n = 5000)

#     auuc_res = {}
#     for treatment in treatment_list:
#         auuc_res[treatment] = np.mean(res[treatment])

#     final_res[n] = auuc_res




import numpy as np
import pandas as pd
import seaborn as sns
from causalml.metrics import auuc_score, plot_gain, plot_lift, plot_qini, qini_score, get_qini, get_cumlift
from sklift.viz.base import plot_uplift_by_percentile

from sklift.metrics import qini_auc_score
from sklift.viz import plot_qini_curve
import matplotlib.pyplot as plt
from sklift.metrics.metrics import uplift_by_percentile
from sklift.viz.base import plot_treatment_balance_curve
from sklift.viz.base import plot_uplift_curve
from sklift.metrics.metrics import uplift_auc_score
from sklift.metrics.metrics import uplift_curve


def get_treatment_comp_data(data, treatment):
    test_t_set = {0, treatment}
    sample = data.query('treatment.isin(@test_t_set).values').copy()
    sample['treatment'] = (sample['treatment'] > 0) * 1
    return sample[['treatment', 'label', f'uplift_{treatment}']].copy()


def get_auuc_score(data, treatment, normalize):
    d = get_treatment_comp_data(data, treatment)
    return auuc_score(d, 'label', 'treatment', normalize=normalize)


def get_qini_auc_score(data, treatment, normalize):
    d = get_treatment_comp_data(data, treatment)
    return qini_auc_score(d, 'label', 'treatment', normalize=normalize)


def get_qini_score(data, treatment, normalize):
    d = get_treatment_comp_data(data, treatment)
    return qini_score(d, 'label', 'treatment', normalize=normalize)


def get_cumlift_plot(data, treatment, normalize):
    d = get_treatment_comp_data(data, treatment)
    return get_cumlift(d, 'label', 'treatment')


def plt_gain(data, treatment, normalize):
    d = get_treatment_comp_data(data, treatment)
    return plot_gain(d, 'label', 'treatment', normalize=normalize)


def plt_lift(data, treatment, normalize):
    d = get_treatment_comp_data(data, treatment)
    return plot_lift(d, 'label', 'treatment')


def plt_qini(data, treatment, normalize):
    d = get_treatment_comp_data(data, treatment)
    return plot_qini(d, 'label', 'treatment', normalize=normalize)


def plt_qini_curve(data, treatment_list, uplift_col):
    for treatment in treatment_list:
        d = get_treatment_comp_data(data, treatment)
        plot_qini_curve(d['label'], d[f'uplift_{treatment}'], d['treatment'], perfect=False)


def plot_uplift_percentile(data, treatment_list):
    for treatment in treatment_list:
        d = get_treatment_comp_data(data, treatment)
        plot_uplift_by_percentile(d['label'], d[f'uplift_{treatment}'], d['treatment'], strategy='overall', bins=10,
                                  kind='bar', string_percentiles=True)


def calc_qini(df, treatment, mark, keep_random=False):
    """
    计算模型预测结果的qini提升
    :param df:
    :param treatment:
    :param mark:
    :return:
    """
    res = get_qini(get_treatment_comp_data(df, treatment), 'label', 'treatment')
    if not keep_random:
        res = res.loc[:, [res.columns[0]]]
        res.columns = [mark]
    else:
        res.columns = ['Random', mark]
    return res


def calc_avgite_cate_percentile(data, outcome_col, treatment_col, treatment_list, except_col, bins=10):
    """
    计算每个percentile内样本预估ite均值 and cate(处理组转化率 - 对照组转化率)
    """

    # model_names = [ x for x in df.columns if x not in [outcome_col, treatment_col] + except_col]
    percentiles = [round(p * 100 / bins) for p in range(1, bins + 1)]
    percentiles = [f"0-{percentiles[0]}"] + [f"{percentiles[i]}-{percentiles[i + 1]}" for i in
                                             range(len(percentiles) - 1)]

    result = {}
    for treatment in treatment_list:
        col = f'uplift_{treatment}'
        df = get_treatment_comp_data(data, treatment)
        sorted_df = df.sort_values(col, ascending=False).reset_index(drop=True)
        sorted_df.index = sorted_df.index + 1

        outcome_bin = np.array_split(sorted_df[outcome_col], bins)
        treatment_bin = np.array_split(sorted_df[treatment_col], bins)
        uplift_bin = np.array_split(sorted_df[col], bins)

        sub_result = [[uplift.mean(), y[t == 1].mean() - y[t == 0].mean()] for y, t, uplift in
                      zip(outcome_bin, treatment_bin, uplift_bin)]
        sub_result = pd.DataFrame(sub_result).reset_index(drop=True)
        sub_result.columns = ['avgite', 'cate']
        sub_result.index = percentiles
        result[col] = sub_result

    return result


def plot_avgite_cate_percentile(df, outcome_col, treatment_col, treatment_list, except_col, bins=10):
    """
    绘制每个percentile内样本avgite 和cate，及两者偏差
    df : DataFrame
    outcome_col : 结果列名称
    treatment_col : 干预列名称
    bins： 分组个数，默认10
    """

    result = calc_avgite_cate_percentile(df, outcome_col, treatment_col, treatment_list, except_col, bins=10)
    model_num = len(result)
    fig, axes = plt.subplots(ncols=2, nrows=model_num, figsize=(40, 30), sharex=True, sharey=True)
    i = 0
    index = list(result.values())[0].index.values
    index_position = np.arange(len(index))
    bar_width = 0.35

    for key, res in result.items():
        avgite = res['avgite'].values
        cate = res['cate'].values
        diff = res['cate'].values - res['avgite'].values

        axes[i][0].plot(index_position, avgite, label='avgite', color='forestgreen')
        axes[i][0].plot(index_position, cate, label='cate', color='orange')
        axes[i][1].plot(index_position, diff, label='diff of cate-avgite', color='red')

        # axes[i][0].ylim(-0.1, 0.1)
        axes[i][0].set_title(f'{key}: avgITE and cate by percentile')
        axes[i][1].set_title(f'{key}: diff of avgITE and cate by percentile')
        axes[i][0].set_xticks(index_position, index, rotation=45)
        axes[i][1].set_xticks(index_position, index, rotation=45)
        axes[i][0].axhline(y=0, color='black', linewidth=1)
        axes[i][1].axhline(y=0, color='black', linewidth=1)
        axes[i][0].legend()
        axes[i][1].legend()
        i += 1
    return axes


