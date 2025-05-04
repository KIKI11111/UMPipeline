import glob as glob
# from unittest.mock import inplace

import pandas as pd
import os

from sympy import subsets

proj_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/hdfs/"

import numpy as np
from datetime import datetime
from pyarrow import orc as orc
# from conf import *
import lightgbm as lgb
import copy
import pyarrow as pa

def load_data():
    data_names = os.listdir(f"{proj_path}/")
    data_all = {data_name: pd.concat([pd.read_csv(f) for f in glob(f"{proj_path}/{data_name}/*.csv")], axis=0)
                for data_name in data_names}
    return data_all

def del_file(file):
    """删除某个文件"""
    exec_res_mark = 0
    if os.path.exists(file):
        print(f"remove history result: {file}")
        exec_res_mark = os.system(f"rm -f {file}")
    if exec_res_mark != 0:
        raise Exception("remove file failed")
    return exec_res_mark

def mv_file(from_file, to_file):
    """移动文件"""
    if os.path.exists(to_file):
        os.system(f"rm -f {to_file}")
    print(f"mv {from_file} {to_file} ...")
    exec_res_mark = os.system(f"mv {from_file} {to_file}")
    print(f"mv {from_file} {to_file} done")
    if exec_res_mark != 0:
        raise Exception("move file failed")
    return exec_res_mark

def save_result_orc(data, result_file_tmp, result_file, result_name, rename_dict=None):
    """将结果保存为ORC文件，先存入本地临时地址：result_file_tmp， 然后移动到result_file"""
    if rename_dict is not None:
        data = data.rename(columns=rename_dict)
    print(f"saving result {result_name} ...")
    del_file(result_file_tmp)
    del_file(result_file)
    tabel_raw = pa.Table.from_pandas(data, preserve_index=False)
    orc.write_table(tabel_raw, result_file_tmp)
    mv_file(result_file_tmp, result_file)
    print(f"saving result {result_name} done.")

def read_all_csv_file_in_path(folder):
    res = []
    for f in glob(f"{folder}/*.csv"):
        res.append(pd.read_csv(f))
    return pd.concat(res, axis=0)

def clean_tmp_result_path():
    """
    清空临时结果目录
    :return:
    """
    tmp_res_path = "/tf/launcher/tmp_1"
    assert tmp_res_path.startswith("/tf/launcher/tmp_")
    os.system(f"rm -rf {tmp_res_path}")
    os.makedirs(tmp_res_path)


def data_process_base(data, dt_col, drop_col_list):
    """
    对数据进行基本检查，删除冗余列，对日期进行类型转化，标注序号
    :param data:
    :param dt_col:
    :param drop_col_list:
    :return:
    """
    process_result = data.copy()
    process_result.drop(columns=drop_col_list, inplace=True)
    assert process_result[dt_col].nunique() == process_result.shape[0], '数据日期存在重复'
    assert process_result[dt_col].nunique() == len(pd.date_range(process_result[dt_col].min(), process_result[dt_col].max())), '数据存在缺失值'
    process_result[dt_col] = pd.to_datetime(process_result[dt_col])
    process_result.reset_index(drop=True, inplace=True)
    process_result["time_idx"] = process_result.index
    return process_result

def add_label(data, predict_length, target_col, dt_col, label_mark):
    """
    计算标签
    :param data:
    :param predict_length:
    :param target_col:
    :param dt_col:
    :param label_mark:
    :return:
    """
    data = data.copy()
    data = data.sort_values(by=dt_col, ascending=True).reset_index(drop=True)
    for i in range(predict_length):
        data[f'{label_mark}{i+1}_real'] = data[target_col].shift(-(i+1))
    return data

def get_feature_and_label_cols(data, target_col, label_mark, feature_do_not_used_cols):
    """
    获取特征列和目标列
    :param data:
    :param target_col:
    :param label_mark:
    :param feature_do_not_used_cols:
    :return:
    """
    labels = list(i for i in data.columns if i.startswith(label_mark))
    features = list(i for i in data.columns if i != target_col and not i.startswith(label_mark) and i not in feature_do_not_used_cols)
    return features, labels

def train_test_split(data, test_day_cnt, dt_col, predict_length, target_col_list):
    """
    训练集测试集划分
    :param data:
    :param test_day_cnt:
    :param dt_col:
    :param predict_length:
    :param target_col_list:
    :return:
    """
    data = data.copy()
    data = data.sort_values(by=dt_col, ascending=True).reset_index(drop=True)
    data.dropna(subset=target_col_list, how='any', inplace=True)
    test_data = data.iloc[-test_day_cnt:, :].copy()
    train_data = data.iloc[:-(test_day_cnt + predict_length), :].copy()
    return train_data, test_data

def get_model(model_params, x_train, y_train, eval_set, best_iteration=None):
    """
    训练模型
    :param model_params:
    :param x_train:
    :param y_train:
    :param eval_set:
    :param best_iteration:
    :return:
    """
    model_params = copy.deepcopy(model_params)
    if best_iteration:
        model_params['n_estimators'] = best_iteration
        model_params['early_stopping_rounds'] = -1
    if 'eval_metric' in model_params:
        eval_metric = model_params.pop('eval_metric')
    else:
        eval_metric = 'mse'
    gbm = lgb.LGBMRegressor(**model_params)
    gbm.fit(x_train, y_train, eval_set=eval_set, verbose=False, eval_metric=eval_metric)
    return gbm

def train_model_and_predict(d, model_params, validation_length, train_vaild_dt_gap, test_length, predict_length, feature_col_list
                            ,target_col_list, dt_col, retrain_to_test, retrain_to_final_predict, model_ver, businessline, scope_mark)
    # 训练集测试集划分
    train_all, test_all = train_test_split(d, test_length, dt_col=dt_col, predict_length=predict_length, target_col_list=target_col_list)
    # 训练集验证集划分
    data_train, data_vaild = train_test_split(train_all, validation_length, dt_col=dt_col, predict_length=train_vaild_dt_gap, target_col_list=target_col_list)

    x_train = data_train[feature_col_list]
    x_vaild = data_vaild[feature_col_list]
    x_train_all = train_all[feature_col_list]
    x_d_all = d[feature_col_list]
    x_test_all = test_all[feature_col_list]
    x_final_pred = d.iloc[[-1], :].loc[:, feature_col_list].copy()

    pred_res = []
    model_performance = {}
    model_dict = {}
    for i in range(predict_length):
        model_performance[i+1] = {}
        label = f'{label_mark}{i+1}_real'
        y_train = data_train[label]
        y_vaild = data_vaild[label]
        y_test_all = test_all[label]
        y_train_all = train_all[label]
        y_d_all = d[label]
        gbm = get_model(model_params, x_train, y_train, eval_set=[(x_train, y_train), (x_vaild, y_vaild)], best_iteration=None)
        best_iteration = gbm.best_iteration_
        mape_train = round(abs(gbm.predict(x_train, num_iteration=best_iteration) - y_train).sum() / y_train.sum(), 3)
        mape_vaild = round(abs(gbm.predict(x_vaild, num_iteration=best_iteration) - y_vaild).sum() / y_vaild.sum(), 3)
        if retrain_to_test:
            gbm = get_model(model_params, x_train_all, y_train_all, eval_set=None, best_iteration=best_iteration)
        mape_test = round(abs(gbm.predict(x_test_all, num_iteration=best_iteration) - y_test_all).sum() / y_test_all.sum(), 3)

        model_performance[i+1]['train_mape'] = mape_train
        model_performance[i+1]['vaild_mape'] = mape_vaild
        model_performance[i+1]['test_mape'] = mape_test
        model_performance[i+1]['best_iteration'] = int(best_iteration)
        model_performance[i+1]['model_ver'] = model_ver

        if retrain_to_final_predict:
            gbm = get_model(model_params, x_d_all, y_d_all, eval_set=None, best_iteration=best_iteration)
        model_dict[i+1] = gbm
        pred_res.append(gbm.predict(x_final_pred, num_iteration=best_iteration)[0])
        return pred_res, model_performance, model_dict

def str_dt(dt):
    return str(dt)[:10]

class DotDict(dict):
    def __int__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class SplitInfo(object):

    def __int__(self, start_dt, end_dt, predict_next_nd, valid_day_cnt, product=False, gap=False):
        self.data_start_dt = start_dt
        self.data_end_dt = end_dt
        self.predict_next_nd = predict_next_nd
        self.valid_day_cnt = valid_day_cnt
        self.product = product
        self.gap = gap
        self.gap_days = predict_next_nd if gap else 0

        self.date_range = pd.date_range(start_dt, end_dt)
        self.test_dt_range = self.date_range[-predict_next_nd:]
        self.valid_dt_range = self.date_range[(-predict_next_nd-self.gap_days-valid_day_cnt):-predict_next_nd-self.gap_days]
        self.train_dt_range = self.date_range[:(-predict_next_nd-self.gap_days*2-valid_day_cnt)]
        self.predict_dt_range = None

        self.train_length = len(self.train_dt_range)
        self.train_test_gap = (self.valid_dt_range[0] - self.test_dt_range[-1]).days -1
        self.valid_length = len(self.valid_dt_range)
        self.test_valid_gap = (self.test_dt_range[0] - self.valid_dt_range[-1]).days - 1
        self.test_length = len(self.test_dt_range)
        self.predict_length = 0

        self.train_start_dt = str(self.train_dt_range[0])[:10]
        self.train_ent_dt = str(self.train_dt_range[-1])[:10]
        self.valid_start_dt = str(self.valid_dt_range[0])[:10]
        self.valid_end_dt = str(self.valid_dt_range[-1])[:10]
        self.test_start_dt = str(self.test_dt_range[0])[:10]
        self.test_end_dt = str(self.test_dt_range[-1])[:10]

        self.predict_start_dt = None
        self.predict_end_dt = None

        if product:
            self.predict_dt_range = pd.date_range(self.date_range[-1] + pd.Timedelta(days=1), self.date_range[-1]+pd.Timedelta(days=predict_next_nd))
            self.predict_start_dt = str(self.predict_dt_range[0])[:10]
            self.predict_end_dt = str(self.predict_dt_range[-1])[:10]
            self.predict_length = len(self.predict_dt_range)

    def info(self):
        return {
            'data_start_dt':self.data_start_dt,
            'data_end_dt':self.data_end_dt,
            'predict_next_nd':self.predict_next_nd,
            'valid_day_cnt':self.valid_day_cnt,
            'product':self.product,
            'train_length':self.train_length,
            'train_test_gap':self.train_test_gap,
            'valid_length':self.valid_length,
            'test_length':self.test_length,
            'predict_length':self.predict_length
        }
    def get_split(self):
        return {
            'train_start_dt':self.train_start_dt,
            'train_end_dt':self.train_ent_dt,
            'valid_start_dt':self.valid_start_dt,
            'valid_end_dt':self.valid_end_dt,
            'test_start_dt':self.test_start_dt,
            'test_end_dt':self.test_end_dt,
            'predict_start_dt':self.predict_start_dt,
            'predict_end_dt':self.predict_end_dt
        }
























