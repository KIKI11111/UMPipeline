from pyexpat import features

import pandas as pd

from utils.util import *
# from utils.pipeline_util import *
from utils.feature_util import *
from utils.model_util import *
# from pipelines.conf import *
from itertools import product
from copy import deepcopy
# from pipelines.run_pipeline import *

import os
import pyarrow as pa
from pyarrow import orc as orc
import gc


def select_data(data_dict, dt_cols_dict, end_dt):
    return {
        'baseline':data_dict['baseline'].copy(),
        'coupon_info':data_dict['coupon_info'].copy(),
        'gmv_real_data':data_dict['gmv_real_data'].query(f"{dt_cols_dict['gmv_real_data']} <= '{end_dt}'").copy(),
        'price_level':data_dict['price_level'].copy()
    }

def split_data(data, si:SplitInfo,feature_cols,label_col,dt_col):
    s = si.get_split()
    return {
        'tr_x': data.query(f"{dt_col} >= '{s['train_start_dt']}' and {dt_col} <= '{s['train_end_dt']}'").loc[:, feature_cols].copy(),
        'tr_y': data.query(f"{dt_col} >= '{s['train_start_dt']}' and {dt_col} <= '{s['train_end_dt']}'").loc[:,
                [label_col]].copy(),
        'valid_x': data.query(f"{dt_col} >= '{s['valid_start_dt']}' and {dt_col} <= '{s['valid_end_dt']}'").loc[:,
                feature_cols].copy(),
        'valid_y': data.query(f"{dt_col} >= '{s['valid_start_dt']}' and {dt_col} <= '{s['valid_end_dt']}'").loc[:,
                   [label_col]].copy(),
        'test_x': data.query(f"{dt_col} >= '{s['test_start_dt']}' and {dt_col} <= '{s['test_end_dt']}'").loc[:,
                   feature_cols].copy(),
        'test_y': data.query(f"{dt_col} >= '{s['test_start_dt']}' and {dt_col} <= '{s['test_end_dt']}'").loc[:,
                   [label_col]].copy(),
        'predict_x': data.query(f"{dt_col} >= '{s['predict_start_dt']}' and {dt_col} <= '{s['predict_end_dt']}'").loc[:,
                  feature_cols].copy(),
        'predict_y': data.query(f"{dt_col} >= '{s['predict_start_dt']}' and {dt_col} <= '{s['predict_end_dt']}'").loc[:,
                  [label_col]].copy(),
        'si':s
    }

def fit_model(model, d):
    model.init()
    model.fit(d['tr_x'], d['tr_y'], d['valid_x'], d['valid_y'])
    model.update_best_iteration()
    model.update_params({'n_estimators': model.best_iteration, 'early_stopping_rounds': -1})
    model.fit(np.vstack([d['tr_x'], d['tr_y']]), np.vstack([d['valid_x'], d['valid_y']]))
    test_predict_res = {
        'y_truth': d['test_y'].values,
        'y_predict': model.predict(d['test_x']),
        'test_start_dt': d['si']['test_start_dt'],
        'test_end_dt': d['si']['test_end_dt']
    }
    return model, test_predict_res

def get_predict(model_fitted, d):
    predres = model_fitted.predict(d['predict_x'])
    return pd.DataFrame(
        predres.reshape(-1, 1),
        index = pd.date_range((d['si']['predict_start_dt'], d['si']['predict_end_dt'])),
        columns = ['y_pred']
    )

def strdt(dt):
    return str(dt)[:10]

def stat_res(prot):
    y_truth = prot['y_truth'].flatten()
    stat_res = pd.Series(
        (prot['y_predict'] - y_truth)/y_truth,
        index = pd.date_range((prot['test_start_dt'], prot['test_end_dt']))
    )
    stat_res = stat_res.to_frame('pe')
    stat_res['nex_n_day'] = np.arange(stat_res.shape[0]) + 1
    return stat_res

def stat_pred(pred_ls):
    pred = pd.concat(pred_ls, axis=0)
    pred.index.name = 'dt'
    pred = pred.reset_index()
    pred = pd.merge(pred, feature[['dt', 'label']], on=['dt'], how='inner').dropna(how='any')
    pred['pe'] = (pred['y_pred'] - pred['label'])/pred['label']
    return {
        'pe_mean': pred['pe'].mean(),
        'ape_mean': pred['pe'].map(abs).mean()
    }

def stat_test(res_ls):
    return {
        'pe_mean': pd.concat(res_ls, axis=0)['pe'].mean(),
        'ape_mean': pd.concat(res_ls, axis=0)['pe'].map(abs).mean()
    }

MODEL_NAME = 'v2'
MODEL_DESC = 'product_model_v2'
label_col = 'label'
dt_col = 'dt'
valid_length = 300
dt_cols = {
    'baseline': 'dt',
    'coupon_info': dt_col,
    'gmv_real_data': 'ancestor_order_create_dt',
    'price_level': 'plan_date'
}

businessline = '轮胎'
model_ver = 'v.5'

feature_method = feature_v_8
model = LgbRegModel(model_params)

data = load_data()
data_dt_max = data['gmv_real_data']['ancestor_order_create_dt'].max()
print("==="*30)
print('dtmax:', data_dt_max)

fres = []
for predict_next_n_day in range(1, 15):
    feature = feature_method(data, predict_next_n_day)
    feature_cols = feature.columns[2:]
    assert dt_col in feature.columns[:2]
    assert label_col in feature.columns[:2]
    data_start_dt = strdt(feature.dt.min())
    data_end_dt = strdt(feature.dropna(subset = [label_col])[dt_col].max())
    si = SplitInfo(data_start_dt, data_end_dt, predict_next_n_day, valid_length, True)
    dtmp = split_data(feature, si, feature_cols, label_col, dt_col)

    model_fitted, predict_res_on_test = fit_model(model, dtmp)
    predres = get_predict(model_fitted, dtmp).iloc[[-1], :]
    fres.append(predres)


predres = pd.concat(fres, axis=0)
predres.to_csv('/tf/launcher/v05.csv')

# save_result_orc(predres, '/tf/launcher/tmp.orc', '/root/hdfs/write/result_hive/res.orc', model_ver)


# d = data_process_base(d, dt_col=dt_col, drop_col_list=drop_col_list)
# d = feature_func(d, dt_col=dt_col, spring=spring, cat_columns=cat_columns, target_col=target_col
# d = add_label(d, predict_length=predict_length, target_col=target_col, dt_col=dt_col, label_mark=label_mark)
#
# feature_col_list, target_col_list = get_feature_and_label_cols(d, target_col=target_col, label_mark=label_mark, feature_do_not_used_cols=feature_do_not_used_cols)
# start_dt = datetime.strptime(modelling_data_latest_dt, '%Y-%m-%d')
# pred_dt_list = [str(dt)[:10] for dt in pd.date_range(start_dt, start_dt+pd.Timedelta(days=predict_length))[1:]]
# pred_res, model_preformance, model_dict = train_model_and_predict(d, model_params, validation_length, train_vaild_dt_gap, test_length, predict_length, feature_col_list, target_col_list, dt_col, retrain_to_test, retrain_to_final_predict, model_ver, businessline, scope_mark)
# pred_res_df = pd.DataFrame(zip(pred_dt_list, pred_res), columns=['effective_dt', 'gmv'])
#
# pred_res_df.to_pickle(f"{tmp_res_path}/{result_name}")


#
# dynamic_template_file = '/tf/launcher/dynamic_template.py'
# template_file = '/tf/launcher/template.py'
# clean_tmp_result_path()
#
# code = open(template_file).read()
# for f in glob("/tf/launcher/dynamic_conf_*.py"):
#     with open(f) as fh:
#         dynamic_conf = fh.read()
#         conf_mark = f.split('/')[-1].split('.')[0]
#         dynamic_conf = dynamic_conf.replace('predict_result_{model_ver_mark}.pkl', conf_mark+f'_predict_result_{model_ver_mark}')
#     dynamic_code = dynamic_conf+'\n'*3+code
#     with open(dynamic_template_file, 'w') as fh2:
#         fh2.write(dynamic_code)
#     os.system(f"python {dynamic_template_file}")
# all_result = pd.concat([pd.read_pickle(f) for f in glob('/tf/launcher/tmp_result/*.pkl')], axis=0)










