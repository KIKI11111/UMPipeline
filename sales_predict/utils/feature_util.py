from datetime import datetime
import numpy as np
import pandas as pd
from utils.util import str_dt
import re

spring_year_day_dict = {
    2024: '2024-02-10',
    2025: '2025-01-29'
}

spring_day_list = [v for k, v in spring_year_day_dict.items()]
cat_columns = ['year_int', 'month_int', 'day_int', 'quarter_int', 'weekofyear_int'
               ,'day_of_year_int', 'weekday_int', 'is_year_start_bool', 'is_quarter_start_bool'
               ,'is_month_start_bool', 'is_month_end_bool', 'is_weekend_bool']

def get_last_same_day(ymd, last_n_year, align_days=21):
    ymd_t_raw = pd.to_datetime(ymd)
    dt_range_raw = {str_dt[dt] for dt in pd.date_range(ymd_t_raw - pd.Timedelta(days=align_days),
                                                       ymd_t_raw + pd.Timedelta(days=align_days))}
    spring_day_ls_base = list(dt_range_raw & set(spring_day_list))
    if spring_day_ls_base:
        spring_day = spring_day_ls_base[0]
        diff_days = (pd.to_datetime(spring_day) - ymd_t_raw).days
        last_n_spring_ind = spring_day_list.index(spring_day) - last_n_year
        last_n_y_d = pd.to_datetime(spring_day_list[last_n_spring_ind]) - pd.Timedelta(days=diff_days)
        return str_dt(last_n_y_d)
    y = int(str_dt(ymd).split("-")[0])
    md = str(ymd)[5:10]
    y = y - last_n_year
    ymd = f"{y}-{md}"
    return ymd

def calc_last_nd_mean(dt_label_dict, dt, last_n_d, predict_next_n_day):
    dt = pd.to_datetime(dt)
    edt = dt - pd.Timedelta(days=predict_next_n_day)
    sdt = edt - pd.Timedelta(days=last_n_d - 1 + predict_next_n_day)
    res = 0
    for dt in pd.date_range(sdt, edt):
        res += dt_label_dict.get(dt, 0)
    return res

def feature_v3(data_dict, predict_next_n_day):
    target = data_dict['gmv_real_data'].rename(columns={'ancestor_order_create_dt':'dt', 'pay_gmv_d1':'label'}).groupby('dt')[['label']].sum().reset_index()
    target['dt'] = pd.to_datetime(target['dt'])
    dt_max = target['dt'].max()
    target_add = pd.Series(np.nan, index=pd.date_range(dt_max+pd.Timedelta(days=1), dt_max+pd.Timedelta(days=predict_next_n_day))).to_frame("label")
    target_add.index.name = 'dt'
    target_add = target_add.reset_index()
    target = pd.concat([target, target_add], axis=0).reset_index(drop=True)
    feature_dt = target.copy()
    feature_dt['weekday'] = feature_dt['dt'].dt.weekday
    # 平滑极端值
    max_r = 1.9
    min_r = 0.7
    rt = (feature_dt.set_index('dt')['label']*2 / (feature_dt.set_index('dt')['label'].shift(7) + feature_dt.set_index('dt')['label'].shift(-7))/2 ).dropna().to_dict()
    fix_dict = ( (feature_dt.set_index('dt')['label'].shift(7) + feature_dt.set_index('dt')['label'].shift(-7)) / 2 ).dropna().to_dict()
    dt_label_dict = feature_dt.set_index('dt')['label'].to_dict()
    rt = rt.to_frame('rt').reset_index()
    rt['label_fix'] = rt.apply(
        lambda x: fix_dict[x['dt']] if x['rt'] >= max_r or x['rt'] <= min_r else dt_label_dict[x['dt']], axis=1
    )
    rt_dict = rt.set_index("dt")["label_fix"].to_dict()
    target_dict = feature_dt.set_index(['dt'])['label'].to_dict()
    feature_dt['last_1_year_label'] = feature_dt.dt.map(
        lambda x: target_dict.get(pd.to_datetime(get_last_same_day(x, 2)), np.nan)
    )
    feature_dt['last_2_year_label'] = feature_dt.dt.map(
        lambda x: target_dict.get(pd.to_datetime(get_last_same_day(x, 2)), np.nan)
    )
    feature_dt['last_2_year_mean_label'] = feature_dt['last_2_year_label'] + feature_dt['last_1_year_label']
    feature_dt['last_1_year_label_7d_mean'] = feature_dt.dt.map(
        lambda x: calc_last_nd_mean(target_dict, get_last_same_day((x, 1), 7, predict_next_n_day))
    )
    feature_dt['label_7d_mean'] = feature_dt.dt.map(
        lambda x: calc_last_nd_mean((target_dict, x, 7, predict_next_n_day))
    )
    feature_dt['label_7d_mean_uplift'] = feature_dt['label_7d_mean'] / feature_dt['last_1_year_label_7d_mean']
    feature_dt['last_1_year_label_fix'] = feature_dt['last_1_year_label'] * feature_dt['label_7d_mean_uplift']
    feature_dt['label'] = feature_dt['dt'].map(rt_dict)

    feature_dt['quarter_int'] = feature_dt['dt'].dt.quarter
    feature_dt['is_weekend_bool'] = np.where(feature_dt['dt'].isin([5, 6]), 1, 0)

    feature_dt['day_to_spring_int'] = feature_dt['dt'].apply(
        lambda x: (datetime.strptime(f'{spring_year_day_dict[x.year]} 00:00:00', "%Y-%m-%d %H:%M:%S") - x).days
    )
    feature_dt['day_to_0501_int'] = feature_dt['dt'].apply(
        lambda x: (datetime.strptime(f'{x.year}-05-01 00:00:00', "%Y-%m-%d %H:%M:%S") - x).days
    )
    feature_dt['day_to_0618_int'] = feature_dt['dt'].apply(
        lambda x: (datetime.strptime(f'{x.year}-06-18 00:00:00', "%Y-%m-%d %H:%M:%S") - x).days
    )
    feature_dt['day_to_1001_int'] = feature_dt['dt'].apply(
        lambda x: (datetime.strptime(f'{x.year}-10-01 00:00:00', "%Y-%m-%d %H:%M:%S") - x).days
    )
    feature_dt['day_to_1111_int'] = feature_dt['dt'].apply(
        lambda x: (datetime.strptime(f'{x.year}-11-11 00:00:00', "%Y-%m-%d %H:%M:%S") - x).days
    )
    feature_dt['month_int'] = feature_dt['dt'].dt.month
    feature_dt['label'] = feature_dt['dt'].map(dt_label_dict)

    feature_dt['stat_min'] = feature_dt['label'].rolling(window=7, min_periods=1).min().shift(predict_next_n_day)
    feature_dt['stat_max'] = feature_dt['label'].rolling(window=7, min_periods=1).max().shift(predict_next_n_day)
    feature_dt['stat_mean'] = feature_dt['label'].rolling(window=7, min_periods=1).mean().shift(predict_next_n_day)
    feature_dt['stat_std'] = feature_dt['label'].rolling(window=7, min_periods=1).std().shift(predict_next_n_day)
    feature_dt['stat_median'] = feature_dt['label'].rolling(window=7, min_periods=1).median().shift(predict_next_n_day)

    # 滞后特征
    for i in predict_next_n_day + np.array([0, 1, 2, 3, 6]):
        feature_dt['sale_lag_{}'.format(i)] = feature_dt['label'].shift(i)
    # 差分特征
    for i in range(predict_next_n_day, predict_next_n_day + 3):
        feature_dt['sale_fod_{}'.format(i)] = feature_dt['sale_lag_{}'.format(i+1)] - feature_dt['sale_lag_{}'.format(i)]
    # 环比特征
    for i in range(predict_next_n_day, predict_next_n_day + 3):
        feature_dt['sale_huanbi_{}'.format(i)] = feature_dt['sale_lag_{}'.format(i+1)] / feature_dt['sale_lag_{}'.format(i)]

    for cat in cat_columns:
        if cat not in feature_dt.columns:
            continue
        feature_dt[cat].astype('category')

    return feature_dt

def _feature_v8(data_dict, predict_next_n_day):
    pl = data_dict['price_level']
    pl_feature = pd.concat([pl.groupby(['plan_date', 'price_level'])['pid'].count().unstack(level=1),
                            pl.groupby(['plan_date', 'dim_prd_brand', 'price_level'])['pid'].count().unstack(lavel=[1,2])], axis=1)
    pl_feature.columns = [str(x) for x in pl_feature.columns]
    pl_feature.index.name = 'dt'
    pl_feature = pl_feature.reset_index()
    pl_feature['dt'] = pd.to_datetime((pl_feature['dt']))
    return pl_feature

def feature_v8(data_dict, predict_next_n_day):
    feature_dt = feature_v3(data_dict, predict_next_n_day)
    pl_feature = _feature_v8(data_dict, predict_next_n_day)
    feature_dt = pd.merge(feature_dt, pl_feature, on=['dt'], how='left').copy()
    feature_dt.columns = [str(x) for x in feature_dt.columns]
    feature_dt.columns = [re.sub('''[\(\)（）,，/'"\s]''', '_', x) for x in feature_dt.columns]
    return feature_dt














