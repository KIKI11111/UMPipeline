from sales_predict.main import model_ver
from utils.util import *
businessline = ''
model_ver = 'v.0.0.5'
model_ver_mark = model_ver.replace(".", "_")
validation_length = 150
model_performance_str = ''
scope_mark = 'non_wx_app'
data_process_func = data_process_base
feature_func = feature_engineering_base

train_valid_dt_gap = 0
retrain_to_test = False
retrain_to_final_predict = False

model_params = dict(
    objection = 'regression',
    num_leaves = 90,
    learning_rate = 0.05,
    n_estimators = 130,
    max_depth = 10,
    num_threads = 15,
    bagging_fraction = 0.8,
    feature_fraction = 0.8,
    early_stopping_rounds = 20,
    eval_matric = 'mse',
)
result_name = f"predict_result_{model_ver_mark}.pkl"

model_commont = f'基础版模型，验证集{validation_length}天，测试集{test_length}天;' \
                f'训练集与测试集间隔{}天'