import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score


def get_predict(model, data):
    treatment_list = list(range(3, 11))
    treatment_col_list = [f'uplift_{t}' for t in treatment_list]
    treat = data['treatment'].copy()
    data['treatment'] = 0
    data['base_p'] = model.predict(data[model.feature_name_])
    for t in treatment_list:
        data['treatment'] = t
        data[f'uplift_{t}'] = model.predict(data[model.feature_name_]) - data['base_p']
    data['treatment'] = treat
    return data




def get_model(data, feature_cols,  target_col, params):

    train_d, test_d = train_test_split(data, test_size=0.2)
    train_d, val_d = train_test_split(train_d, test_size=0.2)

    train_d_set = lgb.Dataset(data=train_d[feature_cols], label=train_d[[target_col]])
    val_d_set = lgb.Dataset(data=val_d[feature_cols], label=val_d[[target_col]])
    booster = lgb.train(params, train_d_set, valid_sets=[train_d_set, val_d_set])

    # 在测试集上进行预测
    y_pred_proba = booster.predict(test_d[feature_cols])
    y_pred = (y_pred_proba > 0.5).astype(int)

    # 计算 AUC 和准确度
    auc = roc_auc_score(test_d[target_col], y_pred_proba)
    accuracy = accuracy_score(test_d[target_col], y_pred)

    print(f"AUC: {auc}")
    print(f"Accuracy: {accuracy}")

    return booster
