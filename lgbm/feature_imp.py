from causalml.feature_selection import FilterSelect

#method in {'F', 'LR', 'KL', 'ED', 'Chi'}
def filter_select_causalml(data, features, label, method, experiment_group_column, control_group, treatment_group, n_bins):
    """
    causaml库特征重要性排名
    :param data:
    :param features:
    :param label:
    :param method:
    :param experiment_group_column:
    :param control_group:
    :param treatment_group:
    :param n_bins:
    :return:
    """
    filter_method = FilterSelect()
    feature_imp = filter_method.get_importance(data, features, label, method,
                                                experiment_group_column=experiment_group_column,
                                                control_group=control_group,
                                                treatment_group=treatment_group,
                                                n_bins=n_bins)
    return feature_imp


def get_shap_imp(model, data):
    """
    shap特征重要性
    :param model:
    :param data:
    :return:
    """
    pass

def get_gbm_imp(model):
    """
    ligbtgbm特征重要性
    :param model:
    :return:
    """

    pass
