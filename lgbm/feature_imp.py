from causalml.feature_selection import FilterSelect

#method in {'F', 'LR', 'KL', 'ED', 'Chi'}
def Filter_select_causalml(data, features, label, method, experiment_group_column, control_group, treatment_group, n_bins):
    filter_method = FilterSelect()
    feature_imp = filter_method.get_importance(data, features, label, method,
                                                experiment_group_column=experiment_group_column,
                                                control_group=control_group,
                                                treatment_group=treatment_group,
                                                n_bins=n_bins)
    return feature_imp

