import numpy as np
from abc import ABC, abstractmethod
import lightgbm as lgb
from copy import deepcopy

FOCUS_LAST_N = 150

def trans_func(x, r=10):
    return r / (1+np.exp(-2*x)) + (1-r/2)

def l2_w(y_predict, data):
    y_true = data.get_label()
    weight = data.get_weight() if data.get_weight() is not None else np.ones(len(y_true)) * (FOCUS_LAST_N + 1)
    residual = (y_true - y_predict).astype("float")
    w = np.abs(np.sum(residual[weight <= FOCUS_LAST_N])) / max(1, np.sum(np.abs(residual[weight <= FOCUS_LAST_N])))
    w = trans_func(w)
    grad = np.where(weight > FOCUS_LAST_N, -2*residual, -2*residual*w)
    hess = np.where(weight > FOCUS_LAST_N, 2, 2*w)
    return grad, hess

def l2_w_valid(y_predict, data):
    y_true = data.get_label()
    residual = (y_true - y_predict).astype("float")
    weight = data.get_weight() if data.get_weight() is not None else np.ones(len(y_true)) * (FOCUS_LAST_N + 1)
    w = np.abs(np.sum(residual[weight <= FOCUS_LAST_N])) / max(1, np.sum(np.abs(residual[weight <= FOCUS_LAST_N])))
    w = trans_func(w)
    loss = np.where(weight < FOCUS_LAST_N, (residual ** 2), (residual ** 2)*w)
    return 'l2_w_valid', np.mean(loss), False

class TSmodel(ABC):
    def __init__(self, params):
        self.params = params
        self.best_iteration = None

    @abstractmethod
    def fit(self, tr_x, tr_y, val_x=None, val_y=None):
        pass

    @abstractmethod
    def predict(self, X, y=None):
        pass

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X, y)

    @abstractmethod
    def update_params(self, params):
        pass

    @abstractmethod
    def update_best_iteration(self):
        pass

    @abstractmethod
    def init(self):
        pass


class LgbRegModel(TSmodel):
    def __int__(self, params):
        super().__init__(params)
        self.init_params = deepcopy(params)
        self.params = params
        self.model = lgb.LGBMRegressor(**params)

    def fit(self, tr_x, tr_y, val_x=None, val_y=None):
        eval_set = [(tr_x, tr_y), (val_x, val_y)] if val_x is not None and val_y is not None else None
        self.model.fit(tr_x, tr_y, eval_set=eval_set, verbose=False, eval_metric='mape')
        self.update_best_iteration()
        return self

    def predict(self, X, y=None):
        return self.model.predict(X)

    def update_params(self, params):
        self.params.update(params)
        self.model = lgb.LGBMRegressor(**self.params)

    def update_best_iteration(self):
        if self.model.best_iteration_:
            self.best_iteration = self.model.best_iteration_
        else:
            self.best_iteration = self.model.get_params()['n_estimators']

    def init(self):
        self.model = lgb.LGBMRegressor(**self.init_params)


class LgbRegModelCustomLoss(TSmodel):
    def __int__(self, params):
        super().__init__(params)
        self.init_params = deepcopy(params)
        self.params = params
        self.model = None

    def fit(self, tr_x, tr_y, val_x=None, val_y=None):
        tr_d = lgb.Dataset(tr_x, tr_y)
        valid_sets = [tr_d]
        val_d = None
        if val_x is not None and val_y is not None:
            val_d = lgb.Dataset(val_x, val_y)
            valid_sets.append(val_d)
        self.model = lgb.train(self.params, tr_d, valid_sets=valid_sets, feval=l2_w_valid, fobj=l2_w)
        self.update_best_iteration()
        return self

    def predict(self, X, y=None):
        return self.model.predict(X)

    def update_params(self, params):
        self.params.update(params)

    def update_best_iteration(self):
        self.best_iteration = self.model.best_iteration

    def init(self):
        self.params = deepcopy(self.init_params)





