import numpy as np
import pandas as pd

# 简单逻辑
def func_sum(args):
    def func(x):
        if not (isinstance(x, np.ndarray) or isinstance((x, pd.Series))):
            raise Exception("x should be one of np.ndarray or pandas.Series")
        try:
            return np.sum(x)
        except:
            return None
    return func


def func_mean(args):
    def func(x):
        if not (isinstance(x, np.ndarray) or isinstance((x, pd.Series))):
            raise Exception("x should be one of np.ndarray or pandas.Series")
        try:
            return np.mean(x)
        except:
            return None
    return func



def func_count_distinct(args):
    def func(x):
        if not (isinstance(x, np.ndarray) or isinstance((x, pd.Series))):
            raise Exception("x should be one of np.ndarray or pandas.Series")
        try:
            return len(np.unique(x))
        except:
            return None
    return func


def func_std(args):
    def func(x):
        if not (isinstance(x, np.ndarray) or isinstance((x, pd.Series))):
            raise Exception("x should be one of np.ndarray or pandas.Series")
        try:
            return np.std(x)
        except:
            return None
    return func


def func_max(args):
    def func(x):
        if not (isinstance(x, np.ndarray) or isinstance((x, pd.Series))):
            raise Exception("x should be one of np.ndarray or pandas.Series")
        try:
            return np.max(x)
        except:
            return None
    return func


def func_min(args):
    def func(x):
        if not (isinstance(x, np.ndarray) or isinstance((x, pd.Series))):
            raise Exception("x should be one of np.ndarray or pandas.Series")
        try:
            return np.min(x)
        except:
            return None
    return func


def func_range(args):
    def func(x):
        if not (isinstance(x, np.ndarray) or isinstance((x, pd.Series))):
            raise Exception("x should be one of np.ndarray or pandas.Series")
        try:
            return np.max(x) - np.min(x)
        except:
            return None
    return func

def func_series_arg_sort_idx(args):

    assert "order" in args, "arg of func_series_arg_sort_idx shoule contain 'order'"
    order = args.get("order")
    assert "number" in args, "arg of func_series_arg_sort_idx shoule contain 'number'"
    num = args.get("number")
    assert order in ("asc", "desc"), "order should be one of ('asc', 'desc')"

    def func(x):
        if not (isinstance(x, np.ndarray) or isinstance((x, pd.Series))):
            raise Exception("x should be one of np.ndarray or pandas.Series")
        try:
            sorted_x = x.sort_values(ascending=(order == "asc"))
            return sorted_x.index[num]
        except Exception as e:
            raise Exception(e)
    return func


def func_select_where(args):
    where_col = args.get("where_col")
    where_val = args.get("where_val")
    select_col = args.get("select_col")
    if select_col is not None and len(select_col.split(',')) > 1:
        select_col = select_col.split(',')

    def func(x, input_where_val=None):
        val = where_val if where_val is not None else input_where_val
        try:
            return x[x[where_col] == val][select_col]
        except Exception as e:
            raise Exception(e)
    return func






