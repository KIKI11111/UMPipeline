import numpy as np
import pandas as pd

def func_group_sum(args):

    assert "by" in args, "arg of func_group_sum should contain 'by'"
    by = args.get("by")
    assert "val" in args, "arg of func_group_sum should contain 'val'"
    val = args.get("val")

    def func(x):
        if not isinstance(x, pd.DataFrame):
            raise Exception("input of func_group_sum should be DataFrame")
        try:
            return x.groupby(by)[val].sum()
        except Exception as e:
            return None
    return func


def func_group_mean(args):
    assert "by" in args, "arg of func_group_sum should contain 'by'"
    by = args.get("by")
    assert "val" in args, "arg of func_group_sum should contain 'val'"
    val = args.get("val")

    def func(x):
        if not isinstance(x, pd.DataFrame):
            raise Exception("input of func_group_sum should be DataFrame")
        try:
            return x.groupby(by)[val].mean()
        except Exception as e:
            return None

    return func


def func_group_count_distinct(args):
    assert "by" in args, "arg of func_group_sum should contain 'by'"
    by = args.get("by")
    assert "val" in args, "arg of func_group_sum should contain 'val'"
    val = args.get("val")

    def func(x):
        if not isinstance(x, pd.DataFrame):
            raise Exception("input of func_group_sum should be DataFrame")
        try:
            return x.groupby(by)[val].nunique()
        except Exception as e:
            return None

    return func


def func_group_count(args):
    assert "by" in args, "arg of func_group_sum should contain 'by'"
    by = args.get("by")
    assert "val" in args, "arg of func_group_sum should contain 'val'"
    val = args.get("val")

    def func(x):
        if not isinstance(x, pd.DataFrame):
            raise Exception("input of func_group_sum should be DataFrame")
        try:
            return x.groupby(by)[val].count()
        except Exception as e:
            return None

    return func