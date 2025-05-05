from src.executor import *
from src.operators.basic import *
import argparse

Parser = argparse.ArgumentParser()

Parser.add_argument('--timestamp', type=str, default='', help='时间戳')
Parser.add_argument('--config_file_path', type=str, default='', help='配置变量')
Parser.add_argument('--sql_args', type=str, default='', help='sql条件变量')
Parser.add_argument('--spark_args', type=str, default='', help='spark配置')

Args = Parser.parse_args()
TIME = Args.timestamp
PATH = Args.config_file_path
SQL_ARGS = Args.sql_args
SPARK_ARGS = Args.spark_args

if __name__ == '__main__':
    print('开始执行时间：', TIME)
    executor = Executor(config_file_path=PATH, sql_args=SQL_ARGS, spark_args=SPARK_ARGS)
    executor.handle()
