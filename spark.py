
from datetime import datetime, timedelta

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import window as Window
from pyspark.sql.functions import col, when, collect_set, sort_array, sum, lit, coalesce
import argparse
import requests
import logging

Parser = argparse.ArgumentParser()
Parser.add_argument('--timestamp', type=str, default='', help='时间戳')
Args = Parser.parse_args()
TIME = Args.timestamp

target_table = ''

if __name__ == '__main__':

    today_date = datetime.now().date()
    yesterday_date = (today_date - timedelta(days=1)).strftime("%Y-%m-%d")
    today_date = today_date.strftime("%Y-%m-%d")
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    if spark is not None:
        print("start spark success!")

    # 读取数据
    data = spark.sql(f"""
            select 
                col_name
            from table_name
            where dt = '{yesterday_date}'
        """)

    # 处理数据1
    def partition_process(data, columns):
        data = pd.DataFrame(data, columns=columns)
        result = '' # pandas处理过程
        result = result.fillna('').astype(str)
        return result.to_dict(orient="records")

    columns = []
    result_rdd = data.repartition(100).rdd.mapPartitions(lambda x: partition_process(x, columns))
    result = spark.createDataFrame(result_rdd)
    result.createReplaceTempView("result")


    # 写入结果表
    insert_table_sql = f"""
        insert overwrite table {target_table} partition (dt)
        select 
        from result
    """
    spark.sql("SET hive.exec.dynamic.partition.mode=nonstrict")
    spark.sql(insert_table_sql)
    print('job done.')


#
# .sh文件
#
# time=$(date "+%Y_%m_%d_%H_%M_%S")
#
# cd /home/mkt_ml/……/algo_conf/
#
# spark-submit --master yarn --deploy-mode client --name spark_test \
#     --driver-memory 10g --executor-memory 10g \
#     --executor-cores 10 --num-executors 10 \
#     --queue mkt_ml \
#     --conf "spark.executor.memoryOverhead=10g" \
#     --conf "spark.dynamicAllocation.maxExecutors=80" \
#     --conf spark.yarn.dist.archives=hdfs:///home/mkt_ml/anaconda3_python37.zip#py37env \
#     --conf spark.pyspark.python=py37env/anaconda3/bin/python \
#     --conf spark.pyspark.driver.python=/home/mkt_ml/miniconda3/envs/usr_lib_anaconda3_base/bin/python \
#     --py-files ***.zip \
#     ****.py --timestamp ${time}


