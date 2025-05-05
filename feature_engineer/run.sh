time=$(date "+%Y_%m_%d_%H_%M_%S")
config_file_path='/home/...'
sql_args=''
spark_args=''

cd /home/...

spark-submit --

    ...
    --py-files src.zip main.py --timestamp ${time} --config_file_path ${config_file_path}


#
#参数形式举例：
#spark_args= 'partition_num=20, cores=10'
#sql_args={"@dt":"2024-01-01", "@treatment":"3"}
#与read_config.sql配合使用：
#"sql": " select userid, treatment, label from table_name where dt = '@dt' and treatment = '@treatment'"