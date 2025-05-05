import json
import importlib
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
import os
import copy
import pyspark.sql.functions as F

class Config:
    """
    将文件解析成可以引用类属性的config类
    """
    def __init__(self, path, file_type='json'):
        self.operate_function = None
        config = {}
        if file_type == 'json':
            if os.path.exists(path):
                print('find file')
            else:
                print('not find file')
            with open(path, 'r') as fcc_file:
                config = json.load(fcc_file)
        if config.get("operate_functions") is not None:
            self.operate_function = config.get("operate_functions")
        else:
            raise Exception("no operators")

        if config.get("read_config") is not None:
            self.read_config = config.get("read_config")
            if self.read_config.get("sql") is None or len(self.read_config.get("sql")) == 0:
                for item in ["source_table", "conditions", "columns", "keys"]:
                    if self.read_config.get(item) is None:
                        raise Exception(f"missing {item} in read_config")
        else:
            raise Exception("miss source table if config")

        if config.get("write_config") is not None:
            self.write_config = config.get("write_config")
            for item in ["target", "partitions"]:
                if self.write_config.get(item) is None:
                    raise Exception(f"missing {item} in write_config")
        else:
            raise Exception("miss write config in config")


class Executor:

    def __init__(self, config_file_path, file_type='json', sql_args=None, spark_args=None):
        self.spark = None
        self._start_spark()
        if self.spark is not None:
            print('start spark success!')
        if "hadoop" in config_file_path:
            print(f"configuration is from hdfs: {config_file_path} \n")
            file_name = config_file_path.split("/")[-1]
            self._get_hdfs_file(config_file_path, file_name)
            config_file_path = file_name
        print("config_file_path:", config_file_path)
        self.config = Config(config_file_path, file_type=file_type)
        self.operate_func_list = []
        self.operate_func_arg_list = []
        self.read_sql = ""
        self.input_list = []
        self.out_list = []
        self.input_columns = []
        self.output_columns = []
        self.sql_args = sql_args
        self.keys = []
        if spark_args is not None and len(spark_args) > 0:
            self.spark_args = {item.split("=")[0]: item.split("=")[1] for item in spark_args.split(',')}
        else:
            self.spark_args = {}
        print("sql_args:", sql_args)
        print("spark_args:", spark_args)

    def handle(self):
        """串流程"""
        self.load_config()
        df: DataFrame = self.read_data()
        df.show()
        print(f"data loaded, size {df.count()}")

        operate_func_list = copy.deepcopy(self.operate_func_list)
        operate_func_arg_list = copy.deepcopy(self.operate_func_arg_list)
        input_list = copy.deepcopy(self.input_list)
        out_list = copy.deepcopy(self.out_list)

        partition_num = int(self.spark_args.get("partition_num", 5))
        rdd_result = df.select(F.concat_ws("#", *self.keys).alias("PRIMARY_KEY"), *self.input_columns).repartition(partition_num, "PRIMARY_KEY").rdd.mapPartition(
            lambda x: operate(x, operate_func_list, operate_func_arg_list, input_list, out_list)
        )

        df_result = self.spark.createDataFrame(rdd_result)
        df_result.createOrReplaceTempView("df_result")

        result_sql = f"""
            select {','.join([f"split(PRIMARY_KEY, '#')[{i}] as {item}" for i, item in enumerate(self.keys)])}
            ,map({','.join([f"'{o}', {o}" for o in self.output_columns])}) as features
            from df_result
        """

        df_result = self.spark.sql(result_sql)
        df_result.show()
        target = self.config.write_config.get("target")
        partitions = self.config.write_config.get("partitions")
        self.write_hive_by_partition(df_result, output_table_name=target, partitions=partitions)

    def _start_spark(self):
        spark = SparkSession.builder.enableHiveSupport().getOrCreate()
        self.spark = spark

    def load_config(self):
        self._load_operator_configs()
        sql = self._read_data_config()
        self.read_sql = sql

    def read_data(self):
        """根据配置读取hive数据"""
        if self.spark is None:
            raise Exception("spark has not launched")
        sql = self.read_sql.replace(',', '\n ,').replace(" ", "")
        print(f"executing sql: \n {sql}")
        return self.spark.sql(self.read_sql)

    def _load_operator_configs(self):
        """加载配置中需要的方法"""
        operate_function_name = self.config.operate_functions
        for item in operate_function_name:
            func = item["func"]
            input = item["input"].replace(" ", "")
            output = item["output"].replace(" ", "")
            arg = item["arg"]
            module_name = '.'.join(func.split(".")[:-1])
            func_name = func.split(".")[-1]
            func = self._get_func_with_name(module_name, func_name)
            self.operate_func_list.append(func)
            self.input_list.append(input)
            self.out_list.append(output)
            self.operate_func_arg_list.append(arg)
        self.input_columns = sorted(
            list(set([item for item in ','.join(self.input_list).split(',') if "mid_output" not in item]))
        )
        self.output_columns = sorted(
            list(set([item for item in ','.join(self.out_list).split(',') if "mid_output" not in item]))
        )

    @staticmethod
    def _get_func_with_name(module_name, function_name):
        """模块名和函数名的字符串"""
        module = importlib.import_module(module_name)
        if hasattr(module, function_name):
            selected_function = getattr(module, function_name)
            return selected_function
        else:
            raise Exception(f"function {function_name} not found in {module_name}")

    def _read_data_config(self):
        keys = self.config.read_config.get("keys")
        if keys is None or len(keys) == 0:
            raise Exception("empty read key")
        self.keys = keys
        if self.config.read_config.get("sql") is not None and len(self.config.read_config.get("sql")) > 0:
            sql = self.config.read_config.get("sql")
        else:
            keys = self.config.read_config.get("keys")
            columns = []
            for c in self.config.read_config["columns"]:
                columns.append(self._analyse_columns(c))
            conditions = [f"{k} = '{v}'" for k, v in self.config.read_config["conditions"].items()]
            source_table = self.config.read_config["source"]

            sql = f"""
                select {','.join(keys)}, {', '.join(columns)} 
                from {source_table}
                where {' and '.join(conditions)}
            """

        if self.sql_args is not None and len(self.sql_args) > 0:
            sql = self._modify_sql(sql)
        return sql

    @staticmethod
    def _analyse_columns(cls, col):
        expr, col_type = col.get("expr"), col.get("type")
        if col_type == "name":
            return expr
        elif col_type == "process":
            return expr
        elif col_type == "generator":
            try:
                return ', '.join(eval(expr))
            except:
                raise Exception(f"wrong generator expession: {expr}")

    def write_hive_by_partition(self, data, output_table_name, partitions: dict):
        field = ", ".join(map(lambda fields: fields.simpleString().replace(":", " "), data.schema.fields))
        data.createOrReplaceTempView("final_data")
        if partitions:
            create_table_sql = f"create table if not exists {output_table_name} ({field}) PARTITION BY ({','.join([f'{name} string' for name in partitions.keys()])}) stored as orc"
            insert_table_sql = f"""
            insert overwrite table {output_table_name} partition ({','.join([f"{name} = '{p}'" for name, p in partitions.items()])})
                select * from final_data
            """
        else:
            create_table_sql = f"create table if not exists {output_table_name} ({field}) stored as orc "
            insert_table_sql = f"""
            insert overwrite table {output_table_name}
            select * from final_data
            """
        insert_table_sql = self._modify_sql(insert_table_sql)
        self.spark.sql(create_table_sql)
        self.spark.sql(insert_table_sql)


    def _modify_sql(self, sql):
        if self.sql_args:
            modify_vars = {item.split("=")[0]: item.split("=")[1] for item in self.sql_args.replace(" ", "").split(",")}
            for k, v in modify_vars.items():
                assert "@" in k, f"wrong format in args: {k}:{v}"
                sql = sql.replace((k, v))
        return sql

    def _get_hdfs_file(self, path, local_path):
        print(f"get file {path}")
        with open(local_path, 'wb') as local_file:
            local_file.write(self.spark.sparkContext.binaryFiles(path).first()[1])
        if os.path.exists(local_path):
            print("success")

def operate(data, func_factory_list, arg_list, input_list, output_list):
    input_columns = ["PRIMARY_KEY"] + sorted(
        list(set([item for item in ','.join(input_list).split(',') if "mid_output" not in item]))
    )
    output_columns = ["PRIMARY_KEY"] + sorted(
        list(set(item for item in ','.join(output_list).split(',') if "mid_output" not in item))
    )

    df = pd.DataFrame(data, columns=input_columns)
    print(f"分区数量大小为{df.shape[0]}, KEY数量为{df.PRIMARY_KEY.unique().shape[0]}")
    func_list = [func[arg] for func, arg in zip(func_factory_list, arg_list)]
    print(f"需要运行的函数：{','.join([f.__name__ for f in func_list])}")

    df_result = []
    for k, tmpdf in df.groupby("PRIMARY_KEY"):
        stream_operator = StreamFeatureCalculator(input_list, func_list, output_list, tmpdf)
        result = stream_operator.run()
        result["PRIMARY_KEY"] = k
        df_result.append(result)
    df_result = pd.DataFrame(df_result).loc[:, output_columns]

    for row in df_result.to_dict('records'):
        yield row

class StreamFeatureCalculator:

    def __int__(self, inputs, funcs, outputs, data: pd.DataFrame):
        self.inputs = inputs
        self.funcs = funcs
        self.outputs = outputs
        self.data = data
        self.mid_outputs = {}
        self.results = {}

    def run(self):
        for i, f, o in zip(self.inputs, self.funcs, self.outputs):
            func_inputs = self._get_input(i)
            result = f(*func_inputs)
            result = result if isinstance(result, tuple) else tuple([result])
            assert len(result) == len(o.split(',')), f"different size of output and out_name when running {f.__name__}"
            self._handle_func_result(result, o)
        return self.results

    def _get_input(self, input_names):
        if not isinstance(input_names, list):
            input_names = [input_names]
        inputs = []
        for i in input_names:
            if "mid_output" in i:
                assert i in self.mid_outputs, f"input {i} has not been produced, check the config"
                inputs.append(self.mid_outputs.get(i))
            else:
                if ',' in i:
                    i = i.split(',')
                data = self.data.loc[:,i]
                inputs.append(data)
        return inputs

    def _handle_func_result(self, outputs, names):
        for o, n in zip(outputs, names.split(',')):
            if "mid_output" in n:
                assert n not in self.mid_outputs, f"duplicate produce mid_output:{o}"
                self.mid_outputs[n] = o
            else:
                assert n not in self.results, f"duplicate produce output:{o}"
                self.results[n] = o











