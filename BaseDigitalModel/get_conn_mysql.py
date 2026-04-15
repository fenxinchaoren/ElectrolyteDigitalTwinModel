import pymysql
from dbutils.pooled_db import PooledDB
import numpy as np

def getData_PooledDB(database_info_dict):
    host = database_info_dict['host']
    user = database_info_dict['user']
    password = database_info_dict['password']
    port = database_info_dict['port']
    db = database_info_dict['db']
    poolDB = PooledDB(pymysql, 5, host=host, user=user, password=password, port=port, db=db, charset='utf8')
    return poolDB.connection()


def get_data_from_database(database_info_dict, sql_cloud):
    # 连接数据库，获取非线性的输入数据
    with getData_PooledDB(database_info_dict).cursor() as cur_sapi:
        cur_sapi.execute(sql_cloud)
        data_read = cur_sapi.fetchall()
    # 将tuple转换为array
    data_read = np.array([*data_read])
    # 将倒序转为正序
    data_read = data_read[::-1]
    return data_read