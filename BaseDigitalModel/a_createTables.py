from func_createTable import createTables
from Function import getData_database_info
user_mark = getData_database_info['mark']

if __name__ == '__main__':
    # 创建数据库表格
    createTables(mark=user_mark)

