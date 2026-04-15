import pandas as pd
from get_conn_mysql import get_data_from_database
import numpy as np
import time
import json

with open('getData_database_info.json', 'r') as file:
    getData_database_info = json.load(file)


def quote_identifier(name):
    return f"`{str(name).replace('`', '``')}`"


def get_source_tables(database_info_dict):
    table_names = database_info_dict.get('tables', [])
    if table_names:
        return table_names

    table_name = database_info_dict.get('table')
    if table_name:
        return [table_name]

    raise KeyError('No source table configured in getData_database_info.json')


def get_time_column(database_info_dict):
    return database_info_dict.get('time_column', 'time')


def get_order_by_column(database_info_dict):
    return database_info_dict.get('order_by_column', 'id')


def join_quoted_identifiers(names):
    return ','.join(quote_identifier(name) for name in names)


def build_source_query(selected_columns_sql, table_names, order_by_column):
    order_alias = quote_identifier('__order_col__')
    union_sql = ' union all '.join(
        f"""select {quote_identifier(order_by_column)} as {order_alias}, {selected_columns_sql} """
        f"""from {quote_identifier(table_name)}"""
        for table_name in table_names
    )
    return (
        f"""select {selected_columns_sql} from ({union_sql}) as union_data """
        f"""order by {order_alias} desc"""
    )


def get_theta(mark):
    while True:
        try:
            theta = pd.read_csv('./model_cloud_correct/{0}_theta_T.csv'.format(mark), index_col=0).values
            break
        except:
            print("theta_T加载失败，正在重新加载！")
            time.sleep(0.5)
            continue
    return theta


def getRawData_from_DB_for_Train(database_info_dict, data_amount):
    # 获取输入-输出变量的信息
    input_output_vars_info = database_info_dict['input_output_vars_info']
    # 依次提取输出变量、线性输入变量和非线性输入变量
    output_var = input_output_vars_info['output_var']
    linearInput_vars = [key for key in input_output_vars_info['linearInput_vars_orders']]
    nonlinearInput_vars = [key for key in input_output_vars_info['nonlinearInput_vars_orders']]
    # 确定从数据库取哪些变量，保证列表中变量的唯一性
    vars_list = list(dict.fromkeys(output_var + linearInput_vars + nonlinearInput_vars))
    vars_strs = join_quoted_identifiers(vars_list)  # 处理成字符串以便sql语句使用
    table_names = get_source_tables(database_info_dict)
    order_by_column = get_order_by_column(database_info_dict)
    # sql语句，用于从数据库提取数据
    sql_cloud = f"""{build_source_query(vars_strs, table_names, order_by_column)} limit 0,{data_amount}"""
    # 连接数据库，获取所需的数据，返回的是一个二维数组，shape=(data_amount, 变量种类)
    data_read = get_data_from_database(database_info_dict, sql_cloud)
    # 将数据与列名对齐
    data_read_framed = pd.DataFrame(data_read, columns=vars_list)
    # 输出数据、线性输入数据、非线性输入数据分别为
    outputData = data_read_framed[output_var].values
    linearInputData = data_read_framed[linearInput_vars].values
    nonlinearInputData = data_read_framed[nonlinearInput_vars].values
    return linearInputData, nonlinearInputData, outputData


def linearInputData_orders_process_for_Train(input_output_vars_info, linearInputData):
    # 处理线性输入数据、非线性输入数据、输出数据
    linearInput_orders = [value for value in input_output_vars_info['linearInput_vars_orders'].values()]
    order_max = max([item for sublist in linearInput_orders for item in sublist])
    orderedData_len = len(linearInputData) - order_max
    linearInputData_ordered = np.empty(shape=(orderedData_len, 0))
    for j in range(len(linearInput_orders)):
        orders_j = linearInput_orders[j]
        linearInputData_var_j = linearInputData[:, j:j + 1]
        linearInputData_ordered_j = np.empty(shape=(orderedData_len, 0))
        for i in orders_j:
            if i == 0:
                linearInputData_ordered_j = np.append(linearInputData_ordered_j,linearInputData_var_j[-orderedData_len:], axis=1)
            else:
                linearInputData_ordered_j = np.append(linearInputData_ordered_j, linearInputData_var_j[-i-orderedData_len:-i], axis=1)
        linearInputData_ordered = np.append(linearInputData_ordered, linearInputData_ordered_j, axis=1)
    # 判断线性模型输入是否加常数项
    if input_output_vars_info['constant_term']:
        amplitude = input_output_vars_info['constant_term']
        linearInputData_ordered = np.append(linearInputData_ordered, amplitude*np.ones(shape=(len(linearInputData_ordered), 1)), axis=1)
    return linearInputData_ordered


def nonlinearInputData_orders_process_for_Train(input_output_vars_info, nonlinearInputData):
    # 处理线性输入数据、非线性输入数据、输出数据
    nonlinearInput_orders = [value for value in input_output_vars_info['nonlinearInput_vars_orders'].values()]
    order_max = max([item for sublist in nonlinearInput_orders for item in sublist])
    orderedData_len = len(nonlinearInputData) - order_max
    nonlinearInputData_ordered = np.empty(shape=(orderedData_len, 0))
    for j in range(len(nonlinearInput_orders)):
        orders_j = nonlinearInput_orders[j]
        nonlinearInputData_var_j = nonlinearInputData[:, j:j + 1]
        nonlinearInputData_ordered_j = np.empty(shape=(orderedData_len, 0))
        for i in orders_j:
            if i == 0:
                nonlinearInputData_ordered_j = np.append(nonlinearInputData_ordered_j,nonlinearInputData_var_j[-orderedData_len:], axis=1)
            else:
                nonlinearInputData_ordered_j = np.append(nonlinearInputData_ordered_j, nonlinearInputData_var_j[-i-orderedData_len:-i], axis=1)
        nonlinearInputData_ordered = np.append(nonlinearInputData_ordered, nonlinearInputData_ordered_j, axis=1)
    return nonlinearInputData_ordered


def get_inputOutput_data_for_onlineTrain(database_info_dict, data_amount):
    input_output_vars_info = database_info_dict['input_output_vars_info']
    # 获取线性输入数据、非线性输入数据、输出数据
    linearInputData, nonlinearInputData, outputData = getRawData_from_DB_for_Train(database_info_dict, data_amount)
    linearInputData_ordered = linearInputData_orders_process_for_Train(input_output_vars_info, linearInputData)
    nonlinearInputData_ordered = nonlinearInputData_orders_process_for_Train(input_output_vars_info, nonlinearInputData)
    # 获取最短的数据长度，以对数据时刻进行对齐处理
    data_len = min(len(linearInputData_ordered), len(nonlinearInputData_ordered), len(outputData))
    # 对数据进行处理，以组成输入输出对
    linear_input_k_1 = linearInputData_ordered[-data_len:]
    nonlinear_input_k_1 = nonlinearInputData_ordered[-data_len:]
    lables_k = outputData[-data_len:]
    # 线性模型参数列向量
    theta = get_theta(database_info_dict['mark'])
    return linear_input_k_1, nonlinear_input_k_1, lables_k, theta


def getRawData_from_DB_for_Forecast(database_info_dict, data_amount):
    # 获取输入-输出变量的信息
    input_output_vars_info = database_info_dict['input_output_vars_info']
    # 依次提取输出变量、线性输入变量和非线性输入变量
    output_var = input_output_vars_info['output_var']
    linearInput_vars = [key for key in input_output_vars_info['linearInput_vars_orders']]
    nonlinearInput_vars = [key for key in input_output_vars_info['nonlinearInput_vars_orders']]
    # 确定从数据库取哪些变量，保证列表中变量的唯一性
    vars_list = list(dict.fromkeys(output_var + linearInput_vars + nonlinearInput_vars))
    time_column = get_time_column(database_info_dict)
    vars_strs = join_quoted_identifiers([time_column] + vars_list)
    table_names = get_source_tables(database_info_dict)
    order_by_column = get_order_by_column(database_info_dict)
    # sql语句，用于从数据库提取数据
    sql_cloud = f"""{build_source_query(vars_strs, table_names, order_by_column)} limit 0,{data_amount}"""
    # 连接数据库，获取所需的数据，返回的是一个二维数组，shape=(data_amount, 变量种类)
    data_read = get_data_from_database(database_info_dict, sql_cloud)
    time_read = data_read[-1][0]
    data_read = data_read[:, 1:]
    # 将数据与列名对齐
    data_read_framed = pd.DataFrame(data_read, columns=vars_list)
    # 输出数据、线性输入数据、非线性输入数据分别为
    outputData = data_read_framed[output_var].values
    linearInputData = data_read_framed[linearInput_vars].values
    nonlinearInputData = data_read_framed[nonlinearInput_vars].values
    return time_read, linearInputData, nonlinearInputData, outputData


def linearInputData_orders_process_for_Forecast(input_output_vars_info, linearInputData):
    # 处理线性输入数据、非线性输入数据、输出数据
    linearInput_orders = [value for value in input_output_vars_info['linearInput_vars_orders'].values()]
    order_max = max([item for sublist in linearInput_orders for item in sublist])
    orderedData_len = len(linearInputData) - order_max
    linearInputData_ordered = np.empty(shape=(orderedData_len, 0))
    for j in range(len(linearInput_orders)):
        orders_j = linearInput_orders[j]
        linearInputData_var_j = linearInputData[:, j:j + 1]
        linearInputData_ordered_j = np.empty(shape=(orderedData_len, 0))
        for i in orders_j:
            if i == 1:
                linearInputData_ordered_j = np.append(linearInputData_ordered_j,linearInputData_var_j[-orderedData_len:], axis=1)
            else:
                linearInputData_ordered_j = np.append(linearInputData_ordered_j, linearInputData_var_j[-i+1-orderedData_len:-i+1], axis=1)
        linearInputData_ordered = np.append(linearInputData_ordered, linearInputData_ordered_j, axis=1)
    # 判断线性模型输入是否加常数项
    if input_output_vars_info['constant_term']:
        amplitude = input_output_vars_info['constant_term']
        linearInputData_ordered = np.append(linearInputData_ordered, amplitude*np.ones(shape=(len(linearInputData_ordered), 1)), axis=1)
    return linearInputData_ordered


def nonlinearInputData_orders_process_for_Forecast(input_output_vars_info, nonlinearInputData):
    # 处理线性输入数据、非线性输入数据、输出数据
    nonlinearInput_orders = [value for value in input_output_vars_info['nonlinearInput_vars_orders'].values()]
    order_max = max([item for sublist in nonlinearInput_orders for item in sublist])
    orderedData_len = len(nonlinearInputData) - order_max
    nonlinearInputData_ordered = np.empty(shape=(orderedData_len, 0))
    for j in range(len(nonlinearInput_orders)):
        orders_j = nonlinearInput_orders[j]
        nonlinearInputData_var_j = nonlinearInputData[:, j:j + 1]
        nonlinearInputData_ordered_j = np.empty(shape=(orderedData_len, 0))
        for i in orders_j:
            if i == 1:
                nonlinearInputData_ordered_j = np.append(nonlinearInputData_ordered_j,nonlinearInputData_var_j[-orderedData_len:], axis=1)
            else:
                nonlinearInputData_ordered_j = np.append(nonlinearInputData_ordered_j, nonlinearInputData_var_j[-i+1-orderedData_len:-i+1], axis=1)
        nonlinearInputData_ordered = np.append(nonlinearInputData_ordered, nonlinearInputData_ordered_j, axis=1)
    return nonlinearInputData_ordered


def getInputData_at_k(database_info_dict, linearInputData, nonlinearInputData, linear_input_k_1, nonlinear_input_k_1, data_len):
    # 获取输入-输出变量的信息
    input_output_vars_info = database_info_dict['input_output_vars_info']
    linearInput_orders = [value for value in input_output_vars_info['linearInput_vars_orders'].values()]
    linearInput_order_min = min([item for sublist in linearInput_orders for item in sublist])
    nonlinearInput_orders = [value for value in input_output_vars_info['nonlinearInput_vars_orders'].values()]
    nonlinearInput_order_min = min([item for sublist in nonlinearInput_orders for item in sublist])
    order_min = min(linearInput_order_min, nonlinearInput_order_min)
    if order_min == 0:
        linear_input_k = linear_input_k_1
        nonlinear_input_k = nonlinear_input_k_1
    else:
        linearInputData_ordered = linearInputData_orders_process_for_Forecast(input_output_vars_info, linearInputData)
        nonlinearInputData_ordered = nonlinearInputData_orders_process_for_Forecast(input_output_vars_info, nonlinearInputData)
        # 对数据进行处理，以组成输入输出对
        linear_input_k = linearInputData_ordered[-data_len:]
        nonlinear_input_k = nonlinearInputData_ordered[-data_len:]
    return linear_input_k, nonlinear_input_k


def get_inputOutput_data_for_onlineForecast(database_info_dict, data_amount):
    input_output_vars_info = database_info_dict['input_output_vars_info']
    # 获取线性输入数据、非线性输入数据、输出数据
    time_read, linearInputData, nonlinearInputData, outputData = getRawData_from_DB_for_Forecast(database_info_dict, data_amount)
    linearInputData_ordered = linearInputData_orders_process_for_Train(input_output_vars_info, linearInputData)
    nonlinearInputData_ordered = nonlinearInputData_orders_process_for_Train(input_output_vars_info, nonlinearInputData)
    # 获取最短的数据长度，以对数据时刻进行对齐处理
    data_len = min(len(linearInputData_ordered), len(nonlinearInputData_ordered), len(outputData))
    # 对数据进行处理，以组成输入输出对
    linear_input_k_1 = linearInputData_ordered[-data_len:]
    nonlinear_input_k_1 = nonlinearInputData_ordered[-data_len:]
    lables_k = outputData[-data_len:]
    # 获取k时刻的输入数据
    linear_input_k, nonlinear_input_k = getInputData_at_k(database_info_dict, linearInputData, nonlinearInputData, linear_input_k_1, nonlinear_input_k_1, data_len)
    # 线性模型参数列向量
    theta = get_theta(database_info_dict['mark'])

    return time_read, linear_input_k_1, linear_input_k, nonlinear_input_k_1, nonlinear_input_k, lables_k, theta





if __name__ == '__main__':
    # 调试函数
    data_amount = 10

    get_inputOutput_data_for_onlineTrain(getData_database_info, data_amount)

    get_inputOutput_data_for_onlineForecast(getData_database_info, data_amount)


