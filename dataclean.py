import json
from pathlib import Path

import pymysql
from pymysql.cursors import DictCursor


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "BaseDigitalModel" / "getData_database_info.json"
SOURCE_TABLE = "zhiqing_control_runtime"
TARGET_TABLE = "dataclean"
INSERT_BATCH_SIZE = 2000


def quote_identifier(name):
    return f"`{str(name).replace('`', '``')}`"


def load_database_info():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_connection(database_info):
    return pymysql.connect(
        host=database_info["host"],
        user=database_info["user"],
        password=database_info["password"],
        port=database_info["port"],
        database=database_info["db"],
        charset="utf8mb4",
        autocommit=False,
        cursorclass=DictCursor,
    )


def normalize_number(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def get_table_columns(cursor, table_name):
    cursor.execute(f"SHOW COLUMNS FROM {quote_identifier(table_name)}")
    return cursor.fetchall()


def get_all_column_names(cursor, table_name):
    return [column["Field"] for column in get_table_columns(cursor, table_name)]


def get_model_columns(database_info):
    input_output = database_info["input_output_vars_info"]
    output_columns = input_output["output_var"]
    current_columns = [
        name
        for name in input_output["nonlinearInput_vars_orders"]
        if name.endswith(".ActualCurrent")
    ]
    return output_columns, current_columns


def create_target_table(cursor):
    cursor.execute(f"DROP TABLE IF EXISTS {quote_identifier(TARGET_TABLE)}")
    cursor.execute(
        f"CREATE TABLE {quote_identifier(TARGET_TABLE)} "
        f"LIKE {quote_identifier(SOURCE_TABLE)}"
    )


def batch_insert_rows(cursor, column_names, rows):
    if not rows:
        return

    insert_sql = (
        f"INSERT INTO {quote_identifier(TARGET_TABLE)} "
        f"({', '.join(quote_identifier(column) for column in column_names)}) "
        f"VALUES ({', '.join(['%s'] * len(column_names))})"
    )
    cursor.executemany(insert_sql, rows)


def is_contradictory(process_flow, total_output_flow):
    return (process_flow > 0 and total_output_flow <= 0) or (
        process_flow <= 0 and total_output_flow > 0
    )


def should_keep_row(row, output_columns, stats):
    set_frequency = normalize_number(row.get("SetFrequencylPumpSet1"))
    process_flow = normalize_number(row.get("ActFlowPumpSET1_process"))
    total_output_flow = sum(normalize_number(row.get(column)) for column in output_columns)

    if set_frequency <= 0:
        stats["stopped_rows"] += 1
        return False

    if process_flow <= 0:
        stats["low_process_rows"] += 1
        return False

    if total_output_flow <= 0:
        stats["low_output_rows"] += 1
        return False

    if is_contradictory(process_flow, total_output_flow):
        stats["contradictory_rows"] += 1
        return False

    return True


def clean_runtime_table(database_info):
    output_columns, _ = get_model_columns(database_info)

    with get_connection(database_info) as connection:
        with connection.cursor() as cursor:
            column_names = get_all_column_names(cursor, SOURCE_TABLE)
            create_target_table(cursor)

            cursor.execute(
                f"SELECT {', '.join(quote_identifier(column) for column in column_names)} "
                f"FROM {quote_identifier(SOURCE_TABLE)} "
                "ORDER BY `id`"
            )

            stats = {
                "source_rows": 0,
                "kept_rows": 0,
                "stopped_rows": 0,
                "low_process_rows": 0,
                "low_output_rows": 0,
                "contradictory_rows": 0,
            }
            batch_rows = []

            for row in cursor.fetchall():
                stats["source_rows"] += 1
                if not should_keep_row(row, output_columns, stats):
                    continue

                batch_rows.append([row.get(column) for column in column_names])
                stats["kept_rows"] += 1

                if len(batch_rows) >= INSERT_BATCH_SIZE:
                    batch_insert_rows(cursor, column_names, batch_rows)
                    batch_rows = []

            if batch_rows:
                batch_insert_rows(cursor, column_names, batch_rows)

            connection.commit()

            cursor.execute(f"SELECT COUNT(*) AS row_count FROM {quote_identifier(TARGET_TABLE)}")
            exact_row_count = cursor.fetchone()["row_count"]

    print("Cleaning rules:")
    print("  1. SetFrequencylPumpSet1 > 0")
    print("  2. ActFlowPumpSET1_process > 0")
    print("  3. sum(string01~21.ElectrolyteFlowAverage) > 0")
    print("  4. process flow and cluster-flow sum are not contradictory")
    print(
        "Cleaning summary:"
        f" source_rows={stats['source_rows']},"
        f" kept_rows={stats['kept_rows']},"
        f" stopped_rows={stats['stopped_rows']},"
        f" low_process_rows={stats['low_process_rows']},"
        f" low_output_rows={stats['low_output_rows']},"
        f" contradictory_rows={stats['contradictory_rows']}"
    )
    print(f"Exact row count in {TARGET_TABLE}: {exact_row_count}")


if __name__ == "__main__":
    clean_runtime_table(load_database_info())
