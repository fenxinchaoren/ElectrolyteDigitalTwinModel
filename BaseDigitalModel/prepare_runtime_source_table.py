import json
from decimal import Decimal

import pymysql
from pymysql.cursors import DictCursor


TARGET_TABLE = "zhiqing_control_runtime"
SOURCE_TABLES = [
    "zhiqing_control_2025_03_07",
    "zhiqing_control_2025_03_08",
    "zhiqing_control_2025_03_09",
    "zhiqing_control_2025_03_10",
    "zhiqing_control_2025_03_11",
    "zhiqing_control_2025_03_12",
    "zhiqing_control_2025_03_13",
    "zhiqing_control_2025_03_14",
    "zhiqing_control_2025_04_10",
    "zhiqing_control_2025_04_14",
    "zhiqing_control_2025_04_15",
]
DROP_COLUMNS = {
    "_time",
    "SetFlowPumpSET1",
    "SetFrequencyManualPumpSet1",
    "ActFlowPumpSET1",
}
INSERT_BATCH_SIZE = 2000
FLOAT_TYPE_KEYWORDS = ("float", "double", "decimal", "real")


def quote_identifier(name):
    return f"`{str(name).replace('`', '``')}`"


def load_database_info():
    with open("getData_database_info.json", "r", encoding="utf-8") as file:
        return json.load(file)


def get_source_tables():
    if not SOURCE_TABLES:
        raise ValueError("No source tables configured in prepare_runtime_source_table.py")
    return SOURCE_TABLES


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


def get_table_columns(cursor, table_name):
    cursor.execute(f"SHOW COLUMNS FROM {quote_identifier(table_name)}")
    return cursor.fetchall()


def is_float_column(column_type):
    lowered = column_type.lower()
    return any(keyword in lowered for keyword in FLOAT_TYPE_KEYWORDS)


def normalize_number(value):
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def is_zero_or_empty(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    number = normalize_number(value)
    if number is not None:
        return number == 0.0
    return False


def build_target_schema(reference_columns):
    retained_columns = [column for column in reference_columns if column["Field"] not in DROP_COLUMNS]
    retained_names = [column["Field"] for column in retained_columns]
    float_column_names = [
        column["Field"] for column in retained_columns if is_float_column(column["Type"])
    ]
    return retained_columns, retained_names, float_column_names


def validate_model_columns(database_info, retained_names):
    input_output_vars_info = database_info.get("input_output_vars_info", {})
    required_columns = set(input_output_vars_info.get("output_var", []))
    required_columns.update(input_output_vars_info.get("linearInput_vars_orders", {}).keys())
    required_columns.update(input_output_vars_info.get("nonlinearInput_vars_orders", {}).keys())

    missing_columns = sorted(required_columns - set(retained_names))
    if missing_columns:
        raise ValueError(
            "Runtime table would miss model-required columns: " + ", ".join(missing_columns)
        )


def create_target_table(cursor, retained_columns):
    cursor.execute(f"DROP TABLE IF EXISTS {quote_identifier(TARGET_TABLE)}")

    column_sql_parts = [
        "`id` BIGINT NOT NULL AUTO_INCREMENT",
        "`time` DATETIME NULL",
    ]
    for column in retained_columns:
        null_sql = "NULL" if column["Null"] == "YES" else "NOT NULL"
        default_sql = ""
        if column["Default"] is not None:
            default_value = str(column["Default"]).replace("'", "''")
            default_sql = f" DEFAULT '{default_value}'"
        extra_sql = f" {column['Extra']}" if column["Extra"] else ""
        column_sql_parts.append(
            f"{quote_identifier(column['Field'])} {column['Type']} {null_sql}{default_sql}{extra_sql}"
        )

    create_sql = (
        f"CREATE TABLE {quote_identifier(TARGET_TABLE)} ("
        + ", ".join(column_sql_parts)
        + ", PRIMARY KEY (`id`), KEY `idx_time` (`time`)"
        + ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
    )
    cursor.execute(create_sql)


def row_should_be_kept(row_data, retained_names, float_column_names):
    retained_values = [row_data.get(column_name) for column_name in retained_names]
    if all(is_zero_or_empty(value) for value in retained_values):
        return False

    float_values = [row_data.get(column_name) for column_name in float_column_names]
    if float_values and all(is_zero_or_empty(value) for value in float_values):
        return False

    return True


def batch_insert_rows(cursor, retained_names, rows):
    if not rows:
        return

    insert_columns = ["time"] + retained_names
    placeholders = ", ".join(["%s"] * len(insert_columns))
    insert_sql = (
        f"INSERT INTO {quote_identifier(TARGET_TABLE)} "
        f"({', '.join(quote_identifier(column) for column in insert_columns)}) "
        f"VALUES ({placeholders})"
    )
    cursor.executemany(insert_sql, rows)


def get_exact_row_count(cursor, table_name):
    cursor.execute(f"SELECT COUNT(*) AS row_count FROM {quote_identifier(table_name)}")
    return cursor.fetchone()["row_count"]


def append_source_rows(cursor, source_table, retained_names, float_column_names):
    select_columns = ["_time"] + retained_names
    select_sql = (
        f"SELECT {', '.join(quote_identifier(column) for column in select_columns)} "
        f"FROM {quote_identifier(source_table)} "
        "ORDER BY `_time`"
    )
    cursor.execute(select_sql)

    inserted_rows = 0
    skipped_rows = 0
    batch_rows = []
    for row in cursor.fetchall():
        if not row_should_be_kept(row, retained_names, float_column_names):
            skipped_rows += 1
            continue

        batch_rows.append([row.get("_time")] + [row.get(column_name) for column_name in retained_names])
        if len(batch_rows) >= INSERT_BATCH_SIZE:
            batch_insert_rows(cursor, retained_names, batch_rows)
            inserted_rows += len(batch_rows)
            batch_rows = []

    if batch_rows:
        batch_insert_rows(cursor, retained_names, batch_rows)
        inserted_rows += len(batch_rows)

    return inserted_rows, skipped_rows


def build_runtime_table(database_info):
    source_tables = get_source_tables()

    with get_connection(database_info) as connection:
        with connection.cursor() as cursor:
            reference_columns = get_table_columns(cursor, source_tables[0])
            retained_columns, retained_names, float_column_names = build_target_schema(reference_columns)
            validate_model_columns(database_info, retained_names)
            create_target_table(cursor, retained_columns)

            total_inserted = 0
            total_skipped = 0
            for source_table in source_tables:
                inserted_rows, skipped_rows = append_source_rows(
                    cursor, source_table, retained_names, float_column_names
                )
                total_inserted += inserted_rows
                total_skipped += skipped_rows
                print(
                    f"Merged {source_table} -> {TARGET_TABLE}: "
                    f"inserted {inserted_rows}, skipped {skipped_rows}"
                )

            connection.commit()
            exact_row_count = get_exact_row_count(cursor, TARGET_TABLE)
            print(
                f"Prepared merged runtime table {TARGET_TABLE} in database {database_info['db']} "
                f"with {total_inserted} rows kept and {total_skipped} rows skipped."
            )
            print(f"Exact row count in {TARGET_TABLE}: {exact_row_count}")


if __name__ == "__main__":
    build_runtime_table(load_database_info())
