import json
from pathlib import Path

import pymysql
from pymysql.cursors import DictCursor


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "BaseDigitalModel" / "getData_database_info.json"
SOURCE_TABLE = "zhiqing_control_runtime"
TARGET_TABLE = "dataclean"
VALIDATION_TABLE = "dataclean_yanzheng"
TOTAL_FLOW_COLUMN = "ActFlowPumpSET1_process"
AVERAGE_FLOW_COLUMN = "ActualAverageFlow"
ACTIVE_CLUSTER_COUNT_COLUMN = "ActiveClusterCount"
INSERT_BATCH_SIZE = 2000
ACTIVE_CLUSTER_FLOW_THRESHOLD = 3.5


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


def get_cluster_flow_columns(column_names):
    return sorted(
        [name for name in column_names if name.endswith(".ElectrolyteFlowAverage")]
    )


def get_cluster_flow_values(row, cluster_flow_columns):
    return [normalize_number(row.get(column)) for column in cluster_flow_columns]


def create_target_table(cursor):
    cursor.execute(f"DROP TABLE IF EXISTS {quote_identifier(TARGET_TABLE)}")
    cursor.execute(
        f"CREATE TABLE {quote_identifier(TARGET_TABLE)} "
        f"LIKE {quote_identifier(SOURCE_TABLE)}"
    )
    cursor.execute(
        f"ALTER TABLE {quote_identifier(TARGET_TABLE)} "
        f"ADD COLUMN {quote_identifier(AVERAGE_FLOW_COLUMN)} DOUBLE NULL "
        "AFTER `ActFlowPumpSET1_process`"
    )
    cursor.execute(
        f"ALTER TABLE {quote_identifier(TARGET_TABLE)} "
        f"ADD COLUMN {quote_identifier(ACTIVE_CLUSTER_COUNT_COLUMN)} INT NULL "
        f"AFTER {quote_identifier(AVERAGE_FLOW_COLUMN)}"
    )


def batch_insert_rows(cursor, table_name, column_names, rows):
    if not rows:
        return

    insert_sql = (
        f"INSERT INTO {quote_identifier(table_name)} "
        f"({', '.join(quote_identifier(column) for column in column_names)}) "
        f"VALUES ({', '.join(['%s'] * len(column_names))})"
    )
    cursor.executemany(insert_sql, rows)


def create_validation_table(cursor):
    cursor.execute(f"DROP TABLE IF EXISTS {quote_identifier(VALIDATION_TABLE)}")
    cursor.execute(
        f"CREATE TABLE {quote_identifier(VALIDATION_TABLE)} "
        f"LIKE {quote_identifier(TARGET_TABLE)}"
    )
    cursor.execute(
        f"ALTER TABLE {quote_identifier(VALIDATION_TABLE)} "
        "ADD COLUMN `source_id` INT NULL AFTER `id`"
    )


def rebuild_validation_table(cursor):
    create_validation_table(cursor)
    validation_columns = get_all_column_names(cursor, VALIDATION_TABLE)
    source_columns = [column for column in validation_columns if column not in {"id", "source_id"}]
    insert_column_names = ["id", "source_id"] + source_columns

    cursor.execute(
        f"SELECT {', '.join(quote_identifier(column) for column in get_all_column_names(cursor, TARGET_TABLE))} "
        f"FROM {quote_identifier(TARGET_TABLE)} "
        "ORDER BY `id`"
    )

    batch_rows = []
    next_id = 1
    for row in cursor.fetchall():
        batch_rows.append(
            [next_id, row.get("id")] + [row.get(column) for column in source_columns]
        )
        next_id += 1

        if len(batch_rows) >= INSERT_BATCH_SIZE:
            batch_insert_rows(cursor, VALIDATION_TABLE, insert_column_names, batch_rows)
            batch_rows = []

    if batch_rows:
        batch_insert_rows(cursor, VALIDATION_TABLE, insert_column_names, batch_rows)


def is_contradictory(process_flow, total_output_flow):
    return (process_flow > 0 and total_output_flow <= 0) or (
        process_flow <= 0 and total_output_flow > 0
    )


def get_active_cluster_flows(row, cluster_flow_columns):
    return [
        normalize_number(row.get(column))
        for column in cluster_flow_columns
        if normalize_number(row.get(column)) > ACTIVE_CLUSTER_FLOW_THRESHOLD
    ]


def calculate_total_cluster_flow(cluster_flows):
    return sum(cluster_flows)


def calculate_actual_average_flow(active_flows):
    positive_flows = active_flows
    if not positive_flows:
        return 0.0
    return sum(positive_flows) / len(positive_flows)


def should_keep_row(row, total_cluster_flow, stats):
    set_frequency = normalize_number(row.get("SetFrequencylPumpSet1"))

    if set_frequency <= 0:
        stats["stopped_rows"] += 1
        return False

    if total_cluster_flow <= 0:
        stats["zero_total_flow_rows"] += 1
        return False

    return True


def clean_runtime_table(database_info):
    with get_connection(database_info) as connection:
        with connection.cursor() as cursor:
            column_names = get_all_column_names(cursor, SOURCE_TABLE)
            cluster_flow_columns = get_cluster_flow_columns(column_names)
            if not cluster_flow_columns:
                raise ValueError(
                    f"No cluster flow columns like stringXX.ElectrolyteFlowAverage found in {SOURCE_TABLE}."
                )
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
                "zero_total_flow_rows": 0,
            }
            batch_rows = []
            insert_column_names = column_names + [AVERAGE_FLOW_COLUMN, ACTIVE_CLUSTER_COUNT_COLUMN]

            for row in cursor.fetchall():
                stats["source_rows"] += 1
                cluster_flows = get_cluster_flow_values(row, cluster_flow_columns)
                active_flows = get_active_cluster_flows(row, cluster_flow_columns)
                total_cluster_flow = calculate_total_cluster_flow(cluster_flows)
                actual_average_flow = calculate_actual_average_flow(active_flows)
                active_cluster_count = len(active_flows)
                if not should_keep_row(row, total_cluster_flow, stats):
                    continue

                row_values = []
                for column in column_names:
                    if column == TOTAL_FLOW_COLUMN:
                        row_values.append(total_cluster_flow)
                    else:
                        row_values.append(row.get(column))

                batch_rows.append(
                    row_values
                    + [actual_average_flow, active_cluster_count]
                )
                stats["kept_rows"] += 1

                if len(batch_rows) >= INSERT_BATCH_SIZE:
                    batch_insert_rows(cursor, TARGET_TABLE, insert_column_names, batch_rows)
                    batch_rows = []

            if batch_rows:
                batch_insert_rows(cursor, TARGET_TABLE, insert_column_names, batch_rows)

            rebuild_validation_table(cursor)
            connection.commit()

            cursor.execute(f"SELECT COUNT(*) AS row_count FROM {quote_identifier(TARGET_TABLE)}")
            exact_row_count = cursor.fetchone()["row_count"]
            cursor.execute(f"SELECT COUNT(*) AS row_count FROM {quote_identifier(VALIDATION_TABLE)}")
            validation_row_count = cursor.fetchone()["row_count"]

    print("Cleaning rules:")
    print("  1. SetFrequencylPumpSet1 > 0")
    print("  2. ActFlowPumpSET1_process > 0")
    print("     ActFlowPumpSET1_process = sum(all stringXX.ElectrolyteFlowAverage)")
    print(
        f"  3. ActualAverageFlow = sum(cluster flows > {ACTIVE_CLUSTER_FLOW_THRESHOLD}) "
        f"/ count(cluster flows > {ACTIVE_CLUSTER_FLOW_THRESHOLD})"
    )
    print(
        f"  4. ActiveClusterCount = count(cluster flows > {ACTIVE_CLUSTER_FLOW_THRESHOLD})"
    )
    print(
        "Cleaning summary:"
        f" source_rows={stats['source_rows']},"
        f" kept_rows={stats['kept_rows']},"
        f" stopped_rows={stats['stopped_rows']},"
        f" zero_total_flow_rows={stats['zero_total_flow_rows']}"
    )
    print(f"Exact row count in {TARGET_TABLE}: {exact_row_count}")
    print(f"Exact row count in {VALIDATION_TABLE}: {validation_row_count}")


if __name__ == "__main__":
    clean_runtime_table(load_database_info())
