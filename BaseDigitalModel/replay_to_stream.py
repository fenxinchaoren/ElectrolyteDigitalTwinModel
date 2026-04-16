import math
import json
import time
from pathlib import Path

import pymysql
from pymysql.cursors import DictCursor


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "getData_database_info.json"
SOURCE_TABLE = "dataclean_yanzheng"
STREAM_TABLE = "dataclean_stream"
INITIAL_ROWS = 500
BATCH_SIZE = 100
INTERVAL_SECONDS = 0.2
WARMUP_SECONDS_AFTER_PRELOAD = 1.0
RESET_STREAM_TABLE = True
RESET_RESULT_TABLES = True


def quote_identifier(name):
    return f"`{str(name).replace('`', '``')}`"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_connection(config):
    return pymysql.connect(
        host=config["host"],
        user=config["user"],
        password=config["password"],
        port=config["port"],
        database=config["db"],
        charset="utf8mb4",
        autocommit=False,
        cursorclass=DictCursor,
    )


def get_column_names(cursor, table_name):
    cursor.execute(f"SHOW COLUMNS FROM {quote_identifier(table_name)}")
    return [row["Field"] for row in cursor.fetchall()]


def create_stream_table(cursor):
    if RESET_STREAM_TABLE:
        cursor.execute(f"DROP TABLE IF EXISTS {quote_identifier(STREAM_TABLE)}")
    cursor.execute(
        f"CREATE TABLE IF NOT EXISTS {quote_identifier(STREAM_TABLE)} "
        f"LIKE {quote_identifier(SOURCE_TABLE)}"
    )


def clear_online_result_tables(cursor, mark):
    if not RESET_RESULT_TABLES:
        return

    for suffix in (
        "_inv_output",
        "_inv_output_next_time",
        "_multistep_output",
        "_pred_performance",
    ):
        table_name = f"{mark}{suffix}"
        cursor.execute(f"DELETE FROM {quote_identifier(table_name)}")


def fetch_rows(cursor, table_name, column_names, offset, limit):
    cursor.execute(
        f"SELECT {', '.join(quote_identifier(column) for column in column_names)} "
        f"FROM {quote_identifier(table_name)} "
        "ORDER BY `id` "
        "LIMIT %s OFFSET %s",
        (limit, offset),
    )
    return cursor.fetchall()


def insert_rows(cursor, table_name, column_names, rows):
    if not rows:
        return

    sql = (
        f"REPLACE INTO {quote_identifier(table_name)} "
        f"({', '.join(quote_identifier(column) for column in column_names)}) "
        f"VALUES ({', '.join(['%s'] * len(column_names))})"
    )
    values = [[row.get(column) for column in column_names] for row in rows]
    cursor.executemany(sql, values)


def main():
    config = load_config()
    mark = config["mark"]
    start_time = time.time()

    with get_connection(config) as connection:
        with connection.cursor() as cursor:
            source_columns = get_column_names(cursor, SOURCE_TABLE)
            create_stream_table(cursor)
            clear_online_result_tables(cursor, mark)

            cursor.execute(f"SELECT COUNT(*) AS row_count FROM {quote_identifier(SOURCE_TABLE)}")
            total_rows = int(cursor.fetchone()["row_count"])

            preload_rows = min(INITIAL_ROWS, total_rows)
            initial_batch = fetch_rows(cursor, SOURCE_TABLE, source_columns, 0, preload_rows)
            insert_rows(cursor, STREAM_TABLE, source_columns, initial_batch)
            connection.commit()

            print(
                f"Prepared {STREAM_TABLE} with {preload_rows}/{total_rows} initial rows from {SOURCE_TABLE}."
            )
            print(
                "Start or keep `python e_edge_predict.py` running now. "
                f"Waiting {WARMUP_SECONDS_AFTER_PRELOAD:.1f}s before streaming the remaining rows..."
            )

        time.sleep(WARMUP_SECONDS_AFTER_PRELOAD)

        inserted_rows = preload_rows
        while inserted_rows < total_rows:
            with connection.cursor() as cursor:
                batch_rows = fetch_rows(cursor, SOURCE_TABLE, source_columns, inserted_rows, BATCH_SIZE)
                if not batch_rows:
                    break
                insert_rows(cursor, STREAM_TABLE, source_columns, batch_rows)
                connection.commit()
                inserted_rows += len(batch_rows)

            if inserted_rows % 500 == 0 or inserted_rows == total_rows:
                elapsed = max(time.time() - start_time, 1e-6)
                streamed_after_preload = max(inserted_rows - preload_rows, 0)
                speed = streamed_after_preload / elapsed if streamed_after_preload else 0.0
                remaining_rows = max(total_rows - inserted_rows, 0)
                eta_seconds = remaining_rows / speed if speed > 0 else float("inf")
                eta_text = (
                    f"{math.ceil(eta_seconds)}s"
                    if math.isfinite(eta_seconds)
                    else "unknown"
                )
                print(
                    f"Streamed {inserted_rows}/{total_rows} rows into {STREAM_TABLE} "
                    f"at ~{speed:.1f} rows/s, ETA {eta_text}."
                )
            time.sleep(INTERVAL_SECONDS)

    print(f"Replay finished. {STREAM_TABLE} now contains {total_rows} rows.")


if __name__ == "__main__":
    main()
