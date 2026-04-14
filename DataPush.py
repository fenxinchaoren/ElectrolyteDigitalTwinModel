import csv
import hashlib
import re
from datetime import datetime
from pathlib import Path


try:
    import pymysql
except ImportError:
    pymysql = None

try:
    import mysql.connector
except ImportError:
    mysql = None
else:
    mysql = mysql.connector


BASE_DIR = Path(__file__).resolve().parent
CSV_DIRECTORIES = [BASE_DIR / "DataChange"]
TIME_COLUMN = "_time"
DATETIME_OUTPUT_FORMAT = "%Y-%m-%d %H:%M:%S"
DATETIME_INPUT_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
)
BOOL_NAME_KEYWORDS = ("on", "open", "closed", "close", "enable", "enabled", "flag", "status")
INT_NAME_KEYWORDS = ("count", "index", "idx", "num", "number", "seq", "step", "levelcode")
INSERT_BATCH_SIZE = 5000
RECREATE_TABLES = True
MYSQL_IDENTIFIER_MAX_LENGTH = 64
SCHEMA_SAMPLE_SIZE = 2000
USE_LOAD_DATA_LOCAL = True

DATABASE_CONFIG = {
    "mark": "sya_zhiqing_djyflow",
    "host": "172.20.68.118",
    "user": "root",
    "password": "uwXXsnbcCRk5",
    "port": 3306,
    "db": "ec_wlzq",
}


def get_connection():
    if pymysql is not None:
        return pymysql.connect(
            host=DATABASE_CONFIG["host"],
            user=DATABASE_CONFIG["user"],
            password=DATABASE_CONFIG["password"],
            port=DATABASE_CONFIG["port"],
            database=DATABASE_CONFIG["db"],
            charset="utf8mb4",
            autocommit=False,
            local_infile=True,
        )

    if mysql is not None:
        return mysql.connect(
            host=DATABASE_CONFIG["host"],
            user=DATABASE_CONFIG["user"],
            password=DATABASE_CONFIG["password"],
            port=DATABASE_CONFIG["port"],
            database=DATABASE_CONFIG["db"],
            charset="utf8mb4",
            allow_local_infile=True,
        )

    raise ImportError(
        "Neither pymysql nor mysql-connector-python is installed. "
        "Install one of them before running DataPush.py."
    )


def quote_identifier(name):
    return f"`{str(name).replace('`', '``')}`"


def sanitize_identifier_text(name):
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized or "col"


def shorten_identifier(name, used_identifiers):
    text = str(name)
    if len(text) <= MYSQL_IDENTIFIER_MAX_LENGTH and text not in used_identifiers:
        used_identifiers.add(text)
        return text

    sanitized = sanitize_identifier_text(text)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    max_base_length = MYSQL_IDENTIFIER_MAX_LENGTH - len(digest) - 1
    candidate = f"{sanitized[:max_base_length].rstrip('_')}_{digest}"

    if not candidate or candidate == f"_{digest}":
        candidate = f"col_{digest}"

    suffix = 1
    unique_candidate = candidate
    while unique_candidate in used_identifiers:
        suffix_text = f"_{suffix}"
        max_candidate_length = MYSQL_IDENTIFIER_MAX_LENGTH - len(suffix_text)
        unique_candidate = f"{candidate[:max_candidate_length]}{suffix_text}"
        suffix += 1

    used_identifiers.add(unique_candidate)
    return unique_candidate


def build_identifier_map(names):
    used_identifiers = set()
    identifier_map = {}
    for name in names:
        identifier_map[name] = shorten_identifier(name, used_identifiers)
    return identifier_map


def clean_cell(value):
    if value is None:
        return ""
    return str(value).strip()


def parse_datetime_value(value):
    text = clean_cell(value)
    if not text:
        return None

    for fmt in DATETIME_INPUT_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def parse_float_value(value):
    text = clean_cell(value)
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def parse_int_value(value):
    number = parse_float_value(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def parse_bool_value(value):
    text = clean_cell(value)
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"true", "t", "yes", "y", "on"}:
        return 1
    if lowered in {"false", "f", "no", "n", "off"}:
        return 0

    number = parse_float_value(text)
    if number is None:
        return None

    if number == 1:
        return 1
    if number == 0:
        return 0
    return None


def looks_like_bool_column(column_name):
    lowered = column_name.lower()
    return any(keyword in lowered for keyword in BOOL_NAME_KEYWORDS)


def looks_like_int_column(column_name):
    lowered = column_name.lower()
    return any(keyword in lowered for keyword in INT_NAME_KEYWORDS)


def infer_column_type(column_name, values):
    non_empty_values = [clean_cell(value) for value in values if clean_cell(value)]
    if column_name == TIME_COLUMN:
        return "DATETIME"

    if not non_empty_values:
        return "FLOAT"

    if all(parse_datetime_value(value) is not None for value in non_empty_values):
        return "DATETIME"

    if looks_like_bool_column(column_name) and all(
        parse_bool_value(value) is not None for value in non_empty_values
    ):
        return "TINYINT(1)"

    numeric_values = [parse_float_value(value) for value in non_empty_values]
    if all(number is not None for number in numeric_values):
        all_integral = all(float(number).is_integer() for number in numeric_values)
        if all_integral and looks_like_int_column(column_name):
            return "INT"
        return "FLOAT"

    return "VARCHAR(255)"


def normalize_value(value, column_type):
    text = clean_cell(value)
    if not text:
        return None

    if column_type == "DATETIME":
        parsed = parse_datetime_value(text)
        if parsed is None:
            raise ValueError(f"Invalid datetime value: {text}")
        return parsed.strftime(DATETIME_OUTPUT_FORMAT)

    if column_type == "TINYINT(1)":
        parsed = parse_bool_value(text)
        if parsed is None:
            raise ValueError(f"Invalid bool value: {text}")
        return parsed

    if column_type == "INT":
        parsed = parse_int_value(text)
        if parsed is None:
            raise ValueError(f"Invalid int value: {text}")
        return parsed

    if column_type == "FLOAT":
        parsed = parse_float_value(text)
        if parsed is None:
            raise ValueError(f"Invalid float value: {text}")
        return parsed

    return text


def read_csv_file(file_path):
    with file_path.open("r", newline="", encoding="utf-8-sig") as source_file:
        reader = csv.DictReader(source_file)
        headers = reader.fieldnames or []
        rows = []
        for row in reader:
            normalized_row = {header: clean_cell(row.get(header)) for header in headers}
            rows.append(normalized_row)

    return headers, rows


def read_csv_headers_and_samples(file_path, sample_size=SCHEMA_SAMPLE_SIZE):
    with file_path.open("r", newline="", encoding="utf-8-sig") as source_file:
        reader = csv.DictReader(source_file)
        headers = reader.fieldnames or []
        sample_rows = []
        for row in reader:
            if len(sample_rows) >= sample_size:
                break
            sample_rows.append({header: clean_cell(row.get(header)) for header in headers})

    return headers, sample_rows


def build_schema(headers, rows):
    return {
        header: infer_column_type(header, [row.get(header, "") for row in rows])
        for header in headers
    }


def create_table(cursor, table_name, headers, schema, column_name_map):
    if RECREATE_TABLES:
        cursor.execute(f"DROP TABLE IF EXISTS {quote_identifier(table_name)}")

    column_sql = []
    for header in headers:
        null_sql = "NOT NULL" if header == TIME_COLUMN else "NULL"
        db_column_name = column_name_map[header]
        column_sql.append(f"{quote_identifier(db_column_name)} {schema[header]} {null_sql}")

    primary_key_sql = ""
    if TIME_COLUMN in headers:
        primary_key_sql = f", PRIMARY KEY ({quote_identifier(column_name_map[TIME_COLUMN])})"

    create_sql = (
        f"CREATE TABLE IF NOT EXISTS {quote_identifier(table_name)} ("
        + ", ".join(column_sql)
        + primary_key_sql
        + ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
    )
    cursor.execute(create_sql)


def batch_iterable(items, batch_size):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def build_datetime_load_expr(variable_name):
    sanitized_value = f"NULLIF(TRIM({variable_name}), '')"
    return (
        "COALESCE("
        f"STR_TO_DATE(REPLACE({sanitized_value}, 'T', ' '), '%Y-%m-%d %H:%i:%s'), "
        f"STR_TO_DATE(REPLACE({sanitized_value}, 'T', ' '), '%Y/%m/%d %H:%i:%s'), "
        f"STR_TO_DATE(REPLACE({sanitized_value}, 'T', ' '), '%Y-%m-%d %H:%i')"
        ")"
    )


def build_load_value_expression(variable_name, column_type):
    sanitized_value = f"NULLIF(TRIM({variable_name}), '')"
    if column_type == "DATETIME":
        return build_datetime_load_expr(variable_name)
    if column_type in {"TINYINT(1)", "INT", "FLOAT", "VARCHAR(255)"}:
        return sanitized_value
    return sanitized_value


def load_rows_with_local_infile(cursor, table_name, headers, schema, column_name_map, file_path):
    variable_names = [f"@v{index}" for index in range(1, len(headers) + 1)]
    variable_sql = ", ".join(variable_names)
    set_sql = ", ".join(
        f"{quote_identifier(column_name_map[header])} = "
        f"{build_load_value_expression(variable_name, schema[header])}"
        for header, variable_name in zip(headers, variable_names)
    )
    file_sql_path = str(file_path.resolve()).replace("\\", "/").replace("'", "''")
    load_sql = (
        f"LOAD DATA LOCAL INFILE '{file_sql_path}' "
        f"INTO TABLE {quote_identifier(table_name)} "
        "CHARACTER SET utf8mb4 "
        "FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' "
        "LINES TERMINATED BY '\\n' "
        "IGNORE 1 LINES "
        f"({variable_sql}) "
        f"SET {set_sql}"
    )
    cursor.execute(load_sql)
    return cursor.rowcount


def insert_rows(connection, cursor, table_name, headers, schema, rows, column_name_map):
    if not rows:
        return

    placeholders = ", ".join(["%s"] * len(headers))
    quoted_headers = ", ".join(quote_identifier(column_name_map[header]) for header in headers)
    insert_sql = (
        f"INSERT INTO {quote_identifier(table_name)} ({quoted_headers}) "
        f"VALUES ({placeholders})"
    )

    if TIME_COLUMN in headers:
        update_columns = [header for header in headers if header != TIME_COLUMN]
        if update_columns:
            update_sql = ", ".join(
                f"{quote_identifier(column_name_map[header])} = "
                f"VALUES({quote_identifier(column_name_map[header])})"
                for header in update_columns
            )
            insert_sql += f" ON DUPLICATE KEY UPDATE {update_sql}"

    payload = []
    for row in rows:
        payload.append([normalize_value(row.get(header, ""), schema[header]) for header in headers])

    for batch in batch_iterable(payload, INSERT_BATCH_SIZE):
        cursor.executemany(insert_sql, batch)
    connection.commit()


def import_csv_file(connection, file_path):
    headers, sample_rows = read_csv_headers_and_samples(file_path)
    if not headers:
        print(f"Skip empty header file: {file_path.name}")
        return

    table_name = file_path.stem
    schema = build_schema(headers, sample_rows)
    column_name_map = build_identifier_map(headers)
    imported_rows = None

    with connection.cursor() as cursor:
        create_table(cursor, table_name, headers, schema, column_name_map)
        if USE_LOAD_DATA_LOCAL:
            try:
                imported_rows = load_rows_with_local_infile(
                    cursor, table_name, headers, schema, column_name_map, file_path
                )
                connection.commit()
            except Exception as exc:
                connection.rollback()
                print(f"LOAD DATA fallback for {file_path.name}: {exc}")

        if imported_rows is None:
            _, rows = read_csv_file(file_path)
            insert_rows(connection, cursor, table_name, headers, schema, rows, column_name_map)
            imported_rows = len(rows)

    renamed_columns = [
        (source_name, target_name)
        for source_name, target_name in column_name_map.items()
        if source_name != target_name
    ]
    if renamed_columns:
        print(f"Shortened column names in {file_path.name}:")
        for source_name, target_name in renamed_columns:
            print(f"  {source_name} -> {target_name}")

    print(f"Imported {file_path.name} -> {table_name} ({imported_rows} rows)")


def iter_csv_files():
    for directory in CSV_DIRECTORIES:
        if not directory.exists():
            print(f"Skip missing directory: {directory}")
            continue

        for file_path in sorted(directory.glob("*.csv")):
            yield file_path


def main():
    csv_files = list(iter_csv_files())
    if not csv_files:
        print("No CSV files found in DataChange.")
        return

    print(
        f"Start importing {len(csv_files)} CSV files to "
        f"{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['db']}"
    )

    connection = get_connection()
    try:
        for file_path in csv_files:
            try:
                import_csv_file(connection, file_path)
            except Exception as exc:
                connection.rollback()
                print(f"Failed to import {file_path.name}: {exc}")
    finally:
        connection.close()

    print("Import finished.")


if __name__ == "__main__":
    main()
