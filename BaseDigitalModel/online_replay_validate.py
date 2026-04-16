import json
from pathlib import Path

import matplotlib.pyplot as plt
import pymysql
from pymysql.cursors import DictCursor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "getData_database_info.json"
OUTPUT_DIR = BASE_DIR / "online_replay_results"
SOURCE_TABLE = "dataclean_yanzheng"


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


def fetch_joined_rows(config):
    mark = config["mark"]
    output_column = config["input_output_vars_info"]["output_var"][0]
    inv_output_table = f"{mark}_inv_output"

    sql = f"""
        SELECT
            s.id AS source_id,
            s.source_id AS raw_source_id,
            s.time AS source_time,
            s.{quote_identifier(output_column)} AS source_output,
            o.inv_real,
            o.inv_pred,
            o.inv_pred_cloud,
            o.yk_linear_pred,
            o.timestep,
            o.n_hidden,
            o.n_layer,
            o.trainable_params,
            o.nontrainable_params,
            o.total_params,
            o.neural_networks_used
        FROM {quote_identifier(SOURCE_TABLE)} AS s
        JOIN (
            SELECT t1.*
            FROM {quote_identifier(inv_output_table)} AS t1
            JOIN (
                SELECT time, MAX(id) AS max_id
                FROM {quote_identifier(inv_output_table)}
                GROUP BY time
            ) AS t2
            ON t1.id = t2.max_id
        ) AS o
        ON s.time = o.time
        ORDER BY s.id
    """

    with get_connection(config) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            return cursor.fetchall()


def compute_metrics(y_true, y_pred):
    return {
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def plot_rows(rows, output_path):
    x_axis = [int(row["source_id"]) for row in rows]
    real_output = [float(row["source_output"]) for row in rows]
    edge_prediction = [float(row["inv_pred"]) for row in rows]
    cloud_prediction = [float(row["inv_pred_cloud"]) for row in rows]
    linear_prediction = [float(row["yk_linear_pred"]) for row in rows]

    plt.figure(figsize=(15, 6))
    plt.plot(x_axis, real_output, label="Real Output", linewidth=2)
    plt.plot(x_axis, edge_prediction, label="Edge Prediction", linewidth=1.6)
    plt.plot(x_axis, cloud_prediction, label="Cloud Prediction", linewidth=1.8)
    plt.plot(x_axis, linear_prediction, label="Linear Prediction", linewidth=1.4, alpha=0.8)
    plt.title("Online Replay Validation: Real vs Predicted Output")
    plt.xlabel("ID")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_csv(rows, output_path):
    import csv

    if not rows:
        return
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    config = load_config()
    mark = config["mark"]
    OUTPUT_DIR.mkdir(exist_ok=True)

    rows = fetch_joined_rows(config)
    if not rows:
        print(
            f"No joined validation rows found between {SOURCE_TABLE} and {mark}_inv_output. "
            "Run e_edge_predict.py and replay_to_stream.py first."
        )
        return

    y_true = [float(row["source_output"]) for row in rows]
    edge_pred = [float(row["inv_pred"]) for row in rows]
    cloud_pred = [float(row["inv_pred_cloud"]) for row in rows]
    linear_pred = [float(row["yk_linear_pred"]) for row in rows]

    edge_metrics = compute_metrics(y_true, edge_pred)
    cloud_metrics = compute_metrics(y_true, cloud_pred)
    linear_metrics = compute_metrics(y_true, linear_pred)

    csv_path = OUTPUT_DIR / f"{mark}_online_replay_validation.csv"
    png_path = OUTPUT_DIR / f"{mark}_online_replay_trend.png"
    save_csv(rows, csv_path)
    plot_rows(rows, png_path)

    print(f"Joined validation rows: {len(rows)}")
    print(f"Validation CSV saved to: {csv_path}")
    print(f"Trend plot saved to: {png_path}")
    print("Edge metrics:")
    print(f"  rmse: {edge_metrics['rmse']:.6f}")
    print(f"  mae: {edge_metrics['mae']:.6f}")
    print(f"  mape: {edge_metrics['mape']:.6f}")
    print(f"  r2: {edge_metrics['r2']:.6f}")
    print("Cloud metrics:")
    print(f"  rmse: {cloud_metrics['rmse']:.6f}")
    print(f"  mae: {cloud_metrics['mae']:.6f}")
    print(f"  mape: {cloud_metrics['mape']:.6f}")
    print(f"  r2: {cloud_metrics['r2']:.6f}")
    print("Linear metrics:")
    print(f"  rmse: {linear_metrics['rmse']:.6f}")
    print(f"  mae: {linear_metrics['mae']:.6f}")
    print(f"  mape: {linear_metrics['mape']:.6f}")
    print(f"  r2: {linear_metrics['r2']:.6f}")


if __name__ == "__main__":
    main()
