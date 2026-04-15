import json
from pathlib import Path

import matplotlib.pyplot as plt
import pymysql


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "getData_database_info.json"
OUTPUT_DIR = BASE_DIR / "plots"


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
    )


def fetch_inv_output_rows(config, limit=500):
    mark = config["mark"]
    table_name = f"{mark}_inv_output"
    sql = f"""
        SELECT time, inv_real, inv_pred, inv_pred_cloud
        FROM `{table_name}`
        ORDER BY id DESC
        LIMIT %s
    """
    with get_connection(config) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
    return list(reversed(rows))


def fetch_pred_performance_rows(config, limit=50):
    mark = config["mark"]
    table_name = f"{mark}_pred_performance"
    sql = f"""
        SELECT time, rmse, mae, mape, r2
        FROM `{table_name}`
        ORDER BY id DESC
        LIMIT %s
    """
    with get_connection(config) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
    return list(reversed(rows))


def plot_inv_output(rows, output_path):
    times = [row[0] for row in rows]
    inv_real = [float(row[1]) if row[1] is not None else None for row in rows]
    inv_pred = [float(row[2]) if row[2] is not None else None for row in rows]
    inv_pred_cloud = [float(row[3]) if row[3] is not None else None for row in rows]
    unique_times = len(set(times))
    use_index_axis = len(rows) <= 5 or unique_times <= 1
    x_axis = list(range(1, len(rows) + 1)) if use_index_axis else times

    plt.figure(figsize=(14, 6))
    plt.plot(x_axis, inv_real, label="Real Output", linewidth=2, marker="o")
    plt.plot(x_axis, inv_pred, label="Edge Prediction", linewidth=1.8, marker="o")
    plt.plot(x_axis, inv_pred_cloud, label="Cloud Prediction", linewidth=1.8, alpha=0.9, marker="o")
    plt.title("Real vs Model Output")
    plt.xlabel("Sample Index" if use_index_axis else "Time")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(alpha=0.25)
    if use_index_axis:
        plt.xticks(x_axis)
    else:
        plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def print_latest_metrics(rows):
    if not rows:
        print("No rows found in pred_performance.")
        return

    latest = rows[-1]
    print("Latest metrics:")
    print(f"  time: {latest[0]}")
    print(f"  rmse: {latest[1]}")
    print(f"  mae: {latest[2]}")
    print(f"  mape: {latest[3]}")
    print(f"  r2: {latest[4]}")


def main():
    config = load_config()
    OUTPUT_DIR.mkdir(exist_ok=True)

    inv_rows = fetch_inv_output_rows(config)
    if not inv_rows:
        print(f"No data found in {config['mark']}_inv_output. Nothing to plot.")
        return

    plot_path = OUTPUT_DIR / f"{config['mark']}_trend.png"
    plot_inv_output(inv_rows, plot_path)
    print(f"Trend plot saved to: {plot_path}")
    print(f"Plotted {len(inv_rows)} inv_output rows.")

    metrics_rows = fetch_pred_performance_rows(config)
    print_latest_metrics(metrics_rows)


if __name__ == "__main__":
    main()
