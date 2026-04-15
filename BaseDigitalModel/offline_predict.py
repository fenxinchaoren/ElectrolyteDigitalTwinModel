import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from user_getData_func import (
    linearInputData_orders_process_for_Train,
    nonlinearInputData_orders_process_for_Train,
)


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "getData_database_info.json"
OUTPUT_DIR = BASE_DIR / "offline_results"
BY_DATE_DIR = OUTPUT_DIR / "by_date"


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
    )


def get_table_names(config):
    table_names = config.get("tables") or []
    if table_names:
        return table_names
    return [config["table"]]


def build_union_query(selected_columns, table_names, order_by_column):
    order_alias = quote_identifier("__order_col__")
    selected_sql = ", ".join(quote_identifier(column) for column in selected_columns)
    union_sql = " union all ".join(
        f"select {quote_identifier(order_by_column)} as {order_alias}, {selected_sql} "
        f"from {quote_identifier(table_name)}"
        for table_name in table_names
    )
    return (
        f"select {selected_sql} from ({union_sql}) as union_data "
        f"order by {order_alias} desc"
    )


def fetch_full_dataset(config):
    input_output = config["input_output_vars_info"]
    output_vars = input_output["output_var"]
    linear_vars = list(input_output["linearInput_vars_orders"].keys())
    nonlinear_vars = list(input_output["nonlinearInput_vars_orders"].keys())
    time_column = config.get("time_column", "time")
    order_by_column = config.get("order_by_column", "id")
    table_names = get_table_names(config)

    selected_columns = [time_column] + list(dict.fromkeys(output_vars + linear_vars + nonlinear_vars))
    query = build_union_query(selected_columns, table_names, order_by_column)

    with get_connection(config) as connection:
        dataframe = pd.read_sql(query, connection)

    time_values = pd.to_datetime(dataframe[time_column])
    output_data = dataframe[output_vars].to_numpy(dtype=float)
    linear_data = dataframe[linear_vars].to_numpy(dtype=float)
    nonlinear_data = dataframe[nonlinear_vars].to_numpy(dtype=float)
    return time_values, linear_data, nonlinear_data, output_data


def load_theta(mark):
    return pd.read_csv(BASE_DIR / "model_cloud_correct" / f"{mark}_theta_T.csv", index_col=0).values


def load_scaler(mark):
    scaler = pd.read_csv(BASE_DIR / "trainData_scaler" / f"{mark}_scaler_train.csv", index_col=0)
    ave = scaler["scaler_ave"].to_numpy(dtype=float)
    std = scaler["scaler_std"].to_numpy(dtype=float)
    return ave, std


def load_model(mark):
    model_path = BASE_DIR / "model_cloud_correct" / f"{mark}_model_cloud_pretrained.h5"
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def get_online_structure(config):
    mark = config["mark"]
    query = (
        f"select timestep, n_hidden, n_layer from {quote_identifier(mark + '_best_para')} "
        "order by id desc limit 1"
    )
    with get_connection(config) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            row = cursor.fetchone()
            if row:
                return int(row[0]), int(row[1]), int(row[2])

            fallback_query = (
                f"select timestep, n_hidden, n_layer from "
                f"{quote_identifier(mark + '_online_model_struc')} "
                "order by id desc limit 1"
            )
            cursor.execute(fallback_query)
            row = cursor.fetchone()

    if not row:
        raise ValueError("No model structure found in best_para or online_model_struc.")
    return int(row[0]), int(row[1]), int(row[2])


def build_offline_predictions(config):
    mark = config["mark"]
    input_output = config["input_output_vars_info"]

    time_values, linear_data, nonlinear_data, output_data = fetch_full_dataset(config)
    linear_ordered = linearInputData_orders_process_for_Train(input_output, linear_data)
    nonlinear_ordered = nonlinearInputData_orders_process_for_Train(input_output, nonlinear_data)
    data_len = min(len(linear_ordered), len(nonlinear_ordered), len(output_data), len(time_values))

    linear_ordered = linear_ordered[-data_len:]
    nonlinear_ordered = nonlinear_ordered[-data_len:]
    output_data = output_data[-data_len:]
    time_values = time_values.iloc[-data_len:]

    theta = load_theta(mark)
    scaler_ave, scaler_std = load_scaler(mark)
    model = load_model(mark)
    timestep, n_hidden, n_layer = get_online_structure(config)

    nonlinear_feature_count = nonlinear_ordered.shape[1]
    input_ave = scaler_ave[:nonlinear_feature_count]
    input_std = scaler_std[:nonlinear_feature_count]
    residual_ave = scaler_ave[nonlinear_feature_count]
    residual_std = scaler_std[nonlinear_feature_count]

    nonlinear_scaled = (nonlinear_ordered - input_ave) / input_std

    sequences = []
    for index in range(timestep - 1, len(nonlinear_scaled)):
        sequences.append(nonlinear_scaled[index - timestep + 1 : index + 1])
    sequences = np.asarray(sequences, dtype=np.float32)

    with torch.no_grad():
        residual_scaled = model(torch.tensor(sequences)).cpu().numpy().reshape(-1)

    residual_pred = residual_scaled * residual_std + residual_ave
    linear_pred = (linear_ordered[timestep - 1 :] @ theta).reshape(-1)
    cloud_pred = linear_pred - residual_pred
    real_output = output_data[timestep - 1 :].reshape(-1)
    aligned_times = time_values.iloc[timestep - 1 :].reset_index(drop=True)

    result = pd.DataFrame(
        {
            "time": aligned_times,
            "real_output": real_output,
            "linear_prediction": linear_pred,
            "cloud_prediction": cloud_pred,
            "residual_prediction": residual_pred,
        }
    )
    result = result.sort_values("time").reset_index(drop=True)
    return result, {"timestep": timestep, "n_hidden": n_hidden, "n_layer": n_layer}


def compute_metrics(result):
    y_true = result["real_output"].to_numpy()
    y_pred = result["cloud_prediction"].to_numpy()
    return {
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def split_by_time_gap(result, max_gap_hours=1.0):
    if result.empty:
        return []

    time_deltas = result["time"].diff()
    split_indexes = [0]
    for index, delta in enumerate(time_deltas, start=0):
        if index == 0:
            continue
        if pd.isna(delta):
            continue
        if delta.total_seconds() > max_gap_hours * 3600:
            split_indexes.append(index)
    split_indexes.append(len(result))

    segments = []
    for start, end in zip(split_indexes[:-1], split_indexes[1:]):
        segment = result.iloc[start:end].copy()
        if not segment.empty:
            segments.append(segment)
    return segments


def plot_predictions(result, output_path):
    plt.figure(figsize=(15, 6))
    segments = split_by_time_gap(result)

    for index, segment in enumerate(segments):
        real_label = "Real Output" if index == 0 else None
        cloud_label = "Cloud Prediction" if index == 0 else None
        linear_label = "Linear Prediction" if index == 0 else None
        plt.plot(segment["time"], segment["real_output"], label=real_label, linewidth=2)
        plt.plot(segment["time"], segment["cloud_prediction"], label=cloud_label, linewidth=1.8)
        plt.plot(
            segment["time"],
            segment["linear_prediction"],
            label=linear_label,
            linewidth=1.6,
            alpha=0.8,
        )
    plt.title("Offline Replay: Real vs Predicted Output")
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def sanitize_date_label(value):
    return str(value).replace(":", "-").replace("/", "-")


def plot_predictions_by_date(result, output_dir):
    output_dir.mkdir(exist_ok=True)
    result = result.copy()
    result["date_only"] = result["time"].dt.date

    generated_paths = []
    for date_value, date_group in result.groupby("date_only"):
        if date_group.empty:
            continue
        output_path = output_dir / f"{sanitize_date_label(date_value)}.png"
        plot_predictions(date_group.drop(columns=["date_only"]), output_path)
        generated_paths.append(output_path)
    return generated_paths


def main():
    config = load_config()
    OUTPUT_DIR.mkdir(exist_ok=True)
    BY_DATE_DIR.mkdir(exist_ok=True)

    result, structure = build_offline_predictions(config)
    metrics = compute_metrics(result)

    csv_path = OUTPUT_DIR / f"{config['mark']}_offline_predictions.csv"
    png_path = OUTPUT_DIR / f"{config['mark']}_offline_trend.png"
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_predictions(result, png_path)
    by_date_paths = plot_predictions_by_date(result, BY_DATE_DIR)

    print(f"Offline prediction rows: {len(result)}")
    print(
        f"Using structure: timestep={structure['timestep']}, "
        f"n_hidden={structure['n_hidden']}, n_layer={structure['n_layer']}"
    )
    print("Offline metrics:")
    print(f"  rmse: {metrics['rmse']:.6f}")
    print(f"  mae: {metrics['mae']:.6f}")
    print(f"  mape: {metrics['mape']:.6f}")
    print(f"  r2: {metrics['r2']:.6f}")
    print(f"Prediction CSV saved to: {csv_path}")
    print(f"Trend plot saved to: {png_path}")
    print(f"Per-date plots saved to: {BY_DATE_DIR} ({len(by_date_paths)} files)")


if __name__ == "__main__":
    main()
