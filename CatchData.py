import os
import re

import pandas as pd


SOURCE_DIR = r"d:\ZhiQingProgram\ElectrolyteDigitalTwinModel\Data"
OUTPUT_DIR = r"d:\ZhiQingProgram\ElectrolyteDigitalTwinModel\DataChange"


def is_zero_or_empty(value):
    if pd.isna(value):
        return True

    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return True
        try:
            return float(text) == 0
        except ValueError:
            return False

    try:
        return float(value) == 0
    except (TypeError, ValueError):
        return False


def remove_zero_or_empty_rows(df):
    if df.empty:
        return df

    if df.shape[1] <= 1:
        return df.iloc[0:0].copy()

    data_columns = df.iloc[:, 1:]
    zero_or_empty_mask = data_columns.apply(lambda column: column.map(is_zero_or_empty))
    rows_to_keep = ~zero_or_empty_mask.all(axis=1)
    return df.loc[rows_to_keep].copy()


def build_columns_to_keep():
    columns_to_keep = [
        "_time",
        "SetFlowPumpSET1",
        "SetFrequencyManualPumpSet1",
        "SET1_Electrolyte_Pressure_Before",
        "SET1_Electrolyte_Pressure_After",
        "SET1_Electrolyte_Temperature",
        "WaterTank_Temperature",
        "WaterTank_LiquidLevel",
        "PumpSet1On",
        "Actual_Pump_SET1",
        "ActFlowPumpSET1",
        "AlkaliValve_E_BC_001_Open",
        "AlkaliValve_E_BC_001_Closed",
        "DrainagePipe_OnOffValve",
        "WaterSupplyPipe_OnOffValve",
    ]

    for i in range(1, 22):
        columns_to_keep.append(f"String{i}_Power_F{i:02d}Closed")

    for i in range(1, 22):
        columns_to_keep.append(f"string{i:02d}.ActualCurrent")

    for i in range(1, 22):
        columns_to_keep.append(f"string{i:02d}.ElectrolyteFlowAverage")

    return columns_to_keep


def extract_time_part(filename):
    name_without_ext = os.path.splitext(filename)[0]
    match = re.search(r"(\d{4}[-_]?\d{2}[-_]?\d{2}[_-]?\d{0,6})", name_without_ext)
    if not match:
        return name_without_ext
    return match.group(1).replace("_", "-").replace("--", "-")


def process_data():
    columns_to_keep = build_columns_to_keep()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    csv_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("No CSV files found in source directory.")
        return

    print(f"Found {len(csv_files)} CSV files. Start processing...")

    for filename in csv_files:
        source_file = os.path.join(SOURCE_DIR, filename)
        time_part = extract_time_part(filename)
        output_filename = f"zhiqing_control_{time_part}.csv"
        output_file = os.path.join(OUTPUT_DIR, output_filename)

        try:
            print(f"Processing: {filename}")
            df = pd.read_csv(source_file)

            existing_columns = [col for col in columns_to_keep if col in df.columns]
            missing_columns = [col for col in columns_to_keep if col not in df.columns]

            if missing_columns:
                print(
                    f"Warning [{filename}]: missing columns ignored: {missing_columns[:5]}"
                )

            if not existing_columns:
                print(f"Warning [{filename}]: no target columns found, skipped.")
                continue

            df_filtered = df[existing_columns].copy()
            original_rows = len(df_filtered)
            df_filtered = remove_zero_or_empty_rows(df_filtered)
            removed_rows = original_rows - len(df_filtered)

            df_filtered.to_csv(output_file, index=False)
            print(
                f"Saved: {output_filename} | kept {len(df_filtered)} rows, removed {removed_rows} rows"
            )

        except Exception as exc:
            print(f"Error processing {filename}: {exc}")

    print("All files processed.")

if __name__ == "__main__":
    process_data()
# 恢复