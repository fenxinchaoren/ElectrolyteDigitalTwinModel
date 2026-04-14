import csv
import struct
from pathlib import Path
from statistics import median


BASE_DIR = Path(__file__).resolve().parent
DATA_CHANGE_DIR = BASE_DIR / "DataChange"
TIME_COLUMN = "_time"
TARGET_COLUMN = "SetFrequencylPumpSet1"
FLOW_PROCESSED_COLUMN = "ActFlowPumpSET1_process"
ACTUAL_COLUMN = "Actual_Pump_SET1"
FLOW_COLUMN = "ActFlowPumpSET1"
FREQUENCY_MIN = 0.0
FREQUENCY_MAX = 120.0
FLOW_MIN = 0.0
FLOW_MAX = 42.0 * 21.0


def safe_float(value):
    if value is None:
        return 0.0

    text = str(value).strip()
    if not text:
        return 0.0

    try:
        return float(text)
    except ValueError:
        return 0.0


def clamp(value, low, high):
    return max(low, min(high, value))


def percentile(values, ratio):
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * ratio
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + (upper_value - lower_value) * weight


def format_frequency(value):
    if value <= 0:
        return "0"
    return f"{value:.1f}"


def format_flow(value):
    if value <= 0:
        return "0"
    return f"{value:.2f}"


def decode_ieee754_float(value):
    raw_value = int(round(safe_float(value)))
    if raw_value <= 0:
        return 0.0

    try:
        decoded_value = struct.unpack(">f", struct.pack(">I", raw_value & 0xFFFFFFFF))[0]
    except (OverflowError, struct.error):
        return 0.0

    if decoded_value != decoded_value:
        return 0.0

    if decoded_value == float("inf") or decoded_value == float("-inf"):
        return 0.0

    return clamp(decoded_value, FLOW_MIN, FLOW_MAX)


def build_flow_frequency(flow_values, actual_values):
    positive_flow = [value for value in flow_values if value > 0]
    positive_actual = [value for value in actual_values if value > 0]

    if not positive_flow:
        return {
            "active_threshold": 0.0,
            "flow_low": 0.0,
            "flow_high": 0.0,
            "active_median": 67.0,
            "active_upper": 67.0,
        }

    active_median = median(positive_actual) if positive_actual else 67.0
    active_upper = max(positive_actual) if positive_actual else active_median
    flow_low = percentile(positive_flow, 0.10)
    flow_high = percentile(positive_flow, 0.90)

    if flow_high <= flow_low:
        flow_high = max(flow_low * 1.05, flow_low + 1.0)

    active_threshold = max(percentile(positive_flow, 0.05) * 0.25, 1.0)

    return {
        "active_threshold": active_threshold,
        "flow_low": flow_low,
        "flow_high": flow_high,
        "active_median": active_median,
        "active_upper": active_upper,
    }


def flow_to_frequency(flow_value, model):
    if flow_value <= model["active_threshold"]:
        return 0.0

    flow_low = model["flow_low"]
    flow_high = model["flow_high"]

    if flow_high <= flow_low:
        return clamp(model["active_median"], FREQUENCY_MIN, FREQUENCY_MAX)

    normalized = clamp((flow_value - flow_low) / (flow_high - flow_low), 0.0, 1.0)
    lower_frequency = max(FREQUENCY_MIN, model["active_median"] - 1.2)
    upper_frequency = min(FREQUENCY_MAX, max(model["active_upper"], model["active_median"] + 0.8))
    return lower_frequency + (upper_frequency - lower_frequency) * normalized


def build_neighbor_frequency(actual_values):
    total = len(actual_values)
    previous_values = [None] * total
    next_values = [None] * total

    last_value = None
    for index, value in enumerate(actual_values):
        if value > 0:
            last_value = value
        previous_values[index] = last_value

    next_value = None
    for index in range(total - 1, -1, -1):
        value = actual_values[index]
        if value > 0:
            next_value = value
        next_values[index] = next_value

    neighbors = []
    for previous_value, following_value in zip(previous_values, next_values):
        if previous_value is not None and following_value is not None:
            neighbors.append((previous_value + following_value) / 2.0)
        elif previous_value is not None:
            neighbors.append(previous_value)
        elif following_value is not None:
            neighbors.append(following_value)
        else:
            neighbors.append(0.0)

    return neighbors


def smooth_active_segments(raw_values, active_mask):
    smoothed = [0.0] * len(raw_values)
    index = 0

    while index < len(raw_values):
        if not active_mask[index]:
            index += 1
            continue

        end = index
        while end < len(raw_values) and active_mask[end]:
            end += 1

        segment = raw_values[index:end]
        previous = segment[0]
        smoothed[index] = previous

        for offset in range(1, len(segment)):
            current = segment[offset]
            previous = 0.35 * previous + 0.65 * current
            smoothed[index + offset] = previous

        index = end

    return smoothed


def generate_set_frequency(actual_values, flow_values):
    model = build_flow_frequency(flow_values, actual_values)
    neighbor_values = build_neighbor_frequency(actual_values)

    raw_values = []
    active_mask = []

    for actual_value, flow_value, neighbor_value in zip(actual_values, flow_values, neighbor_values):
        flow_frequency = flow_to_frequency(flow_value, model)
        is_active = actual_value > 0 or flow_value > model["active_threshold"]
        active_mask.append(is_active)

        if not is_active:
            raw_values.append(0.0)
            continue

        if actual_value > 0:
            reference_frequency = flow_frequency if flow_frequency > 0 else actual_value
            set_frequency = 0.97 * actual_value + 0.03 * reference_frequency + 0.25
        else:
            reference_frequency = neighbor_value if neighbor_value > 0 else flow_frequency
            blend_frequency = flow_frequency if flow_frequency > 0 else reference_frequency
            set_frequency = 0.80 * reference_frequency + 0.20 * blend_frequency + 0.45

        raw_values.append(clamp(set_frequency, FREQUENCY_MIN, FREQUENCY_MAX))

    smoothed_values = smooth_active_segments(raw_values, active_mask)
    return [format_frequency(clamp(value, FREQUENCY_MIN, FREQUENCY_MAX)) for value in smoothed_values]


def process_csv_file(file_path):
    with file_path.open("r", newline="", encoding="utf-8-sig") as source_file:
        reader = csv.reader(source_file)
        rows = list(reader)

    if not rows:
        print(f"Skip empty file: {file_path.name}")
        return

    header = rows[0]
    data_rows = rows[1:]

    if ACTUAL_COLUMN not in header or FLOW_COLUMN not in header:
        print(f"Skip {file_path.name}: missing {ACTUAL_COLUMN} or {FLOW_COLUMN}")
        return

    actual_index = header.index(ACTUAL_COLUMN)
    flow_index = header.index(FLOW_COLUMN)

    actual_values = []
    flow_values = []
    processed_flow_values = []

    for row in data_rows:
        if len(row) < len(header):
            row.extend([""] * (len(header) - len(row)))

        actual_values.append(safe_float(row[actual_index]))
        processed_flow_value = decode_ieee754_float(row[flow_index])
        flow_values.append(processed_flow_value)
        processed_flow_values.append(format_flow(processed_flow_value))

    generated_values = generate_set_frequency(actual_values, flow_values)

    removable_columns = [TARGET_COLUMN, FLOW_PROCESSED_COLUMN]
    removable_indexes = sorted(
        [header.index(column_name) for column_name in removable_columns if column_name in header],
        reverse=True,
    )

    for removable_index in removable_indexes:
        header.pop(removable_index)
        for row in data_rows:
            if len(row) > removable_index:
                row.pop(removable_index)

    insert_index = header.index(TIME_COLUMN) + 1 if TIME_COLUMN in header else 0
    new_header = (
        header[:insert_index]
        + [TARGET_COLUMN, FLOW_PROCESSED_COLUMN]
        + header[insert_index:]
    )
    new_rows = []

    for row, generated_value, processed_flow_value in zip(
        data_rows, generated_values, processed_flow_values
    ):
        new_rows.append(
            row[:insert_index]
            + [generated_value, processed_flow_value]
            + row[insert_index:]
        )

    with file_path.open("w", newline="", encoding="utf-8-sig") as target_file:
        writer = csv.writer(target_file)
        writer.writerow(new_header)
        writer.writerows(new_rows)

    print(f"Updated: {file_path.name}")


def process_datachange_files():
    if not DATA_CHANGE_DIR.exists():
        print(f"Directory not found: {DATA_CHANGE_DIR}")
        return

    csv_files = sorted(DATA_CHANGE_DIR.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in DataChange.")
        return

    print(f"Start processing {len(csv_files)} CSV files...")
    for file_path in csv_files:
        try:
            process_csv_file(file_path)
        except Exception as exc:
            print(f"Error processing {file_path.name}: {exc}")

    print("All files processed.")


if __name__ == "__main__":
    process_datachange_files()
