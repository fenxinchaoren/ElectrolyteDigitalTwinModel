import traceback

from Function import onlineForecast_run, getData_database_info, time

user_mark = getData_database_info["mark"]


if __name__ == "__main__":
    # 在线预报
    while True:
        try:
            onlineForecast_run(mark=user_mark)
        except Exception as exc:
            print(f"[e_edge_predict] error for {user_mark}: {exc}", flush=True)
            traceback.print_exc()
            time.sleep(1)

