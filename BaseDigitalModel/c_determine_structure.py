import traceback

from Function import modelStrucSearch_run, getData_database_info, time

user_mark = getData_database_info["mark"]


if __name__ == "__main__":
    # 在线搜索网络结构
    while True:
        try:
            modelStrucSearch_run(mark=user_mark)
        except Exception as exc:
            print(f"[c_determine_structure] error for {user_mark}: {exc}", flush=True)
            traceback.print_exc()
            time.sleep(5)

