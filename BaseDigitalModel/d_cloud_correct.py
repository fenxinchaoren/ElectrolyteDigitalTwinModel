import traceback

from Function import modelParaTrain_run, getData_database_info, time

user_mark = getData_database_info["mark"]


if __name__ == "__main__":
    # 在线训练网络权重和偏置
    while True:
        try:
            modelParaTrain_run(mark=user_mark)
        except Exception as exc:
            print(f"[d_cloud_correct] error for {user_mark}: {exc}", flush=True)
            traceback.print_exc()
            time.sleep(5)
