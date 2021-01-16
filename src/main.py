import pandas as pd
import logging
import gc
import subprocess
import datetime
import glob
import re

dt_now = datetime.datetime.now()
dt_now = dt_now.strftime("%Y%m%d_%H:%M")
gc.enable()

# branch名の取得
_cmd = "git rev-parse --abbrev-ref HEAD"
branch = subprocess.check_output(_cmd.split()).strip().decode("utf-8")
branch = "-".join(branch.split("/"))

# 流石にロガーはglobalを使うぞ
formatter = "%(levelname)s : %(asctime)s : %(message)s"
logging.basicConfig(level=logging.DEBUG, format=formatter, filename=f"log/logger_{dt_now}_{branch}.log")
logger = logging.getLogger(__name__)


def get_data():
    unuse_columns = ["ID", "種類", "地域", "市区町村コード", "土地の形状", "間口", "延床面積（㎡）", "前面道路：方位", "前面道路：種類", "前面道路：幅員（ｍ）"]
    train_files = sorted(glob.glob("./data/raw/train/*"))
    train_df = []
    for file in train_files:
        train_df.append(pd.read_csv(file, low_memory=False))
    train_df = pd.concat(train_df, axis=0).drop(unuse_columns, axis=1).reset_index(drop=True)
    test_df = pd.read_csv("./data/raw/test.csv").drop(unuse_columns, axis=1).reset_index(drop=True)
    return train_df, test_df


def preprocess(df_train, df_test):
    return


if __name__ == "__main__":
    train_df, test_df = get_data()

    def re_searcher(reg_exp: str, x: str) -> float:
        m = re.search(reg_exp, x)
        if m is not None:
            return float(m.groups()[0])
        else:
            return None

    for df in [train_df, test_df]:
        # 都道府県+市区町村、都道府県+市区町村+地区名を取得
        df["pref"] = df["都道府県名"]
        df["pref_city"] = df["都道府県名"] + df["市区町村名"]
        df["pref_city_district"] = df["都道府県名"] + df["市区町村名"] + df["地区名"]

        # 駅関係を取得
        df["station"] = df["最寄駅：名称"]
        time_to_station_dict = {"1H30?2H": "105", "1H?1H30": "75", "2H?": "120", "30分?60分": "45"}
        df["time_to_station"] = df["最寄駅：距離（分）"].map(time_to_station_dict).astype(float)

        # 物件情報、間取り
        df["plan"] = df["間取り"]
        df["plan_num"] = [re_searcher("(\d)+", x) for x in df["間取り"].fillna("")]
        df["plan_LDK"] = [int("ＬＤＫ" in x) for x in df["間取り"].fillna("")]
        df["plan_LD"] = [int("ＬＤ" in x) for x in df["間取り"].fillna("")]
        df["plan_DK"] = [int("ＤＫ" in x) for x in df["間取り"].fillna("")]
        df["plan_L"] = [int("Ｌ" in x) for x in df["間取り"].fillna("")]
        df["plan_D"] = [int("Ｄ" in x) for x in df["間取り"].fillna("")]
        df["plan_K"] = [int("Ｋ" in x) for x in df["間取り"].fillna("")]
        df["plan_R"] = [int("Ｒ" in x) for x in df["間取り"].fillna("")]
        df["plan_S"] = [int("S" in x) for x in df["間取り"].fillna("")]

        # 物件情報、面積
        df[df["面積（㎡）"] == "2000㎡以上"]["面積（㎡）"] = "2000"
        df["area"] = df["面積（㎡）"].astype(float)

        unuse_columns = ["都道府県名", "市区町村名", "地区名", "最寄駅：名称", "最寄駅：距離（分）", "間取り", "面積（㎡）"]
        df = df.drop(unuse_columns, axis=1)

        logger.debug(f"head : {df.head()}")
