import pandas as pd
import logging
import gc
import subprocess
import datetime
import glob
import re
from jeraconv import jeraconv
import category_encoders as ce

# 日付など
dt_now = datetime.datetime.now()
dt_now = dt_now.strftime("%Y%m%d_%H:%M")
gc.enable()
pd.options.display.max_columns = None

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


def preprocess(train_df, test_df):
    # 目的変数rename
    train_df = train_df.rename(columns={"取引価格（総額）_log": "y"})

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
        df["area"] = [float(x) if x != "2000㎡以上" else 2000 for x in df["面積（㎡）"]]

        # 建築年
        j2w = jeraconv.J2W()
        y = [x if x != "戦前" else "昭和10年" for x in df["建築年"].fillna("不明")]
        df["year_of_construction"] = [j2w.convert(x) if x != "不明" else None for x in y]

        # 建物の構造
        df["structure"] = df["建物の構造"]
        df["structure_block"] = [int("ブロック造" in x) for x in df["建物の構造"].fillna("")]
        df["structure_wood"] = [int("木造" in x) for x in df["建物の構造"].fillna("")]
        df["structure_lightiron"] = [int("軽量鉄骨造" in x) for x in df["建物の構造"].fillna("")]
        df["structure_iron"] = [int("鉄骨造" in x) for x in df["建物の構造"].fillna("")]
        df["structure_RC"] = [int("ＲＣ" in x) for x in df["建物の構造"].fillna("")]  # SRCも含まれるけどいいのか
        df["structure_SRC"] = [int("ＳＲＣ" in x) for x in df["建物の構造"].fillna("")]

        # 用途
        df["usage"] = df["用途"]
        df["usage_sonota"] = [int("その他" in x) for x in df["用途"].fillna("")]
        df["usage_office"] = [int("事務所" in x) for x in df["用途"].fillna("")]
        df["usage_warehouse"] = [int("倉庫" in x) for x in df["用途"].fillna("")]
        df["usage_shop"] = [int("店舗" in x) for x in df["用途"].fillna("")]
        df["usage_parking"] = [int("駐車場" in x) for x in df["用途"].fillna("")]
        df["usage_house"] = [int("住宅" in x) for x in df["用途"].fillna("")]
        df["usage_workshop"] = [int("作業場" in x) for x in df["用途"].fillna("")]
        df["usage_factory"] = [int("工場" in x) for x in df["用途"].fillna("")]

        # 今後の利用目的、都市計画、建ぺい率、改装
        df["future_usage"] = df["今後の利用目的"]
        df["city_plan"] = df["都市計画"]
        df["building_coverage_ratio"] = df["建ぺい率（％）"].astype(float)
        df["floor_area_ratio"] = df["容積率（％）"].astype(float)
        df["remodeling"] = df["改装"]

        # 取引時期など
        df["base_year"] = [float(x[0:4]) for x in df["取引時点"]]
        df["base_quarter"] = [float(x[6:7]) for x in df["取引時点"]]
        df["passed_year"] = df["base_year"] - df["year_of_construction"]

        # 取引の事情等
        df["reason"] = df["取引の事情等"]
        df["reason_other"] = [int("その他事情有り" in x) for x in df["取引の事情等"].fillna("")]
        df["reason_burden"] = [int("他の権利・負担付き" in x) for x in df["取引の事情等"].fillna("")]
        df["reason_auction"] = [int("調停・競売等" in x) for x in df["取引の事情等"].fillna("")]
        df["reason_defects"] = [int("瑕疵有りの可能性" in x) for x in df["取引の事情等"].fillna("")]
        df["reason_related_parties"] = [int("関係者間取引" in x) for x in df["取引の事情等"].fillna("")]
        logger.debug(f"head : {df.head()}")

    # 不要なカラム削除
    unuse_columns = [
        "都道府県名",
        "市区町村名",
        "地区名",
        "最寄駅：名称",
        "最寄駅：距離（分）",
        "間取り",
        "面積（㎡）",
        "建築年",
        "建物の構造",
        "用途",
        "今後の利用目的",
        "都市計画",
        "建ぺい率（％）",
        "容積率（％）",
        "改装",
        "取引時点",
        "取引の事情等",
    ]
    train_df = train_df.drop(unuse_columns, axis=1)
    test_df = test_df.drop(unuse_columns, axis=1)

    # label encoding
    category_columns = [
        "pref",
        "pref_city",
        "pref_city_district",
        "station",
        "plan",
        "structure",
        "usage",
        "future_usage",
        "city_plan",
        "remodeling",
    ]
    ce_oe = ce.OrdinalEncoder()
    train_df.loc[:, category_columns] = ce_oe.fit_transform(train_df[category_columns])
    test_df.loc[:, category_columns] = ce_oe.transform(test_df[category_columns])
    logger.debug(f"train head : {train_df.head()}")
    logger.debug(f"test head : {test_df.head()}")
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = get_data()
    train_df, test_df = preprocess(train_df, test_df)
