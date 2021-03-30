import copy
import datetime
import gc
import glob
import hashlib
import inspect
import logging
import math
import os
import pickle
import random
import re
import sys
from inspect import signature
from pathlib import Path
from typing import Any, ByteString, Callable, Dict, List, Optional, Tuple, Union

import category_encoders as ce
import feather
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from xfeat import Pipeline, SelectNumerical

gc.enable()
pd.options.display.max_columns = None


class Cache:
    def __init__(self, dir_path: str, func_name: str = None):
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(exist_ok=True)
        self.func_name = func_name

    def __call__(self, func: Callable):
        def wrapper(*args, **kwargs):
            if self.func_name is None:
                self.func_name = func.__name__

            # 関数そのものの文字列
            func_source = inspect.getsource(func)
            # 引数取得(https://blog.amedama.jp/entry/2016/10/31/225219)
            func_args_list = []
            sig = signature(func)
            # 受け取ったパラメータをシグネチャにバインドする
            bound_args = sig.bind(*args, **kwargs)
            # 関数名やバインドしたパラメータの対応関係を取得する
            for k, v in bound_args.arguments.items():
                if k == "trainer_instance":
                    # trainerがあればメンバー変数を取得
                    member_vars = ",".join("{_k}={_v}".format(_k=_k, _v=_v) for _k, _v in vars(v).items())
                    func_args_list.append(member_vars)
                    # methodがあればその定義を取得しておく
                    for x in inspect.getmembers(v, inspect.ismethod):
                        func_args_list.append(inspect.getsource(x[1]))
                else:
                    func_args_list.append(f"{k}={v}")
            func_args_list = sorted(func_args_list)
            func_args = "_".join(func_args_list)
            func_info = func_source.encode("utf-8") + func_args.encode("utf-8")
            func_hash = hashlib.md5(func_info).hexdigest()
            # 関数ごとにキャッシュdirを作る
            cache_dir = self.dir_path / self.func_name
            cache_dir.mkdir(exist_ok=True)
            cache_path = cache_dir / (func_hash + ".pickle")

            if os.path.exists(cache_path):
                print(f"cache hit {self.func_name}: {cache_path}")
                ret = self.load(cache_path)
            else:
                print(f"cache does not hit {self.func_name}: {cache_path}")
                ret = func(*args, **kwargs)
                self.save(ret, cache_path)
            return ret

        return wrapper

    @staticmethod
    def save(obj: object, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        return 0

    @staticmethod
    def load(file_path: str):
        with open(file_path, "rb") as f:
            ret = pickle.load(f)
        return ret


def tprint(*args, **kwargs):
    print(datetime.datetime.now(), *args)


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@Cache("./cache")
def reduce_mem_usage(df, logger=None, level=logging.DEBUG):
    print_ = tprint if logger is None else lambda msg: logger.log(level, msg)
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print_("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        try:
            col_type = df[col].dtype
        except:
            print(col)
            print(df[col].head())
        if col_type != "object" and col_type != "datetime64[ns]":
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print_("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print_("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    return df


@Cache("./cache")
def get_data():
    unuse_columns = ["種類", "地域", "土地の形状", "間口", "延床面積（㎡）", "前面道路：方位", "前面道路：種類", "前面道路：幅員（ｍ）"]
    train_files = sorted(glob.glob("./data/raw/train/*"))
    train_df = []
    for file in train_files:
        train_df.append(pd.read_csv(file, low_memory=False))
    train_df = pd.concat(train_df, axis=0).drop(unuse_columns, axis=1).reset_index(drop=True)
    test_df = pd.read_csv("./data/raw/test.csv").drop(unuse_columns, axis=1).reset_index(drop=True)
    sample_submission = pd.read_csv("./data/raw/sample_submission.csv")
    return train_df, test_df, sample_submission


@Cache("./cache")
def preprocess(train_df, test_df):
    # 結合
    train_df["is_train"] = 1
    test_df["is_train"] = 0
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    # rename
    df = df.rename(
        columns={
            "取引価格（総額）_log": "y",
        }
    )

    # 欠損数
    df["null_num"] = df.isnull().sum(axis=1)

    # 都道府県+市区町村、都道府県+市区町村+地区名を取得
    df["pref"] = df["都道府県名"]
    df["pref_city"] = df["都道府県名"] + df["市区町村名"]
    df["pref_city_district"] = df["都道府県名"] + df["市区町村名"] + df["地区名"]
    del df["都道府県名"], df["市区町村名"], df["地区名"]

    # 緯度経度
    lat_lon = pd.read_csv("./data/external/lat_lon.csv")
    lat_lon.columns = ["pref_city_district", "lon", "lat"]
    df_len = len(df.copy())
    df = df.merge(lat_lon, on="pref_city_district", how="left")
    assert len(df) == df_len, f"{len(df)}, {df_len}"
    df["lat_x_lon"] = df["lat"] * df["lon"]
    df["lat_to_lon"] = (df["lat"] ** 2) + (df["lon"] ** 2)
    df["lat_to_lon"] = [x ** 0.5 for x in df["lat_to_lon"]]

    # 駅関係を取得
    df["station"] = df["最寄駅：名称"].replace(r"\(.+\)", "", regex=True)
    df["station"] = df["station"].replace(r"\（.+\）", "", regex=True)
    df["最寄駅：距離（分）"] = [x if x != "1H30?2H" else "05" for x in df["最寄駅：距離（分）"]]
    df["最寄駅：距離（分）"] = [x if x != "1H?1H30" else "75" for x in df["最寄駅：距離（分）"]]
    df["最寄駅：距離（分）"] = [x if x != "2H?" else "120" for x in df["最寄駅：距離（分）"]]
    df["最寄駅：距離（分）"] = [x if x != "30分?60分" else "45" for x in df["最寄駅：距離（分）"]]
    df["time_to_station"] = df["最寄駅：距離（分）"].astype(float)
    del df["最寄駅：距離（分）"], df["最寄駅：名称"]
    station = pd.read_csv("./data/external/station20151215free.txt", sep="\t").rename(
        columns={"station_name": "station", "lon": "station_lon", "lat": "station_lat"}
    )
    station = station[~station[["station", "pref_cd"]].duplicated()]
    station = station[~station[["station", "post"]].duplicated()]
    pref_code = {
        1: "北海道",
        2: "青森県",
        3: "岩手県",
        4: "宮城県",
        5: "秋田県",
        6: "山形県",
        7: "福島県",
        8: "茨城県",
        9: "栃木県",
        10: "群馬県",
        11: "埼玉県",
        12: "千葉県",
        13: "東京都",
        14: "神奈川県",
        15: "新潟県",
        16: "富山県",
        17: "石川県",
        18: "福井県",
        19: "山梨県",
        20: "長野県",
        21: "岐阜県",
        22: "静岡県",
        23: "愛知県",
        24: "三重県",
        25: "滋賀県",
        26: "京都府",
        27: "大阪府",
        28: "兵庫県",
        29: "奈良県",
        30: "和歌山県",
        31: "鳥取県",
        32: "島根県",
        33: "岡山県",
        34: "広島県",
        35: "山口県",
        36: "徳島県",
        37: "香川県",
        38: "愛媛県",
        39: "高知県",
        40: "福岡県",
        41: "佐賀県",
        42: "長崎県",
        43: "熊本県",
        44: "大分県",
        45: "宮崎県",
        46: "鹿児島県",
        47: "沖縄県",
    }
    station["pref"] = station["pref_cd"].map(pref_code)
    station = station[["pref", "station", "station_lat", "station_lon", "line_cd"]]
    station_extra_df = pd.read_csv("./data/external/station_extra.csv")
    station = pd.concat([station, station_extra_df]).reset_index(drop=True)
    station = station[~station[["station", "pref"]].duplicated()]
    df = df.merge(station, on=["pref", "station"], how="left")
    assert len(df) == df_len, f"{len(df)}, {df_len}"

    df["station_lat_x_lon"] = df["station_lat"] * df["station_lon"]
    df["station_lat_to_lon"] = (df["station_lat"] ** 2) + (df["station_lon"] ** 2)
    df["station_lat_to_lon"] = [x ** 0.5 for x in df["station_lat_to_lon"]]
    df["diff_lon"] = df["lon"] - df["station_lon"]
    df["diff_lat"] = df["lat"] - df["station_lat"]
    df["diff_station"] = df["station_lat_to_lon"] - df["lat_to_lon"]

    def get_distance_m(lat1, lon1, lat2, lon2):
        """
        https://qiita.com/fetaro/items/b7c5abee42db54c0f26a
        ２点間の距離(m)
        球面三角法を利用した簡易的な距離計算
        GoogleMapAPIのgeometory.computeDistanceBetweenのロジック
        https://www.suzu6.net/posts/167-php-spherical-trigonometry/
        """
        R = 6378137.0  # 赤道半径
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
        diff_lon = lon1 - lon2
        dist = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(diff_lon)
        return R * math.acos(min(max(dist, -1.0), 1.0))

    df["distance_to_station"] = [
        get_distance_m(lat1, lon1, lat2, lon2)
        for lat1, lon1, lat2, lon2 in zip(df["lat"], df["lon"], df["station_lat"], df["station_lon"])
    ]

    # 国勢調査
    unuse = ["census_都道府県名", "census_都道府県市区町村名"]
    census = pd.read_csv("./data/external/census.csv").drop(unuse, axis=1).reset_index(drop=True)
    df = df.merge(census, on="市区町村コード", how="left")
    assert len(df) == df_len, f"{len(df)}, {df_len}"

    # 公示価格
    price_df = pd.read_csv("./data/external/price.csv")
    df = df.merge(price_df, on="市区町村コード", how="left")
    assert len(df) == df_len, f"{len(df)}, {df_len}"

    # 経済センサス
    econcensus_df = pd.read_csv("./data/external/econ_census.csv")
    df = df.merge(econcensus_df, on="市区町村コード", how="left")
    assert len(df) == df_len, f"{len(df)}, {df_len}"

    def re_searcher(reg_exp: str, x: str) -> float:
        m = re.search(reg_exp, x)
        if m is not None:
            return float(m.groups()[0])
        else:
            return None

    # 物件情報、間取り
    df["plan"] = df["間取り"]
    df["plan_num"] = [re_searcher("(\d)+", x) for x in df["間取り"].fillna("")]
    del df["間取り"]

    # 物件情報、面積
    df["面積（㎡）"] = [x if x != "2000㎡以上" else "2000" for x in df["面積（㎡）"]]
    df["面積（㎡）"] = [x if x != "m^2未満" else "2" for x in df["面積（㎡）"]]
    df["area"] = df["面積（㎡）"].astype(float)
    del df["面積（㎡）"]

    # 建築年
    def convert_years(x: str):
        head_str = x[:2]
        value_str = x[2:-1]
        if head_str == "不明":
            return np.nan
        elif head_str == "戦前":
            return 1945
        elif head_str == "昭和":
            return 1925 + int(value_str)
        elif head_str == "平成":
            return 1988 + int(value_str)
        elif head_str == "令和":
            return 2018 + int(value_str)

    df["year_of_construction"] = [float(convert_years(x)) for x in df["建築年"].fillna("不明")]
    del df["建築年"]

    # その他
    df["structure"] = df["建物の構造"]
    del df["建物の構造"]
    df["usage"] = df["用途"]
    del df["用途"]
    df["reason"] = df["取引の事情等"]
    del df["取引の事情等"]
    df["future_usage"] = df["今後の利用目的"]
    df["city_plan"] = df["都市計画"]
    df["building_coverage_ratio"] = df["建ぺい率（％）"].astype(float)
    df["floor_area_ratio"] = df["容積率（％）"].astype(float)
    df["remodeling"] = df["改装"]
    del df["今後の利用目的"], df["都市計画"], df["建ぺい率（％）"], df["容積率（％）"], df["改装"]
    # 取引時期など
    df["base_year"] = [int(x[0:4]) for x in df["取引時点"]]
    df["base_quarter"] = [int(x[6:7]) for x in df["取引時点"]]
    df["timing_code"] = [y + (4 * (x - 2005)) for x, y in zip(df["base_year"], df["base_quarter"])]
    df["passed_year"] = df["base_year"] - df["year_of_construction"]
    del df["取引時点"]

    # ここからGBDT系専用の処理
    # NN系の処理をすることを見越してcopyしておく
    df_nn = df.copy()

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
        "reason",
    ]
    ce_oe = ce.OrdinalEncoder()
    df.loc[:, category_columns] = ce_oe.fit_transform(df[category_columns])
    df[category_columns] = df[category_columns].astype(int)

    # ここからNN系の処理
    # スケールを揃えておく
    numeric_cols = [x for x in SelectNumerical().fit_transform(df_nn).columns if x not in ["y", "is_train"]]
    transformer = MinMaxScaler()
    df_nn[numeric_cols] = transformer.fit_transform(df_nn[numeric_cols])
    # one-hot特徴作成
    def multi_hot_encoder(df: pd.Series, sep: str) -> pd.DataFrame:
        res = pd.DataFrame()
        colname = df.name
        res[colname] = pd.Series([x.split(sep) for x in df.fillna("不明")])
        res = (
            res.explode(colname)
            .reset_index()
            .pivot_table(index=["index"], columns=[colname], aggfunc=[len], fill_value=0)
        )
        res.columns = [colname + "_" + x[1] for x in res.columns]
        return res.reset_index(drop=True)

    onehot_columns = [
        "plan",
        "structure",
        "usage",
        "future_usage",
        "city_plan",
        "remodeling",
        "reason",
    ]
    onehot_plan = multi_hot_encoder(df_nn["plan"], sep="＋")
    onehot_structure = multi_hot_encoder(df_nn["structure"], sep="、")
    onehot_usage = multi_hot_encoder(df_nn["usage"], sep="、")
    onehot_future_usage = multi_hot_encoder(df_nn["future_usage"], sep="、")
    onehot_city_plan = multi_hot_encoder(df_nn["city_plan"], sep="、")
    onehot_remodeling = multi_hot_encoder(df_nn["remodeling"], sep="、")
    onehot_reason = multi_hot_encoder(df_nn["reason"], sep="、")
    df_nn = (
        pd.concat(
            [
                df_nn,
                onehot_plan,
                onehot_structure,
                onehot_usage,
                onehot_future_usage,
                onehot_city_plan,
                onehot_remodeling,
                onehot_reason,
            ],
            axis=1,
        )
        .drop(onehot_columns, axis=1)
        .reset_index(drop=True)
    )

    # label encoding
    category_columns = ["pref", "pref_city", "pref_city_district", "station"]
    ce_oe = ce.OrdinalEncoder()
    df_nn.loc[:, category_columns] = ce_oe.fit_transform(df_nn[category_columns])
    df_nn[category_columns] = df_nn[category_columns].astype(int)
    # fillna
    df_nn = df_nn.fillna(0)

    # 分割
    train_df = df[(df["is_train"] == 1)].reset_index(drop=True)
    test_df = df[df["is_train"] == 0].reset_index(drop=True)
    del test_df["y"], train_df["is_train"], test_df["is_train"]
    train_df_nn = df_nn[(df_nn["is_train"] == 1)].reset_index(drop=True)
    test_df_nn = df_nn[df_nn["is_train"] == 0].reset_index(drop=True)
    del test_df_nn["y"], train_df_nn["is_train"], test_df_nn["is_train"]
    assert train_df.shape[0] == train_df_nn.shape[0], f"{train_df.shape}, {train_df_nn.shape}"
    assert test_df.shape[0] == test_df_nn.shape[0], f"{test_df.shape}, {test_df_nn.shape}"

    # 不要なカラム削除
    def get_unuse_cols(df: pd.DataFrame, th: float) -> List[str]:
        unuse_cols = []
        numeric_cols = [x for x in SelectNumerical().fit_transform(df).columns if x not in ["y"]]
        for col in numeric_cols:
            std_values = df[col].std()
            if std_values <= th:
                unuse_cols.append(col)
                tprint(f"{col} has std={std_values}")
        return unuse_cols

    # 相関高い変数をリストアップ
    high_corr_cols = ["census_都道府県庁所在市"]

    unuse_cols = get_unuse_cols(train_df, 0.01)
    train_df = train_df.drop(unuse_cols + high_corr_cols, axis=1).reset_index(drop=True)
    test_df = test_df.drop(unuse_cols + high_corr_cols, axis=1).reset_index(drop=True)

    unuse_cols = get_unuse_cols(train_df_nn, 0.01)
    train_df_nn = train_df_nn.drop(unuse_cols + high_corr_cols, axis=1).reset_index(drop=True)
    test_df_nn = test_df_nn.drop(unuse_cols + high_corr_cols, axis=1).reset_index(drop=True)

    return train_df, test_df, train_df_nn, test_df_nn


class GroupKfoldTrainer(object):
    def __init__(self, state_path, predictors, target_col, X, groups, test, n_splits, n_rsb):
        self.state_path = state_path
        self.predictors = predictors
        self.target_col = target_col
        self.X = X
        self.groups = groups
        self.test = test
        self.n_splits = n_splits
        self.n_rsb = n_rsb
        self.oof = np.zeros(len(X))
        self.pred = []
        self.validation_score = []
        self.folds = []
        self.state_path = Path(state_path)
        self.file_path = self.state_path.joinpath(f"{self.name}.pickle")

    def loss_(sel, predictions, targets):
        return mean_absolute_error(targets, predictions)

    def _fit(self, X_train, Y_train, X_valid, Y_valid, loop_seed):
        raise NotImplementedError

    def _predict(self, model, X):
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def save(self):
        with open(f"{self.file_path}", "wb") as f:
            pickle.dump(self, f)
            return 0

    @classmethod
    def load(cls, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def fit(self):
        gf = GroupKFold(n_splits=self.n_splits)
        for fold_cnt, (train_idx, valid_idx) in enumerate(gf.split(self.X, None, self.groups)):
            fold_cnt += 1
            self.fold_cnt = fold_cnt
            tprint(f"START FOLD {fold_cnt}")

            # DataFrame -> train, valid
            X_train, X_valid = self.X.loc[train_idx, self.predictors], self.X.loc[valid_idx, self.predictors]
            Y_train, Y_valid = self.X.loc[train_idx, self.target_col], self.X.loc[valid_idx, self.target_col]
            tprint(f"training years : {set(self.X.loc[train_idx, 'base_year'])}")
            tprint(f"validation years : {set(self.X.loc[valid_idx, 'base_year'])}")
            self.valid_idx = valid_idx

            # random seed blending
            for rsb_idx in range(self.n_rsb):
                tprint(f"     fitting {rsb_idx + 1} th loop of {self.n_rsb}")
                # 学習
                ret = self._fit(X_train, Y_train, X_valid, Y_valid, loop_seed=rsb_idx)
                # save models
                """
                if (fold_cnt == 1) & (rsb_idx == 0):
                    self.folds.append(ret)
                """
                model = ret["model"]

                # oof, predに対して予測
                pred_oof = self._predict(model, X_valid)
                pred_test = self._predict(model, self.test)

                # 格納
                self.oof[valid_idx] += pred_oof / self.n_rsb
                self.pred.append(pred_test)

                # single fold validation　score
                _validation_score = self.loss_(pred_oof, Y_valid.values)
                tprint((f"     finished {rsb_idx + 1}" + f"th loop WITH {_validation_score:.6f}"))

            # rsb validation　score
            _validation_score = self.loss_(self.oof[valid_idx], Y_valid.values)
            self.validation_score.append(_validation_score)
            tprint(f"     END FOLD {fold_cnt} WITH {_validation_score:.6f}")
            tprint("----切り取り----\n")

        # validation score of fold mean
        cv_score = np.mean(self.validation_score, axis=0)
        cv_std = np.std(self.validation_score, axis=0)
        tprint(f"TOTAL CV  SCORE is : {cv_score:.6f} +- {cv_std:.4f}")
        oof_score = self.loss_(self.oof, self.X.loc[:, self.target_col])
        tprint(f"TOTAL OOF SCORE is : {oof_score:.6f}")
        self.pred = np.mean(self.pred, axis=0)
        tprint("----終わり----\n")


class LGBTrainer(GroupKfoldTrainer):
    def __init__(self, state_path, predictors, target_col, X, groups, test, n_splits, n_rsb, params, categorical_cols):
        self.categorical_cols = categorical_cols
        self.params = params
        if params["objective"] == "xentropy":
            self.y_max = X[target_col].max()
            self.y_min = X[target_col].min()
        super().__init__(state_path, predictors, target_col, X, groups, test, n_splits, n_rsb)

    def _get_importance(self, model, importance_type="gain"):
        feature = model.feature_name()
        importance = pd.DataFrame(
            {"features": feature, "importance": model.feature_importance(importance_type=importance_type)}
        )
        return importance

    def _fit(self, X_train, Y_train, X_valid, Y_valid, loop_seed):
        set_seed(loop_seed)
        self.params["random_seed"] = loop_seed
        tprint(f"LGBM params: {self.params}")
        if params["objective"] == "xentropy":
            Y_train = (Y_train - self.y_min) / (self.y_max - self.y_min)
            Y_valid = (Y_valid - self.y_min) / (self.y_max - self.y_min)

        dtrain = lgb.Dataset(
            X_train, label=Y_train, feature_name=self.predictors, categorical_feature=self.categorical_cols
        )
        dvalid = lgb.Dataset(
            X_valid, label=Y_valid, feature_name=self.predictors, categorical_feature=self.categorical_cols
        )
        model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dtrain, dvalid],
            num_boost_round=50000,
            early_stopping_rounds=100,
            verbose_eval=100,
        )
        tprint(f"best params {model.params}")
        ret = {}
        ret["model"] = model
        ret["importance"] = self._get_importance(model, importance_type="gain")
        if (self.params["random_seed"] == 0) and (self.fold_cnt == 1):
            tprint(f'importance(TOP30): {ret["importance"].sort_values(by="importance", ascending=False).head(30)}')
            tprint(f'importance(UND30): {ret["importance"].sort_values(by="importance", ascending=False).tail(30)}')
        return ret

    def _predict(self, model, X):
        pred = model.predict(X[self.predictors])
        if params["objective"] == "xentropy":
            pred = (pred * (self.y_max - self.y_min)) + self.y_min
        return pred


class MEDataset(Dataset):
    def __init__(self, is_train, feature, labels, pref, city, district, station):
        self.is_train = is_train
        self.feature = feature
        self.pref = pref
        self.city = city
        self.district = district
        self.station = station
        self.labels = labels

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        features = self.feature[idx]
        features = torch.Tensor(features)
        pref = torch.LongTensor([self.pref[idx]])
        city = torch.LongTensor([self.city[idx]])
        district = torch.LongTensor([self.district[idx]])
        station = torch.LongTensor([self.station[idx]])
        if not self.is_train:
            return features, pref, city, district, station
        else:
            y = self.labels[idx]
            y = torch.Tensor([y])
            return features, pref, city, district, station, y


class MLPModel(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super(MLPModel, self).__init__()
        pref_dim = 10
        city_dim = 100
        district_dim = 100
        station_dim = 100
        self.emb_pref = nn.Sequential(nn.Embedding(num_embeddings=48, embedding_dim=pref_dim))
        self.emb_city = nn.Sequential(nn.Embedding(num_embeddings=619, embedding_dim=city_dim))
        self.emb_district = nn.Sequential(nn.Embedding(num_embeddings=15457, embedding_dim=district_dim))
        self.emb_station = nn.Sequential(nn.Embedding(num_embeddings=3844, embedding_dim=station_dim))
        self.sq1 = nn.Sequential(
            nn.Linear(pref_dim + city_dim + district_dim + station_dim, 100),
            nn.PReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(dropout_rate),
            nn.Linear(100, 100),
            nn.PReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(dropout_rate),
        )
        self.sq2 = nn.Sequential(
            nn.Linear(input_dim + 100, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x, pref, city, district, station):
        y1 = self.emb_pref(pref.view(-1))
        y2 = self.emb_city(city.view(-1))
        y3 = self.emb_district(district.view(-1))
        y4 = self.emb_station(station.view(-1))
        y = torch.cat((y1, y2, y3, y4), dim=1)
        y = self.sq1(y)
        y = torch.cat((x, y), dim=1)
        y = self.sq2(y)
        return y


class MLPTrainer(GroupKfoldTrainer):
    def __init__(self, state_path, predictors, target_col, X, groups, test, n_splits, n_rsb, params, categorical_cols):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categorical_cols = categorical_cols
        self.numeric_cols = [x for x in predictors if x not in categorical_cols]
        self.params = params
        super().__init__(state_path, predictors, target_col, X, groups, test, n_splits, n_rsb)

    def _get_importance(self, model, importance_type="gain"):
        importance = pd.DataFrame({"features": [], "importance": []})
        return importance

    def _fit(self, X_train, Y_train, X_valid, Y_valid, loop_seed):
        set_seed(loop_seed)
        # preprocess
        # DataFrame -> numpy array
        _X_train = X_train[self.numeric_cols].values.copy()
        _X_valid = X_valid[self.numeric_cols].values.copy()
        _Y_train = Y_train.values.copy()
        _Y_valid = Y_valid.values.copy()
        _X_train_pref = X_train["pref"].astype(int).values.copy()
        _X_train_city = X_train["pref_city"].astype(int).values.copy()
        _X_train_district = X_train["pref_city_district"].astype(int).values.copy()
        _X_train_station = X_train["station"].astype(int).values.copy()
        _X_valid_pref = X_valid["pref"].astype(int).values.copy()
        _X_valid_city = X_valid["pref_city"].astype(int).values.copy()
        _X_valid_district = X_valid["pref_city_district"].astype(int).values.copy()
        _X_valid_station = X_valid["station"].astype(int).values.copy()

        # train
        ret = dict()

        # numpy array -> data loader
        train_set = MEDataset(
            is_train=True,
            feature=_X_train,
            labels=_Y_train,
            pref=_X_train_pref,
            city=_X_train_city,
            district=_X_train_district,
            station=_X_train_station,
        )
        train_loader = DataLoader(train_set, batch_size=self.params["batch_size"], shuffle=True, num_workers=0)
        val_set = MEDataset(
            is_train=True,
            feature=_X_valid,
            labels=_Y_valid,
            pref=_X_valid_pref,
            city=_X_valid_city,
            district=_X_valid_district,
            station=_X_valid_station,
        )
        val_loader = DataLoader(val_set, batch_size=10240, num_workers=0)

        # create network, optimizer, scheduler
        network = MLPModel(_X_train.shape[1], dropout_rate=0.150)
        optimizer = Adam(network.parameters(), lr=self.params["lr"])
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.params["factor"],
            patience=self.params["patience"],
            min_lr=self.params["min_lr"],
            threshold=1e-10,
            verbose=True,
        )
        val_loss_plot = []
        # begin training...
        torch.backends.cudnn.benchmark = True
        self.criterion = nn.L1Loss()
        best_score = 100000
        best_iteration = 0
        for epoch in range(self.params["n_epoch"]):
            # train model...
            for train_batch in train_loader:
                x, pref, city, district, station, label = train_batch
                x = x.to(self.device)
                pref = pref.to(self.device)
                city = city.to(self.device)
                district = district.to(self.device)
                station = station.to(self.device)
                label = label.to(self.device)

                network.train()
                network = network.to(self.device)
                train_preds = network.forward(x, pref, city, district, station)
                train_loss = self.criterion(train_preds, label)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            # get validation score...
            network.eval()
            val_preds, val_targs = [], []
            for val_batch in val_loader:
                x, pref, city, district, station, label = val_batch
                with torch.no_grad():
                    x = x.to(self.device)
                    pref = pref.to(self.device)
                    city = city.to(self.device)
                    district = district.to(self.device)
                    station = station.to(self.device)
                    label = label.to(self.device)
                    network = network.to(self.device)

                    val_preds.append(network.forward(x, pref, city, district, station))
                    val_targs.append(label)
            val_preds = torch.cat(val_preds, axis=0)
            val_targs = torch.cat(val_targs, axis=0)
            val_loss = self.criterion(val_preds, val_targs).cpu().detach().numpy()
            scheduler.step(val_loss)
            print(f"epoch {epoch:0>4}: val_loss {val_loss}")
            val_loss_plot.append(float(val_loss))

            # モデルを保存
            if val_loss < best_score:
                # モデルそのものを保存すると参照渡しになるので、わざわざ重みをコピーしてあとで入れるみたいなことをしている。
                best_model_wts = copy.deepcopy(network.state_dict())
                best_score = val_loss
                best_iteration = epoch
                print(f"model updated. best score is {best_score}")
            # early_stopping
            if (epoch - best_iteration) > self.params["early_stopping_rounds"]:
                print(f"early stopping. use {best_iteration}th model")
                break
        print("\n")
        tprint(f"NN result: {val_loss_plot}")

        # 一番良いところを持ってくる
        network.load_state_dict(best_model_wts)
        ret["model"] = network
        return ret

    def _predict(self, model, X):
        model.eval()
        with torch.no_grad():
            model = model.to(self.device)
            _X = torch.Tensor(X[self.numeric_cols].values).to(self.device)
            pref = torch.LongTensor(X["pref"].astype(int).values).to(self.device)
            city = torch.LongTensor(X["pref_city"].astype(int).values).to(self.device)
            district = torch.LongTensor(X["pref_city_district"].astype(int).values).to(self.device)
            station = torch.LongTensor(X["station"].astype(int).values).to(self.device)
            preds = model.forward(_X, pref, city, district, station).view(-1)
            preds = preds.to("cpu").detach().numpy().copy()
        return preds


class XGBTrainer(GroupKfoldTrainer):
    def __init__(self, state_path, predictors, target_col, X, groups, test, n_splits, n_rsb, params, categorical_cols):
        self.categorical_cols = categorical_cols
        self.params = params
        super().__init__(state_path, predictors, target_col, X, groups, test, n_splits, n_rsb)

    def _get_importance(self, model):
        imp_dict = model.get_fscore()
        # fill default columns
        for col in self.predictors:
            if col not in imp_dict.keys():
                imp_dict[col] = 0

        # to dataframe
        importance = pd.DataFrame.from_dict(imp_dict, orient="index", columns=["importance"])
        importance["features"] = importance.index
        importance = importance[["features", "importance"]].reset_index(drop=True)
        return importance

    def _fit(self, X_train, Y_train, X_valid, Y_valid, loop_seed):
        set_seed(loop_seed)
        self.params["seed"] = loop_seed
        tprint(f"XGB params: {self.params}")

        dtrain = xgb.DMatrix(X_train, label=Y_train, feature_names=self.predictors)
        dvalid = xgb.DMatrix(X_valid, label=Y_valid, feature_names=self.predictors)
        model = xgb.train(
            self.params,
            dtrain,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            num_boost_round=100000,
            early_stopping_rounds=100,
            verbose_eval=1000,
        )
        ret = {}
        ret["model"] = model
        ret["importance"] = self._get_importance(model)
        if (self.params["seed"] == 0) and (self.fold_cnt == 1):
            tprint(f'importance(TOP20): {ret["importance"].sort_values(by="importance", ascending=False).head(20)}')
            tprint(f'importance(UND20): {ret["importance"].sort_values(by="importance", ascending=False).tail(20)}')
        return ret

    def _predict(self, model, X):
        dtest = xgb.DMatrix(X[self.predictors], feature_names=self.predictors)
        return model.predict(dtest, ntree_limit=model.best_ntree_limit)


def get_score(weights, train_idx, oofs, labels):
    blend = np.zeros_like(oofs[0][train_idx])

    for oof, weight in zip(oofs[:-1], weights):
        blend += weight * oof[train_idx]

    blend += (1 - np.sum(weights)) * oofs[-1][train_idx]

    return mean_absolute_error(labels[train_idx], blend)


@Cache("./cache")
def get_best_weights(oofs, labels):
    weight_list = []
    weights = np.array([1 / len(oofs) for x in range(len(oofs) - 1)])

    for n_splits in tqdm([5, 6]):
        for i in range(2):
            kf = KFold(n_splits=n_splits, random_state=i, shuffle=True)
            for fold, (train_idx, valid_idx) in enumerate(kf.split(X=oofs[0])):
                res = minimize(get_score, weights, args=(train_idx, oofs, labels), method="Nelder-Mead", tol=1e-6)
                tprint(f"i: {i} fold: {fold} res.x: {res.x}")
                weight_list.append(res.x)

    mean_weight = np.mean(weight_list, axis=0)
    tprint(f"optimized weight: {mean_weight}")
    return mean_weight


@Cache("./cache")
def fit_trainer(trainer_instance):
    trainer_instance.fit()
    del trainer_instance.X
    trainer_instance.save()
    return trainer_instance


if __name__ == "__main__":
    debug = False
    tprint(f"debug mode {debug}")
    tprint("loading data")
    (train_df, test_df, sample_submission) = get_data()
    if debug:
        train_df = train_df.sample(10000, random_state=100).reset_index(drop=True)
    tprint("preprocessing data")
    (train_df, test_df, train_df_nn, test_df_nn) = preprocess(train_df, test_df)
    tprint("reduce memory usage")
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)
    train_df_nn = reduce_mem_usage(train_df_nn)
    test_df_nn = reduce_mem_usage(test_df_nn)

    if not debug:
        feather.write_dataframe(train_df, "./data/processed/train_df.feather")
        feather.write_dataframe(test_df, "./data/processed/test_df.feather")
        # float8があるのでpickleで保存
        with open("./data/processed/train_df_nn.pickle", "wb") as f:
            pickle.dump(train_df_nn, f)
        with open("./data/processed/test_df_nn.pickle", "wb") as f:
            pickle.dump(test_df_nn, f)
    del train_df_nn, test_df_nn
    n_splits = 6
    if debug:
        n_rsb = 1
    else:
        n_rsb = 5
    predictors = [x for x in train_df.columns if x not in ["y"]]
    tprint(f"predictors length is {len(predictors)}")
    stage2_oofs = []
    stage2_preds = []

    tprint("TRAIN LightGBM")
    params = {
        "objective": "mae",
        "metric": "mae",
        "boosting_type": "gbdt",
        "device": "cpu",
        "feature_fraction": 0.8,
        "num_leaves": 2048,
        "learning_rate": 0.1,
        "verbosity": -1,
    }
    lgb_trainer = LGBTrainer(
        state_path="./models",
        predictors=predictors,
        target_col="y",
        X=train_df,
        groups=train_df["base_year"],
        test=test_df,
        n_splits=n_splits,
        n_rsb=4,
        params=params,
        categorical_cols=["pref", "pref_city", "pref_city_district"],
    )
    lgb_trainer = fit_trainer(lgb_trainer)
    stage2_oofs.append(lgb_trainer.oof)
    stage2_preds.append(lgb_trainer.pred)
    tprint(f"LGBM SCORE IS {np.mean(lgb_trainer.validation_score):.4f}")

    tprint("TRAIN XGBoost")
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_depth": 12,
        "eta": 0.01,
        "tree_method": "hist" if debug else "gpu_hist",
    }
    xgb_trainer = XGBTrainer(
        state_path="./models",
        predictors=predictors,
        target_col="y",
        X=train_df,
        groups=train_df["base_year"],
        test=test_df,
        n_splits=n_splits,
        n_rsb=n_rsb,
        params=params,
        categorical_cols=[],
    )
    xgb_trainer = fit_trainer(xgb_trainer)
    stage2_oofs.append(xgb_trainer.oof)
    stage2_preds.append(xgb_trainer.pred)

    # ここからNN
    with open("./data/processed/train_df_nn.pickle", "rb") as f:
        train_df = pickle.load(f)
    with open("./data/processed/test_df_nn.pickle", "rb") as f:
        test_df = pickle.load(f)
    predictors = [x for x in train_df.columns if x not in ["y"]]
    tprint(f"predictors length is {len(predictors)}")

    tprint("TRAIN NN")
    mlp_trainer = MLPTrainer(
        state_path="./models",
        predictors=predictors,
        target_col="y",
        X=train_df,
        groups=train_df["base_year"],
        test=test_df,
        n_splits=n_splits,
        n_rsb=1,
        params={
            "n_epoch": 10 if debug else 1000,
            "lr": 1e-3,
            "batch_size": 512,
            "patience": 20,
            "factor": 0.1,
            "early_stopping_rounds": 50,
            "min_lr": 1e-5,
        },
        categorical_cols=["pref", "pref_city", "pref_city_district", "station"],
    )
    mlp_trainer = fit_trainer(mlp_trainer)
    stage2_oofs.append(mlp_trainer.oof)
    stage2_preds.append(mlp_trainer.pred)

    # blending
    best_weights = get_best_weights(stage2_oofs, train_df.loc[lgb_trainer.valid_idx, "y"].values)
    best_weights = np.insert(best_weights, len(best_weights), 1 - np.sum(best_weights))
    tprint("post processed optimized weight", best_weights)
    oof_preds = np.stack(stage2_oofs).transpose(1, 0).dot(best_weights)
    blend_preds = np.stack(stage2_preds).transpose(1, 0).dot(best_weights)
    tprint("final oof score", mean_absolute_error(train_df.loc[lgb_trainer.valid_idx, "y"].values, oof_preds))
    tprint("writing result...")

    if not debug:
        with open("./models/final_oof_and_pred.pickle", "wb") as f:
            pickle.dump([oof_preds, blend_preds], f)
        # submit
        sample_submission["取引価格（総額）_log"] = blend_preds
        sample_submission.to_csv("./submit.csv", index=False)
        # submit
        sample_submission["取引価格（総額）_log"] = blend_preds * 0.95
        sample_submission.to_csv("./submit_95.csv", index=False)
        # submit
        sample_submission["取引価格（総額）_log"] = blend_preds * 1.05
        sample_submission.to_csv("./submit_05.csv", index=False)
    tprint("---おわり---")