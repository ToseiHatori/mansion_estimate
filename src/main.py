import argparse
import copy
import datetime
import gc
import glob
import hashlib
import inspect
import logging
import os
import pickle
import random
import re
import subprocess
from inspect import signature
from pathlib import Path
from typing import Any, ByteString, Callable, Dict, List, Optional, Tuple, Union

import category_encoders as ce
import feather
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from jeraconv import jeraconv
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

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
    torch.backends.cudnn.deterministic = False


@Cache("./cache")
def get_data(debug):
    unuse_columns = ["種類", "地域", "市区町村コード", "土地の形状", "間口", "延床面積（㎡）", "前面道路：方位", "前面道路：種類", "前面道路：幅員（ｍ）"]
    train_files = sorted(glob.glob("./data/raw/train/*"))
    if debug:
        train_files = train_files[0:3]
    train_df = []
    for file in train_files:
        train_df.append(pd.read_csv(file, low_memory=False))
    train_df = pd.concat(train_df, axis=0).drop(unuse_columns, axis=1).reset_index(drop=True)
    test_df = pd.read_csv("./data/raw/test.csv").drop(unuse_columns, axis=1).reset_index(drop=True)
    sample_submission = pd.read_csv("./data/raw/sample_submission.csv")
    return train_df, test_df, sample_submission


@Cache("./cache")
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
        df["最寄駅：距離（分）"] = [x if x != "1H30?2H" else "105" for x in df["最寄駅：距離（分）"]]
        df["最寄駅：距離（分）"] = [x if x != "1H?1H30" else "75" for x in df["最寄駅：距離（分）"]]
        df["最寄駅：距離（分）"] = [x if x != "2H?" else "120" for x in df["最寄駅：距離（分）"]]
        df["最寄駅：距離（分）"] = [x if x != "30分?60分" else "45" for x in df["最寄駅：距離（分）"]]
        df["time_to_station"] = df["最寄駅：距離（分）"].astype(float)

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
        df["面積（㎡）"] = [x if x != "2000㎡以上" else "2000" for x in df["面積（㎡）"]]
        df["面積（㎡）"] = [x if x != "m^2未満" else "2" for x in df["面積（㎡）"]]
        df["area"] = df["面積（㎡）"].astype(float)

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
        df["base_year"] = [int(x[0:4]) for x in df["取引時点"]]
        df["base_quarter"] = [int(x[6:7]) for x in df["取引時点"]]
        df["base_year_quater"] = [int(str(x) + str(y)) for x, y in zip(df["base_year"], df["base_quarter"])]
        df["passed_year"] = df["base_year"] - df["year_of_construction"]

        # 取引の事情等
        df["reason"] = df["取引の事情等"]
        df["reason_other"] = [int("その他事情有り" in x) for x in df["取引の事情等"].fillna("")]
        df["reason_burden"] = [int("他の権利・負担付き" in x) for x in df["取引の事情等"].fillna("")]
        df["reason_auction"] = [int("調停・競売等" in x) for x in df["取引の事情等"].fillna("")]
        df["reason_defects"] = [int("瑕疵有りの可能性" in x) for x in df["取引の事情等"].fillna("")]
        df["reason_related_parties"] = [int("関係者間取引" in x) for x in df["取引の事情等"].fillna("")]

        # いろいろな組み合わせ
        inter_cols = [
            "year_of_construction",
            "area",
            "passed_year",
            "time_to_station",
            "base_year",
            "base_quarter",
            "base_year_quater",
            "floor_area_ratio",
            "plan_num",
            "building_coverage_ratio",
        ]
        for i, col_1 in enumerate(inter_cols):
            for j, col_2 in enumerate(inter_cols):
                if i != j:
                    df[f"{col_1}_p_{col_2}"] = df[col_1] + df[col_2]
                    df[f"{col_1}_m_{col_2}"] = df[col_1] - df[col_2]
                    if (df[col_2] == 0).sum() == 0:
                        df[f"{col_1}_d_{col_2}"] = df[col_1] / df[col_2]
                if i < j:
                    df[f"{col_1}_x_{col_2}"] = df[col_1] * df[col_2]
    # null数
    original_columns = [
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
    train_df["null_num"] = train_df[original_columns].isnull().sum(axis=1)
    test_df["null_num"] = test_df[original_columns].isnull().sum(axis=1)

    # 不要なカラム削除
    train_df = train_df.drop(original_columns, axis=1)
    test_df = test_df.drop(original_columns, axis=1)

    # target encoding
    for col in ["pref", "pref_city", "pref_city_district"]:
        te_df = train_df.groupby([col, "base_year"])["y"].mean().reset_index(drop=False)
        te_df = te_df.rename(columns={"y": f"te_{col}"})
        te_df["base_year"] = te_df["base_year"] + 1
        train_df = train_df.merge(te_df, on=[col, "base_year"], how="left")
        test_df = test_df.merge(te_df, on=[col, "base_year"], how="left")

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
    train_df.loc[:, category_columns] = ce_oe.fit_transform(train_df[category_columns])
    test_df.loc[:, category_columns] = ce_oe.transform(test_df[category_columns])
    for cat in category_columns:
        idx = test_df[cat] == -1
        test_df.loc[idx, cat] = 0
    train_df[category_columns] = train_df[category_columns].astype(int)
    test_df[category_columns] = test_df[category_columns].astype(int)

    return train_df, test_df


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
        self.pred = np.zeros(len(test))
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

            # random seed blending
            for rsb_idx in range(self.n_rsb):
                tprint(f"     fitting {rsb_idx + 1} th loop of {self.n_rsb}")
                # 学習
                ret = self._fit(X_train, Y_train, X_valid, Y_valid, loop_seed=rsb_idx)
                # save models
                if (fold_cnt == 1) & (rsb_idx == 0):
                    self.folds.append(ret)
                model = ret["model"]

                # oof, predに対して予測
                pred_oof = self._predict(model, X_valid)
                pred_test = self._predict(model, self.test)

                # 格納
                self.oof[valid_idx] += pred_oof / self.n_rsb
                self.pred += pred_test / (self.n_splits * self.n_rsb)

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
        tprint(f"TOTAL CV SCORE is : {cv_score:.6f} +- {cv_std:.4f}")
        tprint("----終わり----\n")


class LGBTrainer(GroupKfoldTrainer):
    def __init__(self, state_path, predictors, target_col, X, groups, test, n_splits, n_rsb, params, categorical_cols):
        self.categorical_cols = categorical_cols
        self.params = params
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
            categorical_feature=self.categorical_cols,
            early_stopping_rounds=100,
            verbose_eval=100,
        )
        ret = {}
        ret["model"] = model
        ret["importance"] = self._get_importance(model, importance_type="gain")
        tprint(f'importance: {ret["importance"]}')
        return ret

    def _predict(self, model, X):
        return model.predict(X[self.predictors])


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
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        pref_dim = 10
        city_dim = 100
        district_dim = 1000
        station_dim = 100
        dropout_rate = 0.5
        self.emb_pref = nn.Sequential(
            nn.Embedding(num_embeddings=48, embedding_dim=pref_dim),
            # nn.Linear(pref_dim, pref_dim),
            # nn.PReLU(),
            # nn.BatchNorm1d(pref_dim),
            # nn.Dropout(dropout_rate),
        )
        self.emb_city = nn.Sequential(
            nn.Embedding(num_embeddings=619, embedding_dim=city_dim),
            # nn.Linear(city_dim, city_dim),
            # nn.PReLU(),
            # nn.BatchNorm1d(city_dim),
            # nn.Dropout(dropout_rate),
        )
        self.emb_district = nn.Sequential(
            nn.Embedding(num_embeddings=15419, embedding_dim=district_dim),
            # nn.Linear(district_dim, district_dim),
            # nn.PReLU(),
            # nn.BatchNorm1d(district_dim),
            # nn.Dropout(dropout_rate),
        )
        self.emb_station = nn.Sequential(
            nn.Embedding(num_embeddings=3833, embedding_dim=station_dim),
            # nn.Linear(station_dim, station_dim),
            # nn.PReLU(),
            # nn.BatchNorm1d(station_dim),
            # nn.Dropout(dropout_rate),
        )
        self.sq1 = nn.Sequential(
            nn.Linear(pref_dim + city_dim + district_dim + station_dim, 1000),
            nn.PReLU(),
            nn.BatchNorm1d(1000),
            nn.Dropout(dropout_rate),
            nn.Linear(1000, 100),
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
        _X, _test = self.preprocess(X, test, self.numeric_cols)
        self.params = params
        super().__init__(state_path, predictors, target_col, _X, groups, _test, n_splits, n_rsb)

    def _get_importance(self, model, importance_type="gain"):
        importance = pd.DataFrame({"features": [], "importance": []})
        return importance

    @staticmethod
    def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame, numeric_cols: List[str]):
        # -1埋め
        _train_df = train_df.copy()
        _test_df = test_df.copy()
        _train_df[numeric_cols] = _train_df[numeric_cols].fillna(-1)
        _test_df[numeric_cols] = _test_df[numeric_cols].fillna(-1)

        tprint("scaling...")
        transformer = MinMaxScaler()
        # transformer = RobustScaler()
        _train_df[numeric_cols] = transformer.fit_transform(_train_df[numeric_cols])
        _test_df[numeric_cols] = transformer.transform(_test_df[numeric_cols])
        gc.collect()
        return _train_df, _test_df

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
        network = MLPModel(_X_train.shape[1])
        optimizer = Adam(network.parameters(), lr=self.params["lr"])
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=self.params["factor"], patience=self.params["patience"], verbose=True
        )
        val_loss_plot = []
        # begin training...
        torch.backends.cudnn.benchmark = True
        self.criterion = nn.L1Loss()
        best_score = 100000
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
                print(f"model updated. best score is {best_score}")
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
            verbose_eval=100,
        )
        ret = {}
        ret["model"] = model
        ret["importance"] = self._get_importance(model)
        tprint(f'importance: {ret["importance"]}')
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
    trainer_instance.save()
    return trainer_instance


if __name__ == "__main__":
    debug = False
    tprint(f"debug mode {debug}")

    tprint("loading data")
    (train_df, test_df, sample_submission) = get_data(debug)
    tprint("preprocessing data")
    (train_df, test_df) = preprocess(train_df, test_df)
    if debug:
        train_df = train_df.sample(1000, random_state=100).reset_index(drop=True)
    predictors = [
        x for x in train_df.columns if x not in ["ID", "y", "te_pref", "te_pref_city", "te_pref_city_district"]
    ]
    if debug:
        n_splits = 2
        n_rsb = 1
    else:
        n_splits = 6
        n_rsb = 2

    tprint("TRAIN XGBoost")
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eta": 0.1,
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
        n_rsb=3,
        params=params,
        categorical_cols=[],
    )
    xgb_trainer = fit_trainer(xgb_trainer)

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
        params={"n_epoch": 1 if debug else 100, "lr": 1e-3, "batch_size": 512, "patience": 10, "factor": 0.1},
        categorical_cols=["pref", "pref_city", "pref_city_district", "station"],
    )
    mlp_trainer = fit_trainer(mlp_trainer)

    tprint("TRAIN LightGBM")
    params = {
        "objective": "mae",
        "boosting_type": "gbdt",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "device": "cpu",
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
        n_rsb=n_rsb,
        params=params,
        categorical_cols=["pref", "pref_city", "pref_city_district"],
    )
    lgb_trainer = fit_trainer(lgb_trainer)

    # blending
    stage2_oofs = [lgb_trainer.oof, mlp_trainer.oof, xgb_trainer.oof]
    stage2_preds = [lgb_trainer.pred, mlp_trainer.pred, xgb_trainer.pred]
    best_weights = get_best_weights(stage2_oofs, train_df["y"].values)
    best_weights = np.insert(best_weights, len(best_weights), 1 - np.sum(best_weights))
    tprint("post processed optimized weight", best_weights)
    oof_preds = np.stack(stage2_oofs).transpose(1, 0).dot(best_weights)
    blend_preds = np.stack(stage2_preds).transpose(1, 0).dot(best_weights)
    tprint("final oof score", mean_absolute_error(train_df["y"].values, oof_preds))

    # submit
    sample_submission["取引価格（総額）_log"] = blend_preds
    sample_submission.to_csv("./submit.csv", index=False)
    tprint("---おわり---")