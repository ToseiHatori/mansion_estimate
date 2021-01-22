"""
# colab
from google.colab import drive
drive.mount('/content/drive/')
%cd "/content/drive/My Drive/data/"
!pip install category-encoders
!pip install jeraconv
"""
import argparse
import copy
import datetime
import gc
import glob
import logging
import os
import pickle
import random
import re
import subprocess
from pathlib import Path
from typing import Dict, List

import category_encoders as ce
import feather
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from jeraconv import jeraconv
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# 日付とutil系
dt_now = datetime.datetime.now()
dt_now = dt_now.strftime("%Y%m%d_%H:%M")
gc.enable()
pd.options.display.max_columns = None

# 流石にロガーはglobalを使うぞ
formatter = "%(levelname)s : %(asctime)s : %(message)s"
logging.basicConfig(level=logging.DEBUG, format=formatter, filename=f"log/logger_{dt_now}.log")
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_data():
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
        # df["plan_S"] = [int("S" in x) for x in df["間取り"].fillna("")]

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
        # df["structure_block"] = [int("ブロック造" in x) for x in df["建物の構造"].fillna("")]
        df["structure_wood"] = [int("木造" in x) for x in df["建物の構造"].fillna("")]
        # df["structure_lightiron"] = [int("軽量鉄骨造" in x) for x in df["建物の構造"].fillna("")]
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
        # df["usage_workshop"] = [int("作業場" in x) for x in df["用途"].fillna("")]
        # df["usage_factory"] = [int("工場" in x) for x in df["用途"].fillna("")]

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
        # df["reason_burden"] = [int("他の権利・負担付き" in x) for x in df["取引の事情等"].fillna("")]
        df["reason_auction"] = [int("調停・競売等" in x) for x in df["取引の事情等"].fillna("")]
        df["reason_defects"] = [int("瑕疵有りの可能性" in x) for x in df["取引の事情等"].fillna("")]
        df["reason_related_parties"] = [int("関係者間取引" in x) for x in df["取引の事情等"].fillna("")]

        # 容積率 x 面積
        df["floor_area_ratio_x_area"] = df["floor_area_ratio"] * df["area"]
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

    logger.debug(f"train head : {train_df.head()}")
    logger.debug(f"test head : {test_df.head()}")
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
        self.file_path = self.state_path.joinpath(f"{self.name}_{dt_now}.pickle")
        # 無法者なのでここで呼んじゃう
        self.fit()
        self.save()

    def loss_(self, predictions, targets):
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
            logger.info(f"START FOLD {fold_cnt}")

            # DataFrame -> train, valid
            X_train, X_valid = self.X.loc[train_idx, self.predictors], self.X.loc[valid_idx, self.predictors]
            Y_train, Y_valid = self.X.loc[train_idx, self.target_col], self.X.loc[valid_idx, self.target_col]
            logger.debug(f"training years : {set(self.X.loc[train_idx, 'base_year'])}")
            logger.debug(f"validation years : {set(self.X.loc[valid_idx, 'base_year'])}")

            # random seed blending
            for rsb_idx in range(self.n_rsb):
                logger.info(f"     fitting {rsb_idx + 1} th loop of {self.n_rsb}")
                # 学習
                ret = self._fit(X_train, Y_train, X_valid, Y_valid, loop_seed=rsb_idx)
                # save models
                self.folds.append(ret)
                model = ret["model"]

                # oof, predに対して予測
                pred_oof = self._predict(model, X_valid)
                pred_test = self._predict(model, self.test)
                if type(pred_oof) == torch.Tensor:
                    pred_oof = pred_oof.cpu().detach().numpy()
                    pred_test = pred_test.cpu().detach().numpy()
                # 格納
                self.oof[valid_idx] += pred_oof / self.n_rsb
                self.pred += pred_test / (self.n_splits * self.n_rsb)

                # single fold validation　score
                _validation_score = self.loss_(pred_oof, Y_valid.values)
                logger.info((f"     finished {rsb_idx + 1}" + f"th loop WITH {_validation_score:.6f}"))

            # rsb validation　score
            _validation_score = self.loss_(self.oof[valid_idx], Y_valid.values)
            self.validation_score.append(_validation_score)
            logger.info(f"     END FOLD {fold_cnt} WITH {_validation_score:.6f}")
            logger.info("----切り取り----\n")

        # validation score of fold mean
        cv_score = np.mean(self.validation_score, axis=0)
        cv_std = np.std(self.validation_score, axis=0)
        logger.info(f"TOTAL CV SCORE is : {cv_score:.6f} +- {cv_std:.4f}")
        logger.info("----終わり----\n")


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
        logger.debug(f"LGBM params: {self.params}")

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
        logger.debug(f'importance: {ret["importance"]}')
        return ret

    def _predict(self, model, X):
        return model.predict(X[self.predictors])


class MEDataset(Dataset):
    def __init__(self, is_train, feature, labels, pref, city, district):
        self.is_train = is_train
        self.feature = feature
        self.pref = pref
        self.city = city
        self.district = district
        self.labels = labels

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        features = self.feature[idx]
        features = torch.Tensor(features)
        pref = torch.LongTensor([self.pref[idx]])
        city = torch.LongTensor([self.city[idx]])
        district = torch.LongTensor([self.district[idx]])
        if not self.is_train:
            return features, pref, city, district
        else:
            y = self.labels[idx]
            y = torch.Tensor([y])
            return features, pref, city, district, y


class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        pref_dim = 10
        city_dim = 100
        district_dim = 1000
        self.emb_pref = nn.Sequential(
            nn.Embedding(num_embeddings=42, embedding_dim=pref_dim),
            # nn.Linear(pref_dim, pref_dim),
            # nn.PReLU(),
            # nn.BatchNorm1d(pref_dim),
            # nn.Dropout(0.5),
        )
        self.emb_city = nn.Sequential(
            nn.Embedding(num_embeddings=618, embedding_dim=city_dim),
            # nn.Linear(city_dim, city_dim),
            # nn.PReLU(),
            # nn.BatchNorm1d(city_dim),
            # nn.Dropout(0.5),
        )
        self.emb_district = nn.Sequential(
            nn.Embedding(num_embeddings=15418, embedding_dim=district_dim),
            # nn.Linear(district_dim, district_dim),
            # nn.PReLU(),
            # nn.BatchNorm1d(district_dim),
            # nn.Dropout(0.5),
        )
        self.sq1 = nn.Sequential(
            nn.Linear(pref_dim + city_dim + district_dim, 100),
            # nn.Linear(1024, 1024),
            # nn.PReLU(),
            # nn.BatchNorm1d(1024),
            # nn.Dropout(0.5),
        )
        self.sq2 = nn.Sequential(
            nn.Linear(input_dim + 100, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, pref, city, district):
        y1 = self.emb_pref(pref.reshape(-1))
        y2 = self.emb_city(city.reshape(-1))
        y3 = self.emb_district(district.reshape(-1))
        y = torch.cat((y1, y2, y3), dim=1)
        y = self.sq1(y)
        y = torch.cat((x, y), dim=1)
        y = self.sq2(y)
        return y


class MLPTrainer(GroupKfoldTrainer):
    def __init__(self, state_path, predictors, target_col, X, groups, test, n_splits, n_rsb, params, categorical_cols):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categorical_cols = categorical_cols
        self.numeric_cols = [x for x in predictors if x not in categorical_cols]
        X, test = self.preprocess(X, test, self.numeric_cols)
        self.params = params
        super().__init__(state_path, predictors, target_col, X, groups, test, n_splits, n_rsb)

    def _get_importance(self, model, importance_type="gain"):
        importance = pd.DataFrame({"features": [], "importance": []})
        return importance

    @staticmethod
    def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame, numeric_cols: List[str]):
        logger.info("scaling...")
        # transformer = MinMaxScaler()
        transformer = RobustScaler()
        train_df["is_train"] = 1
        test_df["is_train"] = 0
        df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
        df[numeric_cols] = transformer.fit_transform(df[numeric_cols])
        train_df = df[df["is_train"] == 1].fillna(0).drop("is_train", axis=1).reset_index(drop=False)
        test_df = df[df["is_train"] == 0].fillna(0).drop("is_train", axis=1).reset_index(drop=False)
        del df
        gc.collect()
        return train_df, test_df

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
        _X_valid_pref = X_valid["pref"].astype(int).values.copy()
        _X_valid_city = X_valid["pref_city"].astype(int).values.copy()
        _X_valid_district = X_valid["pref_city_district"].astype(int).values.copy()

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
        )
        train_loader = DataLoader(train_set, batch_size=1024, shuffle=True, num_workers=0)
        val_set = MEDataset(
            is_train=True,
            feature=_X_valid,
            labels=_Y_valid,
            pref=_X_valid_pref,
            city=_X_valid_city,
            district=_X_valid_district,
        )
        val_loader = DataLoader(val_set, batch_size=1024, num_workers=0)

        # create network, optimizer, scheduler
        network = MLPModel(_X_train.shape[1])
        optimizer = Adam(network.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)
        val_loss_plot = []
        # begin training...
        torch.backends.cudnn.benchmark = True
        self.criterion = nn.L1Loss()
        best_score = 1000
        for epoch in range(100):
            # train model...
            for train_batch in train_loader:
                x, pref, city, district, label = train_batch
                x = x.to(self.device)
                pref = pref.to(self.device)
                city = city.to(self.device)
                district = district.to(self.device)
                label = label.to(self.device)

                network.train()
                network = network.to(self.device)
                train_preds = network.forward(x, pref, city, district)
                train_loss = self.criterion(train_preds, label)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            # get validation score...
            network.eval()
            val_preds, val_targs = [], []
            for val_batch in val_loader:
                x, pref, city, district, label = val_batch
                with torch.no_grad():
                    x = x.to(self.device)
                    pref = pref.to(self.device)
                    city = city.to(self.device)
                    district = district.to(self.device)
                    label = label.to(self.device)
                    network = network.to(self.device)

                    val_preds.append(network.forward(x, pref, city, district))
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
        logger.debug(f"NN result: {val_loss_plot}")

        # 一番良いところを持ってくる
        network.load_state_dict(best_model_wts)
        ret["model"] = network
        return ret

    def _predict(self, model, X):
        _X_valid = torch.Tensor(X[self.numeric_cols].values).to(self.device)
        pref = torch.LongTensor(X["pref"].astype(int).values).to(self.device)
        city = torch.LongTensor(X["pref_city"].astype(int).values).to(self.device)
        district = torch.LongTensor(X["pref_city_district"].astype(int).values).to(self.device)
        preds = model.forward(_X_valid, pref, city, district)
        preds = preds.cpu().detach().numpy().reshape(-1)
        return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug")
    parser.add_argument("--update_dm")
    args = parser.parse_args()
    debug = args.debug
    update_dm = args.update_dm
    if update_dm is None:
        update_dm = False
    if debug is None:
        debug = False
    logger.info(f"debug mode {debug}")
    logger.info(f"update dm  {update_dm}")

    train_df_path = "./data/processed/train_df.feather"
    test_df_path = "./data/processed/test_df.feather"
    if (not os.path.exists(train_df_path)) | update_dm:
        logger.info("loading data")
        train_df, test_df, sample_submission = get_data()
        logger.info("preprocessing data")
        train_df, test_df = preprocess(train_df, test_df)
        feather.write_dataframe(train_df, train_df_path)
        feather.write_dataframe(test_df, test_df_path)
    else:
        train_df = feather.read_dataframe(train_df_path)
        test_df = feather.read_dataframe(test_df_path)

    logger.info("training data")
    predictors = [
        x
        for x in train_df.columns
        if x not in ["ID", "y", "base_year", "floor_area_ratio", "te_pref", "te_pref_city", "te_pref_city_district"]
    ]
    logger.debug(f"predictors: {predictors}")
    if debug:
        n_splits = 2
        n_rsb = 1
    else:
        n_splits = 6
        n_rsb = 5
    # params = {
    #    "objective": "mae",
    #    "boosting_type": "gbdt",
    #    "subsample": 0.8,
    #    "colsample_bytree": 0.8,
    #    "verbosity": -1,
    # }
    # lgb_booster = LGBTrainer(
    #    state_path="./models",
    #    predictors=predictors,
    #    target_col="y",
    #    X=train_df,
    #    groups=train_df["base_year"],
    #    test=test_df,
    #    n_splits=n_splits,
    #    n_rsb=n_rsb,
    #    params=params,
    #    categorical_cols=["pref", "pref_city", "pref_city_district", "remodeling"],
    # )
    predictors = [
        x for x in train_df.columns if x not in ["ID", "y", "te_pref", "te_pref_city", "te_pref_city_district"]
    ]
    if debug:
        n_splits = 2
        n_rsb = 1
    else:
        n_splits = 6
        n_rsb = 1
    mlp_trainer = MLPTrainer(
        state_path="./models",
        predictors=predictors,
        target_col="y",
        X=train_df,
        groups=train_df["base_year"],
        test=test_df,
        n_splits=n_splits,
        n_rsb=n_rsb,
        params={},
        categorical_cols=["pref", "pref_city", "pref_city_district"],
    )
    # submit
    sample_submission["取引価格（総額）_log"] = mlp_trainer.pred
    sample_submission.to_csv("./submit.csv", index=False)
