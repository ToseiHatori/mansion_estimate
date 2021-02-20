import pandas as pd
import numpy as np
import glob
import requests
import tqdm
import time
import os
from bs4 import BeautifulSoup


def get_data():
    unuse_columns = ["種類", "地域", "市区町村コード", "土地の形状", "間口", "延床面積（㎡）", "前面道路：方位", "前面道路：種類", "前面道路：幅員（ｍ）"]
    train_files = sorted(glob.glob("./data/raw/train/*"))
    train_df = []
    for file in train_files:
        train_df.append(pd.read_csv(file, low_memory=False))
    train_df = pd.concat(train_df, axis=0).drop(unuse_columns, axis=1).reset_index(drop=True)
    test_df = pd.read_csv("./data/raw/test.csv").drop(unuse_columns, axis=1).reset_index(drop=True)
    sample_submission = pd.read_csv("./data/raw/sample_submission.csv")
    return train_df, test_df, sample_submission


if __name__ == "__main__":
    train_df, test_df, sample_submission = get_data()
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    df["pref"] = df["都道府県名"]
    df["pref_city"] = df["都道府県名"] + df["市区町村名"]
    df["pref_city_district"] = df["都道府県名"] + df["市区町村名"] + df["地区名"]
    del train_df, test_df, sample_submission
    districts = df["pref_city_district"].unique()

    csv_path = "./data/processed/district_latlon.csv"
    if os.path.exists(csv_path):
        district_df = pd.read_csv(csv_path)
    else:
        lats = np.zeros(len(districts))
        lons = np.zeros(len(districts))
        district_df = pd.DataFrame([districts, lats, lons]).T
        district_df.columns = ["pref_city_district", "lat", "lon"]
        district_df.to_csv(csv_path, index=False)
    err_districts = []
    for i, row in tqdm.tqdm(enumerate(district_df.iterrows())):
        if (row[1].lat == 0) | (row[1].lon == 0):
            district = row[1].pref_city_district
            r = requests.get(f"https://www.geocoding.jp/api/?q={district}")
            soup = BeautifulSoup(r.text, "html.parser")
            district_df.iloc[i, 1] = float(soup.lat.contents[0])
            district_df.iloc[i, 2] = float(soup.lng.contents[0])
            district_df.to_csv(csv_path, index=False)
            time.sleep(10)