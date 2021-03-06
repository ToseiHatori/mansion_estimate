# kaggleのpython環境をベースにする, ver94に固定
FROM gcr.io/kaggle-images/python:v94

# ライブラリの追加インストール
RUN pip install -U pip && \
    pip install xfeat && \
    pip install pytorch-tabnet
