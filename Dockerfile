# kaggleのpython環境をベースにする
FROM gcr.io/kaggle-images/python:v94

# ライブラリの追加インストール
RUN pip install -U pip && \
    pip install jeraconv
