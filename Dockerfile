# kaggleのpython環境をベースにする
FROM gcr.io/kaggle-images/python:v94

# ライブラリの追加インストール
# jeraconvだけは必要
RUN pip install -U pip && \
    pip install jeraconv
