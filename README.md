# colabで動かす場合
* colab pro, ハイメモリ, GPU環境下で実行
* 事前にこれを実行する
```
from google.colab import drive
drive.mount('/content/drive/')
%cd "/content/drive/My Drive/Colab Notebooks/mansion_estimate"
!pip install xgboost==1.3.3
!pip install category-encoders
!pip install xfeat
!export CUDA_LAUNCH_BLOCKING=1
```

## 備考
* 外部データは事前に作成して `data/external/` に配置しておく


# dockerを使って動かす場合
## build docker image
* kaggle docker
```
docker-compose up --build
```

## start container
```
docker-compose up
```

## run python script
```
docker exec -it mansion_estimate_jupyter_1 python src/main.py 
```
