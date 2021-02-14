from typing import Any, Union, Dict, List, Callable, Optional, ByteString, Tuple
from pathlib import Path
import hashlib
import inspect
from inspect import signature
import pickle
import os


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
            func_args = ",".join("{k}={v}".format(k=k, v=v) for k, v in bound_args.arguments.items())
            for k, v in bound_args.arguments.items():
                if k == "trainer_instance":
                    # trainerがあればメンバー変数を取得
                    func_args_list.append(",".join("{_k}={_v}".format(_k=_k, _v=_v) for _k, _v in vars(v).items()))
                    # 関数があればその定義を取得しておく
                    for x in inspect.getmembers(v, inspect.ismethod):
                        func_args_list.append(inspect.getsource(x[1]))
                else:
                    func_args_list.append(f"{k}={v}")
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