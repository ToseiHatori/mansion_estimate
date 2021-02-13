from typing import Any, Union, Dict, List, Callable, Optional, ByteString, Tuple
from pathlib import Path
import hashlib
import inspect
import pickle
import os


class Cache:
    def __init__(self, dir_path: str):
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(exist_ok=True)

    def __call__(self, func: Callable):
        func_name = func.__name__
        func_hash = str(func.__hash__())
        func_source = inspect.getsource(func)
        func_source_hash = hashlib.md5(func_source.encode("utf-8")).hexdigest()

        def wrapper(*args, **kwargs):
            # 関数ごとにキャッシュdirを作る
            cache_dir = self.dir_path / func_name
            cache_dir.mkdir(exist_ok=True)
            cache_path = cache_dir / (func_source_hash + ".pickle")

            if os.path.exists(cache_path):
                print("cache hit", cache_path)
                ret = self.load(cache_path)
            else:
                print("cache does not hit")
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