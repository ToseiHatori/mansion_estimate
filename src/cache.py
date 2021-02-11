from typing import Any, Union, Dict, List, Callable, Optional, ByteString, Tuple
from pathlib import Path
import hashlib


class Cache:
    def __init__(self, dir_path: str):
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(exist_ok=True)

    def __call__(self, func: Callable):
        func_name = func.__name__

        def wrapper(*args, **kwargs):
            print("args", *args)
            print("kwargs", **kwargs)
            # 関数定義
            print(func)
            """
            hashlib.sha512(password).hexdigest()
            cache_path = self.dir_path / func_name /
            if :
                print('cache hit')
                pass
            else:
                print('cache does not hit')
                ret = func(*args, **kwargs)
            """
            ret = func(*args, **kwargs)
            return ret

        return wrapper