from typing import Any, Union, Dict, List, Callable, Optional, ByteString, Tuple
import hashlib
import pickle
from pandas.util import hash_pandas_object
import pandas as pd
from pathlib import Path
from inspect import signature
import numpy as np
from collections import OrderedDict
import operator
import functools
import json
from json import JSONEncoder

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping


def _hash(obj: ByteString) -> str:
    return hashlib.md5(obj).hexdigest()


class Cache:
    def __init__(self, dir_path: str, rerun: bool = False, with_param: bool = False):
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(exist_ok=True)
        self.with_param = with_param
        self.rerun = rerun

    def __call__(self, func: Callable):
        func_name = func.__name__

        def wrapper(*args, **kwargs):
            sig = signature(func)
            # ignore default value
            bound_args = sig.bind(*args, **kwargs)
            unique_id: str = self._get_unique_id(bound_args.arguments)
            path: Path = self.dir_path.joinpath(f"{func_name}_{unique_id}")

            logger.info(f"{func_name}_{unique_id} has been called")
            ret = Cache._read_cache(path, rerun=self.rerun)
            if ret is None:
                logger.info(f"{func_name}_{unique_id} cache not found")
                ret = func(*args, **kwargs)
                Cache._write(path, ret)
            return ret

        return wrapper