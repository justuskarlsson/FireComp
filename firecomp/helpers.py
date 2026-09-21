# no firecomp imports

from contextlib import contextmanager
import dataclasses
import json
import re
import shutil
from time import time
from datetime import datetime, date, timezone
import typing
from uuid import uuid4

from pathlib import Path


@contextmanager
def ram_paths(*paths):
    copied = []
    SHM_DIR = Path("/dev/shm")
    try:
        for path in paths:
            name = str(uuid4())
            ext = Path(path).suffix
            copied.append(SHM_DIR / f"{name}{ext}")
            shutil.copy2(path, copied[-1])
        yield copied
    finally:
        for path in copied:
            path.unlink()


def dump_default(o):
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, date):
        return o.isoformat()
    raise TypeError


def prepare_dump_json(data) -> typing.Any:
    return json.loads(json.dumps(data, default=dump_default))


def dump_json(data, path, indent=2):
    with open(path, "w") as f:
        json.dump(data, f, default=dump_default, indent=indent)


def load_json(path):
    ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T.*)?$")

    def object_hook(d):
        for k, v in list(d.items()):
            if isinstance(v, str) and ISO_RE.match(v):
                d[k] = datetime.fromisoformat(v)
                if "T" not in v:
                    d[k] = d[k].date()
        return d

    with open(path, "r") as f:
        return json.load(f, object_hook=object_hook)


def load_dataclass(path, cls):
    data = load_json(path)
    if typing.get_origin(cls) == dict:
        cls = typing.get_args(cls)[1]
        return {k: cls(**v) for k, v in data.items()}
    elif typing.get_origin(cls) == list:
        cls = typing.get_args(cls)[0]
        return [cls(**d) for d in data]
    return cls(**data)


def log_time(func):
    num_calls = 0

    def wrapper(*args, **kwargs):
        nonlocal num_calls
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        t = end_time - start_time
        minutes = t // 60
        s = t % 60
        print(f"TIME: {func.__name__}@{num_calls} took {minutes:.2f} min {s:.2f} s")
        num_calls += 1
        return result

    return wrapper


class Pipeline:
    def __init__(self):
        self.pipeline: list[typing.Callable] = []
        for name in self.__dir__():
            val = getattr(self, name)
            if hasattr(val, "_pipeline_creation_t"):
                self.pipeline.append((val, val._pipeline_creation_t))
        self.pipeline.sort(key=lambda x: x[1])
        self.pipeline = [x[0] for x in self.pipeline]


def pipeline(fn):
    fn._pipeline_creation_t = time()

    return fn


def run_pipeline(pipeline: Pipeline):
    import sys

    for i in sys.argv[1:]:
        i = int(i)
        pipeline.pipeline[i]()


def geod_to_pix_bbox(bbox: list[float], deg_cell_size: float):

    return [
        round((bbox[0] + 180) / deg_cell_size),
        round((bbox[1] + 90) / deg_cell_size),
        round((bbox[2] + 180) / deg_cell_size),
        round((bbox[3] + 90) / deg_cell_size),
    ]
