from contextlib import contextmanager
from dataclasses import Field, asdict, dataclass, field, fields
import datetime
from glob import glob
import os
from typing import Annotated, Literal, Optional, Type, TypeVar, TypedDict
import typing
import h5py
from firecomp.config import config
import numpy as np

T = TypeVar("T")


def fragment_id_find(store_dir: str):
    matches = glob(os.path.join(store_dir, "*"))
    matches = [m for m in matches if os.path.isdir(m)]
    if len(matches) == 0:
        return 0
    matches = sorted(matches)
    return int(os.path.basename(matches[-1]).split(".")[0]) + 1


def get_fragment_str(fragment_id: int):
    return f"{fragment_id:06d}"


def final_type(t):
    if typing.get_args(t):
        return typing.get_args(t)[0]
    return t


class Store:
    # @classmethod
    # def load_all(cls, h5: h5py.File):
    #     kv = {}
    #     for attr in h5.keys():
    #         kv[attr] = cls.load(h5, attr)
    #     return cls(**kv)

    @classmethod
    def load(cls, h5: h5py.File, attr: str, key: Optional[str] = None):
        group = h5[attr]
        if key:
            if key not in group:
                return None
            group = group[key]
        Class = cls.__dataclass_fields__[attr].type

        if typing.get_origin(Class) == dict:
            Class = typing.get_args(Class)[1]

        kv = {}
        not_in: list[Field] = []
        size = 0
        for field in fields(Class):
            if field.name not in group:
                not_in.append(field)
                continue

            val = group[field.name][...]
            t = field.type
            has_args = len(typing.get_args(t)) > 0
            if has_args:
                t = typing.get_args(t)[0]
            # ======== str ========
            if has_args and t == str:
                val = [s.decode("utf-8") for s in val]
            elif t == str:
                val = val.decode("utf-8")
            # ======== datetime64 ========
            if t == np.datetime64:
                val = val.astype("datetime64[ms]")
            if has_args:
                size = len(val)
            kv[field.name] = val
        for field in not_in:
            kv[field.name] = np.zeros((size,), dtype=final_type(field.type))

        return Class(**kv)

    def save(self, h5: h5py.File):
        def save_dataclass(group, data):
            for field in fields(data):
                val = getattr(data, field.name)
                if final_type(field.type) == np.datetime64:
                    val = val.astype(np.int64)
                group.create_dataset(field.name, data=val)

        for f in fields(self):
            group = h5.create_group(f.name)
            if typing.get_origin(f.type) == dict:
                for k, v in getattr(self, f.name).items():
                    sub_group = group.create_group(k)
                    save_dataclass(sub_group, v)
            else:
                save_dataclass(group, getattr(self, f.name))

    ## STATIC METHODS
    @staticmethod
    def mask(table: T, mask: np.ndarray[np.bool]) -> T:
        kv = {}
        for f in fields(table):
            val = getattr(table, f.name)
            if isinstance(val, list):
                mask_list = mask.tolist()
                val = [v for v, m in zip(val, mask_list) if m]
            else:
                val = val[mask]
            kv[f.name] = val
        return table.__class__(**kv)

    @staticmethod
    def cast_fields(table: T):
        for f in fields(table):
            if typing.get_args(f.type):
                dtype = typing.get_args(f.type)[0]
                setattr(table, f.name, np.array(getattr(table, f.name), dtype=dtype))

    @staticmethod
    def merge(data_items: list[T], op: Literal["concat", "stack"] = "concat"):
        if len(data_items) == 0:
            return None
        types = {}

        def operation(items: list, t: Type):
            final_t = final_type(t)
            if typing.get_origin(t) == list:
                if op == "concat":
                    res = []
                    for item in items:
                        res += item
                    return res
                else:
                    return items
            else:
                return (
                    np.concatenate(items, dtype=final_t)
                    if op == "concat"
                    else np.array(items, dtype=final_t)
                )

        d = {
            field.name: operation(
                [getattr(d, field.name) for d in data_items], field.type
            )
            for field in fields(data_items[0])
        }

        return data_items[0].__class__(**d)

    @staticmethod
    def cast(data_item: T):
        Class = data_item.__class__
        d = {}
        for f in fields(Class):
            t = final_type(f.type)
            d[f.name] = np.array(getattr(data_item, f.name), dtype=t)
        return Class(**d)

    @staticmethod
    def to_json(dc: T, idxs: Optional[list[int]] = None):
        if idxs is None:
            idxs = range(len(getattr(dc, fields(dc)[0].name)))
        objs = [
            {f.name: getattr(dc, f.name)[i].item() for f in fields(dc)} for i in idxs
        ]
        # convert datetime to string
        for obj in objs:
            for k, v in obj.items():
                if isinstance(v, datetime.date):
                    obj[k] = v.strftime("%Y-%m-%dT%H:%M:%S")
        return objs
