from dataclasses import dataclass
import os

repo_root = os.path.abspath(os.path.join(__file__, "../../"))

# Override with FIRECOMP_DATA to keep the dataset outside the repo checkout.
data_dir = os.environ.get("FIRECOMP_DATA", os.path.join(repo_root, "data"))


@dataclass
class Config:
    data_dir: str = data_dir
    vnp14_fires_dir: str = os.path.join(data_dir, "vnp14_fires")
    vnp14_dir: str = os.path.join(data_dir, "vnp14")
    vnp03img_dir: str = os.path.join(data_dir, "vnp03img")
    runs_dir: str = os.path.join(data_dir, "runs")

    @property
    def vnp14_path(self):
        return os.path.join(self.vnp14_fires_dir, "vnp14_fires_2012.h5")


config = Config()

FLOAT32_NODATA = -32768.0


def init_config(**kwargs):
    global config
    for key, value in kwargs.items():
        setattr(config, key, value)
