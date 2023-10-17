import os
from pathlib import Path
from typing import Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic.dataclasses import dataclass


@dataclass
class DataPath:
    """
    Basic directory structure can be specified here.
    """
    data: Path = Path("data")

    raw: Path = data / "raw"
    interim: Path = data / "interim"
    processed: Path = data / "processed"


class Base(BaseSettings):

    ROOT_PROJECT: str = str((Path(__file__).parent.resolve()
                             if ("__file__" in locals())
                             else Path(os.getcwd()).parent))

    ROOT_DATA_PATH: Path = Path(ROOT_PROJECT, DataPath.data)
    DATA_PATH: DataPath = DataPath()

    RAW_PATH: Path = ROOT_PROJECT / DATA_PATH.raw
    INTERIM_PATH: Path = ROOT_PROJECT / DATA_PATH.interim
    PROCESSED_PATH: Path = ROOT_PROJECT / DATA_PATH.processed

    CTORBHSV_PATH: Path = RAW_PATH / "features_ctorbhsv"
    DL_PATH: Path = RAW_PATH / "features_dl_230621"

    MODEL_PATH: Path = Path(ROOT_PROJECT) / Path("models")

    model_config = SettingsConfigDict(env_file='.env',
                                      env_file_encoding='utf-8')
