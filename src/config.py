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

    ROOT_DATA_PATH: Path = Path(ROOT_PROJECT, 'data')
    ROOT_OUPUT: Path = Path(ROOT_PROJECT, 'output')

    model_config = SettingsConfigDict(env_file='.env',
                                      env_file_encoding='utf-8')
