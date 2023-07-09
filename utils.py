import logging
from pathlib import Path
from pydantic import BaseModel


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class AppPath:
    ROOT_DIR = Path('.')
    DATA_DIR = ROOT_DIR / 'data'
    # store raw data
    RAW_DATA_DIR = DATA_DIR / 'raw_data'
    # store processed data
    TRAIN_DATA_DIR = DATA_DIR / 'train_data'
    # store configs
    MODEL_CONFIG_DIR = ROOT_DIR / 'config'
    # store captured data
    CAPTURED_DATA_DIR = DATA_DIR / 'captured_data'
    # store models
    MODEL_DIR = ROOT_DIR / 'models'

AppPath.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
AppPath.TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
AppPath.MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
AppPath.CAPTURED_DATA_DIR.mkdir(parents=True, exist_ok=True)
AppPath.MODEL_DIR.mkdir(parents=True, exist_ok=True)


class Data(BaseModel):
    id: str
    rows: list
    columns: list
