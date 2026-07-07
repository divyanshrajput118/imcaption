import os
from box.exceptions import BoxValueError
import yaml
from imgCaption import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, Union
import base64
import pickle



@ensure_annotations
def read_yaml(path_to_yaml: Union[Path, str]) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (Union[Path, str]): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(Path(path_to_yaml)) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError("yaml file is empty")
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        verbose (bool, optional): print log message. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(Path(path), exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Union[Path, str], data: dict):
    """save json data

    Args:
        path (Union[Path, str]): path to json file
        data (dict): data to be saved in json file
    """
    with open(Path(path), "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Union[Path, str]) -> ConfigBox:
    """load json files data

    Args:
        path (Union[Path, str]): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(Path(path)) as f:
        content = json.load(f)

    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Union[Path, str]):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Union[Path, str]): path to binary file
    """
    joblib.dump(value=data, filename=Path(path))
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Union[Path, str]) -> Any:
    """load binary data

    Args:
        path (Union[Path, str]): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(Path(path))
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Union[Path, str]) -> str:
    """get size in KB

    Args:
        path (Union[Path, str]): path of the file

    Returns:
        str: size in KB
    """
    size_in_bytes = os.path.getsize(Path(path))
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    size_in_kb = round(size_in_bytes / 1024)
    return f"~ {size_in_kb} KB"


@ensure_annotations
def decodeImage(img_string: str, file_name: Union[Path, str]):
    img_data = base64.b64decode(img_string)
    with open(Path(file_name), 'wb') as f:
        f.write(img_data)


@ensure_annotations
def encodeImageIntoBase64(cropped_image_path: Union[Path, str]) -> str:
    with open(Path(cropped_image_path), "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
def save_pkl_file(obj: Any, file_path: Union[Path, str]):
    with open(Path(file_path), "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Pickle file saved at: {file_path}")


def load_pkl_file(file_path: Union[Path, str]) -> Any:
    with open(Path(file_path), "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Pickle file loaded from: {file_path}")
    return obj
