from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from torch import nn
from yacs.config import CfgNode as CN


def find_config_path(filename: str) -> str:
    """Resolve a config filename to an absolute path.

    Searches first the current working directory, then the bundled
    ``configs/`` directory next to this module.

    Args:
        filename (str): Config file name or path.

    Returns:
        str: Absolute path to the config file.
    """
    if Path(filename).is_file():
        return str(Path(filename).resolve())
    bundled = Path(__file__).parent / "configs" / filename
    if bundled.is_file():
        return str(bundled.resolve())
    raise FileNotFoundError(f"Could not find config file: {filename}")


def import_class_from_path(path: str) -> Any:
    """Import a class given its dotted module path.

    Args:
        path (str): Dotted path like ``src.StenhendeV2.digitizer.model.unet.UNet``.

    Returns:
        Any: The imported class object.
    """
    module = importlib.import_module(".".join(path.split(".")[:-1]))
    return getattr(module, path.split(".")[-1])


def load_model(config: CN) -> nn.Module:
    """Instantiate a model from a config node.

    Args:
        config (CN): Config with a ``MODEL.class_path`` and ``MODEL.KWARGS``.

    Returns:
        nn.Module: The instantiated model.
    """
    model_class = import_class_from_path(config.MODEL.class_path)
    return model_class(**config.MODEL.KWARGS)  # type: ignore
