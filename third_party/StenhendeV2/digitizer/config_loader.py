from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from yacs.config import CfgNode as CN

_C = CN()


def get_cfg(
    config_file: Optional[str] = None,
    new_allowed: bool = True,
) -> CN:
    """Load a yacs config and resolve env vars + sibling YAML paths.

    After merging the YAML, walks the config and:
        1. Expands ``${VAR}`` / ``$VAR`` in any string value.
        2. Resolves any string ending in ``.yml``/``.yaml`` against the
           directory of the loading YAML, if it isn't already absolute.

    Together these let weight paths use ``${STENHEDE_WEIGHTS_PATH}/...``
    and inter-config references use bare filenames.

    Args:
        config_file (Optional[str]): Path to a YAML config to merge.
        new_allowed (bool): Whether yacs allows new keys at merge time.

    Returns:
        CN: A yacs CfgNode with all path strings fully resolved.
    """
    cfg = _C.clone()
    cfg.set_new_allowed(new_allowed)
    if config_file:
        cfg_path = Path(config_file).resolve()
        cfg.merge_from_file(str(cfg_path))
        _resolve_paths(cfg, base_dir=cfg_path.parent)
    return cfg


def merge_cfg(cfg_1: CN, cfg_2: CN) -> CN:
    """Return a copy of ``cfg_1`` merged with ``cfg_2``.

    Args:
        cfg_1 (CN): Base config.
        cfg_2 (CN): Override config.

    Returns:
        CN: Merged config (does not mutate inputs).
    """
    cfg_1 = cfg_1.clone()
    cfg_1.merge_from_other_cfg(cfg_2)
    return cfg_1


def _resolve_paths(cfg: CN, base_dir: Path) -> None:
    """Recursively expand env vars and resolve relative YAML paths in ``cfg``."""
    for k, v in list(cfg.items()):
        if isinstance(v, CN):
            _resolve_paths(v, base_dir)
        elif isinstance(v, str):
            v2 = os.path.expandvars(v)
            if v2.endswith((".yml", ".yaml")) and not Path(v2).is_absolute():
                v2 = str((base_dir / v2).resolve())
            if v2 != v:
                cfg[k] = v2
