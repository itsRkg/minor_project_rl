from omegaconf import OmegaConf
from pathlib import Path

def load_config(config_path: str):
    """
    Load config with inheritance support.

    Args:
        config_path: path to experiment yaml

    Returns:
        OmegaConf DictConfig

    """
    cfg = OmegaConf.load(config_path)

    # Resolve defaults (inheritance)
    if "defaults" in cfg:
        config_dir = Path(config_path).resolve().parent
        base_cfg = OmegaConf.load(config_dir / f"{cfg.defaults[0]}.yaml")
        cfg = OmegaConf.merge(base_cfg, cfg)

    return cfg

