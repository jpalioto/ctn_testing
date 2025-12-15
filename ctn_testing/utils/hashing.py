"""Hash utilities for computing fingerprints.

Used for reproducibility verification - if hashes match, inputs are identical.
"""
import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def md5_hash(content: str | bytes) -> str:
    """Compute MD5 hash of content.
    
    Args:
        content: String or bytes to hash
        
    Returns:
        32-character hex digest
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.md5(content).hexdigest()


def hash_dict(d: dict[str, Any]) -> str:
    """Compute MD5 hash of dict via canonical JSON.
    
    Args:
        d: Dictionary to hash
        
    Returns:
        32-character hex digest
    """
    # Sort keys for deterministic ordering
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return md5_hash(canonical)


def hash_config(config: Any) -> str:
    """Compute MD5 hash of a config object.
    
    Works with dataclasses or objects with to_dict().
    
    Args:
        config: Config object (dataclass or has to_dict)
        
    Returns:
        32-character hex digest
    """
    if is_dataclass(config) and not isinstance(config, type):
        d = asdict(config)
    elif hasattr(config, "to_dict"):
        d = config.to_dict()
    elif hasattr(config, "__dict__"):
        d = config.__dict__
    else:
        raise TypeError(f"Cannot hash config of type {type(config)}")
    
    return hash_dict(d)


def hash_file(path: str) -> str:
    """Compute MD5 hash of file contents.
    
    Args:
        path: Path to file
        
    Returns:
        32-character hex digest
    """
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
