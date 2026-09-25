"""Typed pipeline configuration loaded from YAML.

* Files may ``extends: other.yaml`` (resolved relative to the file); later
  files and ``--set a.b=value`` overrides are deep-merged on top.
* Loading is strict: unknown keys and wrong scalar types raise
  :class:`ConfigError`, so a typo can't silently fall back to a default.
* Paths in ``paths`` are relative to the project root (the ``student_resource``
  directory, which contains ``dataset/``), unless absolute.
"""

from __future__ import annotations

import hashlib
import json
import os
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR.parents[1] / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


class ConfigError(ValueError):
    pass


# ---------------------------------------------------------------- schema
# Stage sections only hold parameters that exist today; tasks add fields as
# they introduce them (and document them in configs/default.yaml).


@dataclass
class PathsConfig:
    root: str = ""  # "" = auto-detect (BER_ROOT env, or nearest dir containing dataset/)
    data_dir: str = "dataset"
    artifacts_dir: str = "artifacts"
    output_dir: str = "output"


@dataclass
class DataConfig:
    slice: str | None = None  # None = full data; "dev" = dev slice (T0.3)


@dataclass
class CandidatesConfig:
    ngram: int = 3
    k_name: int = 10
    k_addr: int = 10
    k_translit: int = 10
    chunk_size: int = 20_000


@dataclass
class Config:
    run_id: str = "main"
    seed: int = 42
    log_level: str = "INFO"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    candidates: CandidatesConfig = field(default_factory=CandidatesConfig)

    # ------------------------------------------------------ derived paths
    @property
    def root(self) -> Path:
        return find_root(self.paths.root)

    def resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.root / path

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.paths.data_dir)

    @property
    def run_dir(self) -> Path:
        return self.resolve(self.paths.artifacts_dir) / self.run_id

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.paths.output_dir)

    # ------------------------------------------------------- serialising
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        """Stable short hash of the full config (excluding run_id)."""
        d = self.to_dict()
        d.pop("run_id")
        blob = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


# ---------------------------------------------------------------- loading


def find_root(explicit: str = "") -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.environ.get("BER_ROOT"):
        return Path(os.environ["BER_ROOT"]).expanduser().resolve()
    for start in (Path.cwd(), PACKAGE_DIR):
        for d in (start, *start.parents):
            if (d / "dataset").is_dir():
                return d
    raise ConfigError("project root not found: set paths.root or BER_ROOT "
                      "(the directory that contains dataset/)")


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in _seen:
        raise ConfigError(f"circular extends: {' -> '.join(map(str, (*_seen, path)))}")
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parent = d.pop("extends", None)
    if parent:
        d = _deep_merge(_load_yaml(path.parent / parent, (*_seen, path)), d)
    return d


def _apply_override(d: dict, expr: str) -> None:
    if "=" not in expr:
        raise ConfigError(f"override must look like key.sub=value, got {expr!r}")
    key, raw = expr.split("=", 1)
    parts = key.strip().split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {expr!r}: {p} is not a section")
    node[parts[-1]] = yaml.safe_load(raw)


def _check_scalar(hint: Any, value: Any, where: str) -> None:
    allowed = typing.get_args(hint) or (hint,)
    if value is None:
        if type(None) not in allowed:
            raise ConfigError(f"{where}: may not be null")
        return
    for t in allowed:
        if t is float and isinstance(value, (int, float)) and not isinstance(value, bool):
            return
        if t in (int, str, bool) and type(value) is t:
            return
        if typing.get_origin(t) in (list, dict) and isinstance(value, typing.get_origin(t)):
            return
        if t in (list, dict) and isinstance(value, t):
            return
    raise ConfigError(f"{where}: expected {hint}, got {type(value).__name__} ({value!r})")


def from_dict(cls: type, d: dict, where: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ConfigError(f"unknown config key(s) {sorted(where + k for k in unknown)}; "
                          f"valid under '{where or '<top>'}': {sorted(names)}")
    kwargs = {}
    for k, v in d.items():
        hint = hints[k]
        if is_dataclass(hint):
            if not isinstance(v, dict):
                raise ConfigError(f"{where}{k}: expected a section")
            kwargs[k] = from_dict(hint, v, f"{where}{k}.")
        else:
            _check_scalar(hint, v, f"{where}{k}")
            kwargs[k] = v
    return cls(**kwargs)


def load_config(paths: list[str | Path] | None = None,
                overrides: list[str] | None = None,
                run_id: str | None = None) -> Config:
    """Merge YAML files (default: configs/default.yaml), then overrides."""
    d: dict = {}
    for p in paths or [DEFAULT_CONFIG]:
        d = _deep_merge(d, _load_yaml(Path(p)))
    for expr in overrides or []:
        _apply_override(d, expr)
    if run_id:
        d["run_id"] = run_id
    return from_dict(Config, d)
