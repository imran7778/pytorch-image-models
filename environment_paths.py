"""Select machine-specific paths without rewriting shared YAML files."""

from __future__ import annotations

from typing import Any


ENVIRONMENT_CHOICES = ("local", "server")


def environment_name(requested: str | None, local_test: bool = False) -> str:
    if local_test and requested not in (None, "local"):
        raise ValueError("--local-test cannot be combined with --environment server")
    return "local" if local_test else (requested or "server")


def apply_environment(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    environments = cfg.get("environments")
    if not isinstance(environments, dict) or name not in environments:
        raise KeyError(f"Missing environments.{name} configuration")
    selected = environments[name]
    required = ("project_root", "export_root", "source_dataset", "prepared_dataset")
    missing = [key for key in required if not selected.get(key)]
    if missing:
        raise KeyError(f"Missing environments.{name} paths: {', '.join(missing)}")
    resolved = dict(cfg)
    resolved["active_environment"] = name
    resolved["environment_paths"] = dict(selected)
    # Preserve existing ${project_root}/${export_root} interpolation references.
    resolved["project_root"] = selected["project_root"]
    resolved["export_root"] = selected["export_root"]
    return resolved
