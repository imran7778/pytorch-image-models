"""Select machine-specific paths without rewriting shared YAML files."""

from __future__ import annotations

import re
from typing import Any


ENVIRONMENT_CHOICES = ("local", "server")
INTERPOLATION = re.compile(r"\$\{([^}]+)}")


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
    return resolve_tree(resolved, resolved)


def resolve_tree(value: Any, root: dict[str, Any]) -> Any:
    """Resolve ${dotted.key} references throughout a configuration tree."""
    if isinstance(value, dict):
        return {key: resolve_tree(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_tree(item, root) for item in value]
    if not isinstance(value, str):
        return value
    exact = INTERPOLATION.fullmatch(value)
    if exact:
        return resolve_tree(lookup(root, exact.group(1)), root)
    result = value
    for _ in range(10):
        updated = INTERPOLATION.sub(lambda match: str(lookup(root, match.group(1))), result)
        if updated == result:
            return updated
        result = updated
    raise ValueError(f"Interpolation did not converge: {value}")


def lookup(root: dict[str, Any], dotted_key: str) -> Any:
    value: Any = root
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Unknown interpolation ${{{dotted_key}}}")
        value = value[part]
    return value
