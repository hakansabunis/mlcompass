"""Static deployment-readiness checks.

No model is loaded — we inspect file metadata (size, magic bytes,
format-specific layout cues) and the dependency manifest, then assemble
a structured report. Loading arbitrary model files in a deploy advisor
would be a footgun (pickle exec, allocator surprises, GPU init) and is
explicitly out of scope.

The output dict shape is part of mlcompass's internal contract; the
LLM advisor and the rich renderer key into the same fields.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #


SUPPORTED_TARGETS = ("local", "sagemaker", "lambda", "kubernetes", "vertex")

# ML / DL packages we expect to find pinned in production manifests.
_ML_PACKAGES = frozenset(
    {
        "torch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "keras",
        "jax",
        "scikit-learn",
        "sklearn",
        "xgboost",
        "lightgbm",
        "catboost",
        "numpy",
        "pandas",
        "scipy",
        "onnx",
        "onnxruntime",
        "transformers",
        "tokenizers",
        "safetensors",
    }
)

# Lambda's deploy size ceiling (zipped, ~250 MB unzipped) so we can flag
# anything that obviously won't fit.
_LAMBDA_BYTES_LIMIT = 250 * 1024 * 1024


class DeployAnalysisError(ValueError):
    """Raised when a path can't be analysed at all (missing / unreadable)."""


# --------------------------------------------------------------------------- #
# Public surface                                                              #
# --------------------------------------------------------------------------- #


def assess_deployment(
    model_path: Path | str,
    *,
    requirements_path: Path | str | None = None,
    target: str = "local",
) -> dict[str, Any]:
    """Produce a structured deployment-readiness report.

    Args:
        model_path: Path to the trained model file.
        requirements_path: Optional path to a dependency manifest
            (``requirements.txt`` / ``pyproject.toml`` / ``environment.yml``).
        target: Deployment target name. ``local`` is the default and
            performs only the universal checks.

    Returns:
        Structured dict with keys: ``model``, ``dependencies``, ``target``,
        ``checklist``, ``warnings``.
    """
    model_info = analyze_model_file(model_path)
    deps_info = analyze_dependencies(requirements_path) if requirements_path is not None else None

    if target not in SUPPORTED_TARGETS:
        raise DeployAnalysisError(
            f"Unknown deploy target {target!r}. Supported: {', '.join(SUPPORTED_TARGETS)}"
        )

    target_findings = _target_specific_checks(model_info, deps_info, target=target)
    checklist = _production_checklist(model_info, deps_info, target=target)

    warnings = list(model_info.get("warnings") or [])
    if deps_info is not None:
        warnings.extend(deps_info.get("warnings") or [])
    warnings.extend(target_findings)

    return {
        "model": model_info,
        "dependencies": deps_info,
        "target": target,
        "checklist": checklist,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Model file analysis                                                         #
# --------------------------------------------------------------------------- #


def analyze_model_file(path: Path | str) -> dict[str, Any]:
    """Inspect a model file without loading it.

    Returns:
        Dict with ``path``, ``size_bytes``, ``size_class``, ``format``,
        ``warnings``, ``suggestions``.

    Raises:
        DeployAnalysisError: If the file is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise DeployAnalysisError(f"Model file not found: {path}")

    with path.open("rb") as f:
        header = f.read(64)
    size = int(path.stat().st_size)
    size_class = _size_class(size)
    fmt = _detect_format(header, path.suffix.lower())

    warnings: list[str] = []
    suggestions: list[str] = []

    if fmt == "pickle":
        warnings.append(
            "Model is a raw pickle. Pickle deserialisation can execute "
            "arbitrary code; never load one from an untrusted source."
        )
        suggestions.append(
            "Convert to a safer format: joblib for sklearn, ONNX for "
            "framework-agnostic deploy, or safetensors for transformers."
        )
    elif fmt == "joblib":
        suggestions.append(
            "Joblib is safer than raw pickle but still executes Python on "
            "load. Lock the package versions used at training time."
        )
    elif fmt == "pytorch":
        suggestions.append(
            "Confirm the file is a state_dict (`torch.save(model.state_dict(), …)`) "
            "and not the whole model object — full-model saves embed the class "
            "graph and break on minor refactors."
        )
    elif fmt == "tensorflow_h5":
        suggestions.append(
            "Keras H5 is a legacy format. For new deploys, prefer "
            "`SavedModel` or `TFLite` for edge."
        )
    elif fmt == "onnx":
        suggestions.append(
            "Verify the ONNX opset matches your inference runtime "
            "(`onnx.checker.check_model` + `onnxruntime` smoke test)."
        )
    elif fmt == "safetensors":
        suggestions.append(
            "Safetensors is the recommended format for transformer weights "
            "in production — no risk of arbitrary code execution."
        )
    elif fmt == "unknown":
        warnings.append(
            "Could not identify the model format from the file header or "
            "extension. Verify before shipping."
        )

    if size_class == "huge":
        warnings.append(
            f"Model is {_pretty_size(size)} — make sure your inference "
            "host has enough RAM/VRAM and consider quantising or pruning."
        )

    return {
        "path": str(path),
        "size_bytes": size,
        "size_pretty": _pretty_size(size),
        "size_class": size_class,
        "format": fmt,
        "warnings": warnings,
        "suggestions": suggestions,
    }


def _size_class(size: int) -> str:
    if size < 10 * 1024 * 1024:
        return "small"
    if size < 100 * 1024 * 1024:
        return "medium"
    if size < 1024 * 1024 * 1024:
        return "large"
    return "huge"


def _pretty_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    value: float = float(size)
    for unit in ("KB", "MB", "GB", "TB"):
        value = value / 1024
        if value < 1024:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} PB"  # pragma: no cover - bigger than we'll ever see


def _detect_format(header: bytes, suffix: str) -> str:
    """Detect a model format from magic bytes, falling back to extension."""
    # Magic-byte signatures.
    if header.startswith(b"PK\x03\x04"):
        # PyTorch zip-bundled saves, Hugging Face model archives, etc.
        return "pytorch"
    if header.startswith(b"\x89HDF\r\n\x1a\n"):
        return "tensorflow_h5"
    if header.startswith(b"\x08"):
        # ONNX models are protobuf and almost always start with field 1
        # (`ir_version`, varint). This is a heuristic, not airtight.
        return "onnx"
    if header.startswith(b"\x80"):
        # Pickle opcode HEADER + protocol byte. Joblib files share this
        # magic byte — defer to the file extension when it's present.
        if suffix == ".joblib":
            return "joblib"
        return "pickle"
    # safetensors stores a JSON length header; rely on the extension here
    # because the raw bytes are ambiguous.
    if suffix == ".safetensors" and (
        header[:4].isdigit() or _looks_like_safetensors_header(header)
    ):
        return "safetensors"

    # Fallback by extension.
    by_ext = {
        ".pkl": "pickle",
        ".pickle": "pickle",
        ".joblib": "joblib",
        ".pt": "pytorch",
        ".pth": "pytorch",
        ".ckpt": "pytorch",
        ".bin": "pytorch",
        ".h5": "tensorflow_h5",
        ".keras": "tensorflow_h5",
        ".onnx": "onnx",
        ".safetensors": "safetensors",
        ".gguf": "gguf",
        ".tflite": "tflite",
        ".mlpackage": "coreml",
    }
    return by_ext.get(suffix, "unknown")


def _looks_like_safetensors_header(header: bytes) -> bool:
    """safetensors starts with an 8-byte little-endian header length."""
    if len(header) < 8:
        return False
    return all(0 <= b <= 0xFF for b in header[:8])


# --------------------------------------------------------------------------- #
# Dependency analysis                                                         #
# --------------------------------------------------------------------------- #


def analyze_dependencies(path: Path | str) -> dict[str, Any]:
    """Parse a dependency manifest and report pinning + ML coverage."""
    path = Path(path)
    if not path.is_file():
        raise DeployAnalysisError(f"Dependency file not found: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    name = path.name.lower()

    if name == "requirements.txt" or suffix == ".txt":
        manifest = "requirements.txt"
        entries = _parse_requirements_txt(text)
    elif name == "pyproject.toml" or suffix == ".toml":
        manifest = "pyproject.toml"
        entries = _parse_pyproject_toml(text)
    elif name in ("environment.yml", "environment.yaml") or suffix in (".yml", ".yaml"):
        manifest = "environment.yml"
        entries = _parse_environment_yml(text)
    else:
        raise DeployAnalysisError(
            f"Unsupported dependency manifest: {path}. "
            "Supported: requirements.txt, pyproject.toml, environment.yml."
        )

    pinned: list[str] = []
    unpinned: list[str] = []
    ml_packages: list[str] = []
    for raw in entries:
        normalised = _normalise_package(raw)
        if normalised is None:
            continue
        name_only, has_pin = normalised
        if has_pin:
            pinned.append(raw)
        else:
            unpinned.append(raw)
        if name_only in _ML_PACKAGES:
            ml_packages.append(name_only)

    warnings: list[str] = []
    if unpinned:
        warnings.append(
            f"{len(unpinned)} dependency(ies) are unpinned. Production runs "
            "should pin exact versions to keep the build reproducible."
        )
    if not ml_packages:
        warnings.append(
            "No ML / DL packages detected in the manifest. If your model "
            "needs torch / sklearn / etc. at inference time, add them."
        )

    return {
        "path": str(path),
        "manifest": manifest,
        "count": len(entries),
        "pinned": pinned,
        "unpinned": unpinned,
        "ml_packages": sorted(set(ml_packages)),
        "warnings": warnings,
    }


def _parse_requirements_txt(text: str) -> list[str]:
    entries: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-") or line.startswith("--"):
            continue
        entries.append(line)
    return entries


def _parse_pyproject_toml(text: str) -> list[str]:
    try:
        import tomllib  # type: ignore[import-not-found,unused-ignore]
    except ImportError:  # pragma: no cover - py<3.11
        try:
            import tomli as tomllib  # type: ignore[import-not-found,no-redef,unused-ignore]
        except ImportError:
            return []
    data = tomllib.loads(text)
    deps = data.get("project", {}).get("dependencies", [])
    return [str(d) for d in deps]


def _parse_environment_yml(text: str) -> list[str]:
    data = yaml.safe_load(text) or {}
    deps_field = data.get("dependencies", [])
    out: list[str] = []
    for entry in deps_field:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict):
            # ``pip:`` sub-block.
            for k, v in entry.items():
                if k == "pip" and isinstance(v, list):
                    out.extend(str(x) for x in v)
    return out


_PIN_OPERATORS = re.compile(r"(==|>=|<=|~=|!=|<|>|@|;)")


def _normalise_package(spec: str) -> tuple[str, bool] | None:
    spec = spec.strip()
    if not spec:
        return None
    # ``package[extra]==1.2.3`` etc.
    # Pull out the bare distribution name (left of any operator / extras).
    head = re.split(r"[\[\s=<>!~@;]", spec, maxsplit=1)[0]
    head = head.strip().lower()
    if not head:
        return None
    has_pin = bool(_PIN_OPERATORS.search(spec))
    return head, has_pin


# --------------------------------------------------------------------------- #
# Target-specific + universal checklist                                       #
# --------------------------------------------------------------------------- #


def _target_specific_checks(
    model_info: dict[str, Any],
    deps_info: dict[str, Any] | None,
    *,
    target: str,
) -> list[str]:
    warnings: list[str] = []
    size = model_info.get("size_bytes", 0)

    if target == "lambda":
        if size > _LAMBDA_BYTES_LIMIT:
            warnings.append(
                f"Model is {_pretty_size(size)} but AWS Lambda's unzipped "
                "package limit is ~250 MB. Use Lambda layers, container "
                "images, or move to SageMaker / Vertex."
            )
        if model_info.get("format") in {"pickle", "joblib"}:
            warnings.append(
                "Lambda cold starts pay for the full import graph on every "
                "boot. A pickled sklearn model can take seconds to load — "
                "consider keeping a warm container or moving to ONNX."
            )

    if target == "sagemaker" and model_info.get("format") == "unknown":
        warnings.append(
            "SageMaker built-in containers expect specific formats. "
            "Confirm yours matches the chosen container image."
        )

    if target == "vertex" and model_info.get("format") not in {
        "onnx",
        "tensorflow_h5",
        "pytorch",
    }:
        warnings.append(
            "Vertex AI prediction containers handle TensorFlow, PyTorch, "
            "and ONNX cleanly out of the box. Other formats need a custom "
            "container."
        )

    if target == "kubernetes" and deps_info is not None and deps_info.get("unpinned"):
        warnings.append(
            "Container images that float package versions are a common "
            "source of '500-only-on-Friday' incidents. Pin everything."
        )

    return warnings


def _production_checklist(
    model_info: dict[str, Any],
    deps_info: dict[str, Any] | None,
    *,
    target: str,
) -> list[dict[str, Any]]:
    """Return a list of ``{item, status, detail}`` rows for the rich UI."""
    items: list[dict[str, Any]] = []

    items.append(
        {
            "item": "Model file format identified",
            "status": "ok" if model_info["format"] != "unknown" else "warn",
            "detail": model_info["format"],
        }
    )

    items.append(
        {
            "item": "Model file under 1 GB",
            "status": "ok" if model_info["size_class"] != "huge" else "warn",
            "detail": model_info["size_pretty"],
        }
    )

    if deps_info is not None:
        items.append(
            {
                "item": "Dependency manifest provided",
                "status": "ok",
                "detail": deps_info["manifest"],
            }
        )
        items.append(
            {
                "item": "All dependencies pinned",
                "status": "ok" if not deps_info["unpinned"] else "warn",
                "detail": (
                    f"{len(deps_info['unpinned'])} unpinned"
                    if deps_info["unpinned"]
                    else "all pinned"
                ),
            }
        )
        items.append(
            {
                "item": "ML packages present",
                "status": "ok" if deps_info["ml_packages"] else "warn",
                "detail": ", ".join(deps_info["ml_packages"][:4]) or "none detected",
            }
        )
    else:
        items.append(
            {
                "item": "Dependency manifest provided",
                "status": "warn",
                "detail": "--requirements not set",
            }
        )

    # Universal monitoring items — these never auto-pass; they're nudges.
    for label in (
        "Inference latency measured on representative input",
        "Input-data drift monitoring planned",
        "Model-output drift monitoring planned",
        "Rollback path agreed with stakeholders",
        "Logging + alerting in place",
    ):
        items.append({"item": label, "status": "info", "detail": "manual check"})

    return items
