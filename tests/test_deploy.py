"""Tests for the deterministic deployment-analysis tool layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlcompass.tools.deploy import (
    DeployAnalysisError,
    SUPPORTED_TARGETS,
    analyze_dependencies,
    analyze_model_file,
    assess_deployment,
)


# --------------------------------------------------------------------------- #
# Magic-byte fixtures                                                         #
# --------------------------------------------------------------------------- #


def _write_bytes(parent: Path, name: str, payload: bytes) -> Path:
    p = parent / name
    p.write_bytes(payload)
    return p


def _pytorch_zip(tmp_path: Path) -> Path:
    # PyTorch's torch.save bundles a Zip with PK\x03\x04 header.
    return _write_bytes(tmp_path, "model.pt", b"PK\x03\x04" + b"\x00" * 60)


def _pickle_file(tmp_path: Path) -> Path:
    # Pickle protocol marker: \x80\x04
    return _write_bytes(tmp_path, "model.pkl", b"\x80\x04" + b"\x00" * 60)


def _joblib_file(tmp_path: Path) -> Path:
    return _write_bytes(tmp_path, "model.joblib", b"\x80\x04" + b"\x00" * 60)


def _h5_file(tmp_path: Path) -> Path:
    # HDF5 signature.
    return _write_bytes(
        tmp_path, "model.h5", b"\x89HDF\r\n\x1a\n" + b"\x00" * 56
    )


def _onnx_file(tmp_path: Path) -> Path:
    # ONNX (protobuf) — varint field tag for ir_version.
    return _write_bytes(tmp_path, "model.onnx", b"\x08\x07" + b"\x00" * 60)


def _safetensors_file(tmp_path: Path) -> Path:
    return _write_bytes(
        tmp_path, "model.safetensors", b"\x10\x00\x00\x00" + b"\x00" * 60
    )


# --------------------------------------------------------------------------- #
# analyze_model_file — format detection                                       #
# --------------------------------------------------------------------------- #


def test_analyze_model_file_detects_pytorch(tmp_path: Path) -> None:
    info = analyze_model_file(_pytorch_zip(tmp_path))
    assert info["format"] == "pytorch"
    assert info["size_class"] == "small"
    assert info["suggestions"]


def test_analyze_model_file_detects_pickle(tmp_path: Path) -> None:
    info = analyze_model_file(_pickle_file(tmp_path))
    assert info["format"] == "pickle"
    # Pickle should trigger a security warning.
    assert any("pickle" in w.lower() for w in info["warnings"])


def test_analyze_model_file_detects_joblib_by_extension(tmp_path: Path) -> None:
    info = analyze_model_file(_joblib_file(tmp_path))
    assert info["format"] == "joblib"


def test_analyze_model_file_detects_h5(tmp_path: Path) -> None:
    info = analyze_model_file(_h5_file(tmp_path))
    assert info["format"] == "tensorflow_h5"


def test_analyze_model_file_detects_onnx(tmp_path: Path) -> None:
    info = analyze_model_file(_onnx_file(tmp_path))
    assert info["format"] == "onnx"


def test_analyze_model_file_detects_safetensors(tmp_path: Path) -> None:
    info = analyze_model_file(_safetensors_file(tmp_path))
    assert info["format"] == "safetensors"


def test_analyze_model_file_unknown_format_warns(tmp_path: Path) -> None:
    p = tmp_path / "model.weird"
    p.write_bytes(b"random content")
    info = analyze_model_file(p)
    assert info["format"] == "unknown"
    assert any("identify" in w.lower() for w in info["warnings"])


def test_analyze_model_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(DeployAnalysisError):
        analyze_model_file(tmp_path / "nope.pt")


def test_analyze_model_file_size_class_thresholds(tmp_path: Path) -> None:
    p = tmp_path / "big.pt"
    # 200 MB → "large".
    with p.open("wb") as f:
        f.write(b"PK\x03\x04")
        f.seek(200 * 1024 * 1024 - 1)
        f.write(b"\x00")
    info = analyze_model_file(p)
    assert info["size_class"] == "large"


# --------------------------------------------------------------------------- #
# analyze_dependencies                                                        #
# --------------------------------------------------------------------------- #


def test_analyze_dependencies_pins_and_unpinned(tmp_path: Path) -> None:
    p = tmp_path / "requirements.txt"
    p.write_text(
        "torch==2.1.0\nscikit-learn==1.3.0\nrequests\n# comment\nnumpy>=1.21\n",
        encoding="utf-8",
    )
    info = analyze_dependencies(p)
    assert info["manifest"] == "requirements.txt"
    assert info["count"] == 4
    assert "requests" in [u.lower() for u in info["unpinned"]]
    assert any("unpinned" in w.lower() for w in info["warnings"])
    assert "torch" in info["ml_packages"]


def test_analyze_dependencies_pyproject(tmp_path: Path) -> None:
    p = tmp_path / "pyproject.toml"
    p.write_text(
        '[project]\n'
        'name = "demo"\n'
        'version = "0.1.0"\n'
        'dependencies = [\n'
        '    "torch>=2.0,<3.0",\n'
        '    "pandas>=2.0",\n'
        ']\n',
        encoding="utf-8",
    )
    info = analyze_dependencies(p)
    assert info["manifest"] == "pyproject.toml"
    assert info["count"] == 2
    assert info["pinned"]  # both have version specifiers


def test_analyze_dependencies_environment_yml(tmp_path: Path) -> None:
    p = tmp_path / "environment.yml"
    p.write_text(
        "name: demo\n"
        "dependencies:\n"
        "  - python=3.11\n"
        "  - pip\n"
        "  - pip:\n"
        "      - torch==2.1.0\n",
        encoding="utf-8",
    )
    info = analyze_dependencies(p)
    assert info["manifest"] == "environment.yml"
    assert "torch" in info["ml_packages"]


def test_analyze_dependencies_warns_when_no_ml_packages(tmp_path: Path) -> None:
    p = tmp_path / "requirements.txt"
    p.write_text("flask==3.0.0\nrequests==2.31.0\n", encoding="utf-8")
    info = analyze_dependencies(p)
    assert info["ml_packages"] == []
    assert any("ml" in w.lower() for w in info["warnings"])


def test_analyze_dependencies_unsupported_manifest_raises(tmp_path: Path) -> None:
    p = tmp_path / "Pipfile"
    p.write_text("anything", encoding="utf-8")
    with pytest.raises(DeployAnalysisError, match="Unsupported"):
        analyze_dependencies(p)


# --------------------------------------------------------------------------- #
# assess_deployment                                                           #
# --------------------------------------------------------------------------- #


def test_assess_deployment_returns_top_level_keys(tmp_path: Path) -> None:
    model = _pytorch_zip(tmp_path)
    result = assess_deployment(model, target="local")
    for key in ("model", "dependencies", "target", "checklist", "warnings"):
        assert key in result
    assert result["dependencies"] is None  # not supplied


def test_assess_deployment_with_requirements(tmp_path: Path) -> None:
    model = _pytorch_zip(tmp_path)
    req = tmp_path / "requirements.txt"
    req.write_text("torch==2.1.0\nnumpy==1.26.0\n", encoding="utf-8")
    result = assess_deployment(model, requirements_path=req, target="local")
    assert result["dependencies"]["ml_packages"] == ["numpy", "torch"]


def test_assess_deployment_unknown_target_raises(tmp_path: Path) -> None:
    model = _pytorch_zip(tmp_path)
    with pytest.raises(DeployAnalysisError, match="target"):
        assess_deployment(model, target="mars")


def test_assess_deployment_lambda_size_warning(tmp_path: Path) -> None:
    p = tmp_path / "huge.pt"
    with p.open("wb") as f:
        f.write(b"PK\x03\x04")
        f.seek(300 * 1024 * 1024 - 1)
        f.write(b"\x00")
    result = assess_deployment(p, target="lambda")
    assert any("lambda" in w.lower() for w in result["warnings"])


def test_assess_deployment_lambda_pickle_warning(tmp_path: Path) -> None:
    model = _pickle_file(tmp_path)
    result = assess_deployment(model, target="lambda")
    assert any("cold start" in w.lower() or "pickled" in w.lower() for w in result["warnings"])


def test_assess_deployment_checklist_marks_unknown_format(tmp_path: Path) -> None:
    p = tmp_path / "model.weird"
    p.write_bytes(b"random")
    result = assess_deployment(p, target="local")
    items = {row["item"]: row["status"] for row in result["checklist"]}
    assert items["Model file format identified"] == "warn"


def test_assess_deployment_checklist_includes_monitoring_nudges(tmp_path: Path) -> None:
    model = _pytorch_zip(tmp_path)
    result = assess_deployment(model, target="local")
    items = [row["item"] for row in result["checklist"]]
    assert any("latency" in item.lower() for item in items)
    assert any("drift" in item.lower() for item in items)
    assert any("rollback" in item.lower() for item in items)


def test_assess_deployment_supports_all_documented_targets(tmp_path: Path) -> None:
    model = _pytorch_zip(tmp_path)
    for target in SUPPORTED_TARGETS:
        result = assess_deployment(model, target=target)
        assert result["target"] == target
