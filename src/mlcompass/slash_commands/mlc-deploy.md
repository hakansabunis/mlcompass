---
description: Deployment-readiness check on a saved model file (lambda / sagemaker / k8s / vertex)
---

Use the `mlcompass_deploy` tool on the model file at: $ARGUMENTS

If the user mentioned a deployment target (lambda, sagemaker, kubernetes, vertex), pass it as `target=<name>`. Default to `target="local"` if unspecified.

If a `requirements.txt`, `pyproject.toml`, or `environment.yml` is in the same directory as the model, pass it as `requirements_path=<that file>`.

After the tool returns, in the user's language:

1. Model file: detected format (pytorch, sklearn-pickle, joblib, h5, onnx, safetensors), size, size class.
2. Dependency manifest: number of pinned vs unpinned packages, ML-relevant packages found, any warnings.
3. Target-specific checks (e.g. Lambda's 250 MB ceiling, SageMaker pickle warning).
4. The "production checklist" — show all items, color-code by status.

End with the top blocker (if any). If everything is green, say "ready to ship".
