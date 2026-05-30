"""Static analysis of Python training scripts.

Pure ``ast`` — no execution, no LLM calls. Walks the parse tree looking
for the eight most common reproducibility, correctness, and stability
issues we see in ML practitioners' training scripts.

Output is a structured dict consumed by ``ui.audit`` and (later, in
v0.2.1) the optional LLM auditor agent.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# Public types                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class Finding:
    """A single static-analysis hit."""

    rule_id: str
    severity: str  # "error" | "warning" | "info"
    message: str
    suggestion: str
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}

ALL_RULE_IDS: tuple[str, ...] = (
    "seed",
    "val_split",
    "optimizer",
    "loss_stability",
    "dataloader",
    "grad_clipping",
    "eval_mode",
    "batch_size",
)

# Recognised ML framework imports.
_FRAMEWORK_TOP_LEVELS = frozenset(
    {
        "torch",
        "tensorflow",
        "tf",
        "keras",
        "sklearn",
        "numpy",
        "np",
        "random",
        "jax",
        "lightning",
        "pytorch_lightning",
        "pl",
    }
)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def audit_script(
    path: Path | str,
    *,
    skip_rules: set[str] | None = None,
) -> dict[str, Any]:
    """Run all enabled static checks on ``path``.

    Args:
        path: Filesystem path to a Python source file.
        skip_rules: Rule IDs (from :data:`ALL_RULE_IDS`) to skip.

    Returns:
        ::

            {
              "path": str,
              "lines": int,
              "frameworks": list[str],   # detected ML frameworks
              "findings": list[dict],    # Finding.to_dict() entries,
                                          # sorted by severity then line
            }
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    frameworks = _detect_frameworks(tree)

    skip = skip_rules or set()
    checks: list[tuple[str, Callable[[ast.AST, set[str]], list[Finding]]]] = [
        ("seed", _check_seed),
        ("val_split", _check_val_split),
        ("optimizer", _check_optimizer),
        ("loss_stability", _check_loss_stability),
        ("dataloader", _check_dataloader),
        ("grad_clipping", _check_gradient_clipping),
        ("eval_mode", _check_eval_mode),
        ("batch_size", _check_batch_size),
    ]

    findings: list[Finding] = []
    for rule_id, check_fn in checks:
        if rule_id in skip:
            continue
        findings.extend(check_fn(tree, frameworks))

    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 99), f.line or 0))

    return {
        "path": str(path),
        "lines": len(source.splitlines()),
        "frameworks": sorted(frameworks),
        "findings": [f.to_dict() for f in findings],
    }


# --------------------------------------------------------------------------- #
# AST helpers                                                                 #
# --------------------------------------------------------------------------- #


def _attr_chain(node: ast.AST) -> list[str]:
    """Return the dotted-name chain for ``foo.bar.baz`` style expressions."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return _attr_chain(node.value) + [node.attr]
    return []


def _call_matches(node: ast.Call, *qualnames: str) -> bool:
    """True if ``node`` is a call to any of ``qualnames`` (e.g., ``torch.manual_seed``)."""
    chain = _attr_chain(node.func)
    if not chain:
        return False
    dotted = ".".join(chain)
    for q in qualnames:
        if dotted == q:
            return True
        # Also accept ``X.<tail>`` for an alias prefix match.
        if dotted.endswith("." + q):
            return True
    return False


def _has_keyword(call: ast.Call, name: str) -> ast.keyword | None:
    """Return the matching ``ast.keyword`` from ``call.keywords`` or None."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw
    return None


def _detect_frameworks(tree: ast.AST) -> set[str]:
    """Identify ML frameworks imported in the script."""
    frameworks: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _FRAMEWORK_TOP_LEVELS:
                    frameworks.add(top)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0]
            if top in _FRAMEWORK_TOP_LEVELS:
                frameworks.add(top)
    return frameworks


# --------------------------------------------------------------------------- #
# Rule implementations                                                        #
# --------------------------------------------------------------------------- #


_SEED_CALLS = (
    "torch.manual_seed",
    "torch.cuda.manual_seed",
    "torch.cuda.manual_seed_all",
    "numpy.random.seed",
    "np.random.seed",
    "random.seed",
    "tensorflow.random.set_seed",
    "tf.random.set_seed",
    "keras.utils.set_random_seed",
    "pytorch_lightning.seed_everything",
    "lightning.seed_everything",
    "pl.seed_everything",
)


def _check_seed(tree: ast.AST, frameworks: set[str]) -> list[Finding]:
    """Reproducibility — was any random seed actually set?"""
    # If the script imports no stochastic framework, nothing to flag.
    stochastic = {
        "torch",
        "tensorflow",
        "tf",
        "keras",
        "numpy",
        "np",
        "random",
        "jax",
        "lightning",
        "pytorch_lightning",
        "pl",
    }
    if not (frameworks & stochastic):
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_matches(node, *_SEED_CALLS):
            return []  # found a seed call somewhere

    return [
        Finding(
            rule_id="seed",
            severity="error",
            message="No random seed is set anywhere in the script.",
            suggestion=(
                "Add `torch.manual_seed(42)` (and `np.random.seed(42)`, "
                "`random.seed(42)`) near the top so runs are reproducible."
            ),
        )
    ]


def _check_val_split(tree: ast.AST, frameworks: set[str]) -> list[Finding]:
    """Look for validation split; warn if absent or implausibly small."""
    findings: list[Finding] = []
    saw_split = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if _call_matches(
            node,
            "train_test_split",
            "sklearn.model_selection.train_test_split",
            "torch.utils.data.random_split",
            "random_split",
        ):
            saw_split = True
            kw = _has_keyword(node, "test_size") or _has_keyword(node, "val_size")
            if kw is not None and isinstance(kw.value, ast.Constant):
                value = kw.value.value
                if isinstance(value, (int, float)) and 0 < value < 0.05:
                    findings.append(
                        Finding(
                            rule_id="val_split",
                            severity="warning",
                            message=(
                                f"Validation split is very small ({value!r}); "
                                "metrics may be too noisy to compare runs."
                            ),
                            suggestion=(
                                "Use `test_size` of 0.1–0.2 or k-fold CV instead."
                            ),
                            line=node.lineno,
                        )
                    )

    if not saw_split and ("sklearn" in frameworks or "torch" in frameworks):
        findings.append(
            Finding(
                rule_id="val_split",
                severity="warning",
                message="No validation split detected in the script.",
                suggestion=(
                    "Use `train_test_split` or `torch.utils.data.random_split` "
                    "to hold out 10–20% of the data."
                ),
            )
        )

    return findings


_OPTIMIZERS = ("Adam", "AdamW", "SGD", "RMSprop", "Adagrad", "Adadelta", "Adamax", "NAdam")


def _check_optimizer(tree: ast.AST, frameworks: set[str]) -> list[Finding]:
    """Catch common optimizer misconfigurations."""
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _attr_chain(node.func)
        if not chain or chain[-1] not in _OPTIMIZERS:
            continue
        optimizer_name = chain[-1]

        # Bogus kwarg: Adam-family doesn't take ``momentum``
        if optimizer_name in {"Adam", "AdamW", "NAdam", "Adamax"}:
            kw = _has_keyword(node, "momentum")
            if kw is not None:
                findings.append(
                    Finding(
                        rule_id="optimizer",
                        severity="error",
                        message=(
                            f"{optimizer_name} does not accept a `momentum=` "
                            "argument; it will be silently ignored."
                        ),
                        suggestion=(
                            f"Remove `momentum=` from the {optimizer_name} "
                            "constructor, or switch to SGD if you need momentum."
                        ),
                        line=node.lineno,
                    )
                )

        # SGD without momentum is usually wasted compute
        if optimizer_name == "SGD" and _has_keyword(node, "momentum") is None:
            findings.append(
                Finding(
                    rule_id="optimizer",
                    severity="info",
                    message=(
                        "SGD without `momentum=` rarely converges well on "
                        "modern deep nets."
                    ),
                    suggestion="Try `momentum=0.9` or switch to AdamW.",
                    line=node.lineno,
                )
            )

        # Implausible learning rate
        lr_kw = _has_keyword(node, "lr") or _has_keyword(node, "learning_rate")
        if lr_kw is not None and isinstance(lr_kw.value, ast.Constant):
            lr_value = lr_kw.value.value
            if isinstance(lr_value, (int, float)):
                if lr_value > 1.0:
                    findings.append(
                        Finding(
                            rule_id="optimizer",
                            severity="warning",
                            message=f"Learning rate {lr_value} is unusually high.",
                            suggestion="Most adaptive optimizers want 1e-4 – 1e-3.",
                            line=node.lineno,
                        )
                    )
                elif 0 < lr_value < 1e-7:
                    findings.append(
                        Finding(
                            rule_id="optimizer",
                            severity="warning",
                            message=f"Learning rate {lr_value} may be too small to train.",
                            suggestion="Most adaptive optimizers want 1e-4 – 1e-3.",
                            line=node.lineno,
                        )
                    )

    return findings


_LOG_CALLS = (
    "torch.log",
    "torch.log10",
    "torch.log2",
    "np.log",
    "numpy.log",
    "tf.math.log",
)


def _check_loss_stability(tree: ast.AST, frameworks: set[str]) -> list[Finding]:
    """Flag obvious ``log(x)`` without clamping/eps in the same expression."""
    findings: list[Finding] = []
    seen_lines: set[int] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _call_matches(node, *_LOG_CALLS):
            continue
        if node.lineno in seen_lines:
            continue

        if _expr_has_clamp_or_eps(node):
            continue
        seen_lines.add(node.lineno)
        findings.append(
            Finding(
                rule_id="loss_stability",
                severity="warning",
                message=(
                    "Direct `log(x)` call without epsilon clipping; "
                    "if `x` can be 0 you will get `-inf` then NaN."
                ),
                suggestion=(
                    "Wrap as `log(x.clamp(min=1e-7))` or add a small epsilon."
                ),
                line=node.lineno,
            )
        )

    return findings


def _expr_has_clamp_or_eps(node: ast.AST) -> bool:
    """Heuristic: does the call's argument expression mention clamp / eps / 1e-?"""
    src = ast.unparse(node).lower()
    return "clamp" in src or "clip" in src or "eps" in src or "1e-" in src or "+ 1e" in src


def _check_dataloader(tree: ast.AST, frameworks: set[str]) -> list[Finding]:
    """Warn about DataLoader / fit() calls with missing ``shuffle=`` flag."""
    if "torch" not in frameworks:
        return []

    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _attr_chain(node.func)
        if not chain or chain[-1] != "DataLoader":
            continue
        if _has_keyword(node, "shuffle") is None:
            findings.append(
                Finding(
                    rule_id="dataloader",
                    severity="warning",
                    message=(
                        "DataLoader created without an explicit `shuffle=` flag; "
                        "default is `False`, which is usually wrong for training."
                    ),
                    suggestion=(
                        "Pass `shuffle=True` for the training loader and "
                        "`shuffle=False` for validation/test loaders."
                    ),
                    line=node.lineno,
                )
            )

    return findings


_RNN_TRANSFORMER_NAMES = frozenset(
    {
        "LSTM",
        "GRU",
        "RNN",
        "RNNBase",
        "Transformer",
        "TransformerEncoder",
        "TransformerDecoder",
        "TransformerEncoderLayer",
        "TransformerDecoderLayer",
        "MultiheadAttention",
    }
)


def _check_gradient_clipping(tree: ast.AST, frameworks: set[str]) -> list[Finding]:
    """If the model uses RNN/Transformer modules, expect gradient clipping."""
    if "torch" not in frameworks:
        return []

    has_recurrent = False
    has_clipping = False
    rnn_line: int | None = None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain = _attr_chain(node.func)
            if not chain:
                continue
            tail = chain[-1]
            if tail in _RNN_TRANSFORMER_NAMES:
                has_recurrent = True
                if rnn_line is None:
                    rnn_line = node.lineno
            if tail in {"clip_grad_norm_", "clip_grad_value_"}:
                has_clipping = True

    if has_recurrent and not has_clipping:
        return [
            Finding(
                rule_id="grad_clipping",
                severity="warning",
                message=(
                    "Script uses an RNN / Transformer module but never calls "
                    "`clip_grad_norm_` / `clip_grad_value_`."
                ),
                suggestion=(
                    "Add `torch.nn.utils.clip_grad_norm_(model.parameters(), "
                    "max_norm=1.0)` after `loss.backward()`."
                ),
                line=rnn_line,
            )
        ]
    return []


def _check_eval_mode(tree: ast.AST, frameworks: set[str]) -> list[Finding]:
    """Heuristic: torch script that never calls ``.eval()`` likely has a bug."""
    if "torch" not in frameworks:
        return []

    saw_train_call = False
    saw_eval_call = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr == "eval":
                saw_eval_call = True
            elif attr == "train" and not node.args and not node.keywords:
                # ``something.train()`` with no args looks like model.train()
                saw_train_call = True

    if saw_train_call and not saw_eval_call:
        return [
            Finding(
                rule_id="eval_mode",
                severity="warning",
                message=(
                    "Script flips a module into training mode but never "
                    "calls `.eval()`; dropout and batch-norm will misbehave "
                    "at validation/inference time."
                ),
                suggestion="Call `model.eval()` before validation and inference.",
            )
        ]
    return []


def _check_batch_size(tree: ast.AST, frameworks: set[str]) -> list[Finding]:
    """Sanity-check explicit batch_size= kwargs (too tiny / too huge)."""
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kw = _has_keyword(node, "batch_size")
        if kw is None or not isinstance(kw.value, ast.Constant):
            continue
        value = kw.value.value
        if not isinstance(value, int):
            continue
        if value < 4:
            findings.append(
                Finding(
                    rule_id="batch_size",
                    severity="info",
                    message=(
                        f"batch_size={value} is very small; gradient estimates "
                        "will be noisy."
                    ),
                    suggestion=(
                        "Consider batch_size between 16 and 256 unless memory "
                        "is the constraint."
                    ),
                    line=node.lineno,
                )
            )
        elif value > 4096:
            findings.append(
                Finding(
                    rule_id="batch_size",
                    severity="info",
                    message=(
                        f"batch_size={value} is very large; consider gradient "
                        "accumulation if you don't need that much memory parallelism."
                    ),
                    suggestion=(
                        "Try a smaller batch_size with more epochs, or use "
                        "gradient accumulation to keep the effective batch large."
                    ),
                    line=node.lineno,
                )
            )

    return findings
