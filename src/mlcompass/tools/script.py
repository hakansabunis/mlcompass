"""Static analysis of Python training scripts.

Pure ``ast`` — no execution, no LLM calls. Walks the parse tree looking
for the most common reproducibility, correctness, and stability issues
we see in ML practitioners' training scripts.

Two families of rule live here. The first eight are deep-learning
shaped and mostly torch-gated. The six after them cover tabular
``pandas + scikit-learn`` training scripts, which are the most common
kind of ML script there is and which the first eight are structurally
unable to say anything about — the ``seed`` rule returns early unless a
stochastic framework is imported, and scikit-learn is not one.

Every rule here is written to be quiet on correct code. Where a shape
cannot be decided from an AST, the rule stays silent: a false positive
costs more than a miss, because it teaches users to stop reading the
output.

Output is a structured dict consumed by ``ui.audit`` and (later, in
v0.2.1) the optional LLM auditor agent.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    # Deep-learning shaped.
    "seed",
    "val_split",
    "optimizer",
    "loss_stability",
    "dataloader",
    "grad_clipping",
    "eval_mode",
    "batch_size",
    # Tabular pandas + scikit-learn shaped.
    "preprocess_leak",
    "refit_across_split",
    "target_leak",
    "random_state",
    "unused_holdout",
    "metric_choice",
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
        ("preprocess_leak", _check_preprocess_leak),
        ("refit_across_split", _check_refit_across_split),
        ("target_leak", _check_target_leak),
        ("random_state", _check_random_state),
        ("unused_holdout", _check_unused_holdout),
        ("metric_choice", _check_metric_choice),
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
                            suggestion=("Use `test_size` of 0.1–0.2 or k-fold CV instead."),
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
                    message=("SGD without `momentum=` rarely converges well on modern deep nets."),
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
                suggestion=("Wrap as `log(x.clamp(min=1e-7))` or add a small epsilon."),
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
                        f"batch_size={value} is very small; gradient estimates will be noisy."
                    ),
                    suggestion=(
                        "Consider batch_size between 16 and 256 unless memory is the constraint."
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


# --------------------------------------------------------------------------- #
# Tabular pandas + scikit-learn: shared analysis                              #
# --------------------------------------------------------------------------- #
#
# The rules below are gated by the scikit-learn call names they match, not by
# an import whitelist. That is deliberate: the ``seed`` rule's whitelist is
# exactly why it cannot fire on a scikit-learn script.

_SCOPE_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
)


def _scope_local_nodes(tree: ast.AST) -> list[list[ast.AST]]:
    """Group nodes by enclosing scope, without descending into nested scopes.

    Within one scope, source order is execution order for the straight-line
    statements these rules care about. Across a scope boundary it is not — a
    helper defined above a call site still runs after it — so the ordering
    rules never compare line numbers across this boundary.
    """
    scopes: list[list[ast.AST]] = []

    def collect(scope: ast.AST) -> None:
        local: list[ast.AST] = []
        stack: list[ast.AST] = list(ast.iter_child_nodes(scope))
        while stack:
            node = stack.pop()
            if isinstance(node, _SCOPE_NODES):
                collect(node)
                continue
            local.append(node)
            stack.extend(ast.iter_child_nodes(node))
        scopes.append(local)

    collect(tree)
    return scopes


def _assignments_in_source_order(tree: ast.AST) -> list[ast.Assign]:
    """Every ``ast.Assign`` in the tree, ordered by position in the source."""
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
    assigns.sort(key=lambda n: (n.lineno, n.col_offset))
    return assigns


def _constructor_names(tree: ast.AST) -> dict[str, str]:
    """Map ``name -> constructor class`` for every ``name = SomeClass(...)``."""
    out: dict[str, str] = {}
    for node in _assignments_in_source_order(tree):
        if not isinstance(node.value, ast.Call):
            continue
        chain = _attr_chain(node.value.func)
        if not chain:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out[target.id] = chain[-1]
    return out


_SPLIT_CALL = ("train_test_split", "sklearn.model_selection.train_test_split")


def _split_assignments(tree: ast.AST) -> list[tuple[ast.Call, list[ast.expr]]]:
    """``(call, targets)`` for every ``a, b, ... = train_test_split(...)``."""
    out: list[tuple[ast.Call, list[ast.expr]]] = []
    for node in _assignments_in_source_order(tree):
        if not isinstance(node.value, ast.Call):
            continue
        if not _call_matches(node.value, *_SPLIT_CALL):
            continue
        for target in node.targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                out.append((node.value, list(target.elts)))
    return out


# --------------------------------------------------------------------------- #
# Rule: preprocess_leak                                                       #
# --------------------------------------------------------------------------- #

# Transformers that learn a statistic from every row they are fitted on.
# Fitting one of these on the full frame and splitting afterwards puts test-set
# information into the training features.
#
# Deliberately absent, each for a reason:
#   Normalizer            — row-wise, learns nothing from other rows
#   FunctionTransformer   — stateless unless given a fitted func
#   LabelEncoder /
#   LabelBinarizer        — a label vocabulary for the target, not a statistic
#   OneHotEncoder /
#   OrdinalEncoder        — learns a category vocabulary; pre-split fitting is
#                           near-universal practice and the accepted fix is
#                           handle_unknown=, so flagging it would be noise
#   Pipeline / make_pipeline — this is the recommended fix, not the defect
_LEAKY_TRANSFORMERS = frozenset(
    {
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler",
        "MaxAbsScaler",
        "QuantileTransformer",
        "PowerTransformer",
        "SimpleImputer",
        "KNNImputer",
        "IterativeImputer",
        "SelectKBest",
        "SelectPercentile",
        "SelectFromModel",
        "VarianceThreshold",
        "RFE",
        "RFECV",
        "TargetEncoder",
        "ColumnTransformer",
    }
)

_FIT_ATTRS = frozenset({"fit", "fit_transform"})

# Calls after which the data is considered split — anything fitted before one
# of these was fitted on rows that end up on both sides.
_SPLIT_BOUNDARY_CALLS = (
    "train_test_split",
    "cross_val_score",
    "cross_validate",
)


def _fitted_transformer(call: ast.Call, ctors: dict[str, str]) -> str | None:
    """Class name if ``call`` is ``<stateful transformer>.fit()/.fit_transform()``."""
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr not in _FIT_ATTRS:
        return None

    receiver = func.value
    cls: str | None = None
    if isinstance(receiver, ast.Call):
        # StandardScaler().fit_transform(X)
        chain = _attr_chain(receiver.func)
        cls = chain[-1] if chain else None
    elif isinstance(receiver, ast.Name):
        # scaler = StandardScaler() ... scaler.fit_transform(X)
        cls = ctors.get(receiver.id)

    return cls if cls in _LEAKY_TRANSFORMERS else None


def _check_preprocess_leak(tree: ast.AST, frameworks: set[str]) -> list[Finding]:
    """Preprocessing fitted on the whole dataset before it is split."""
    findings: list[Finding] = []
    ctors = _constructor_names(tree)

    for local in _scope_local_nodes(tree):
        calls = sorted(
            (n for n in local if isinstance(n, ast.Call)),
            key=lambda c: (c.lineno, c.col_offset),
        )
        boundary = next((c for c in calls if _call_matches(c, *_SPLIT_BOUNDARY_CALLS)), None)
        if boundary is None:
            continue

        boundary_chain = _attr_chain(boundary.func)
        boundary_name = boundary_chain[-1] if boundary_chain else "the split"
        # A fit nested inside the split call's own arguments also runs first.
        nested = {id(n) for n in ast.walk(boundary)} - {id(boundary)}

        seen: set[int] = set()
        for call in calls:
            runs_first = call.lineno < boundary.lineno or id(call) in nested
            if not runs_first or call.lineno in seen:
                continue
            cls = _fitted_transformer(call, ctors)
            if cls is None:
                continue
            seen.add(call.lineno)
            findings.append(
                Finding(
                    rule_id="preprocess_leak",
                    severity="error",
                    message=(
                        f"`{cls}` is fitted on the full dataset before "
                        f"`{boundary_name}` on line {boundary.lineno}; the statistics "
                        "it learns include the held-out rows, which inflates every "
                        "metric measured afterwards."
                    ),
                    suggestion=(
                        "Split first, then fit on the training half only "
                        f"(`{cls}().fit_transform(X_train)` / `.transform(X_test)`), or "
                        f"put `{cls}` inside a `Pipeline` so cross-validation refits it "
                        "per fold."
                    ),
                    line=call.lineno,
                )
            )

    return findings


# --------------------------------------------------------------------------- #
# Rule: refit_across_split                                                    #
# --------------------------------------------------------------------------- #


def _check_refit_across_split(tree: ast.AST, frameworks: set[str]) -> list[Finding]:
    """One object fitted on *both* halves of the same split.

    ``scaler.fit_transform(X_train)`` followed by
    ``scaler.fit_transform(X_test)`` is the canonical form, and it is the
    ``transform``-was-meant typo: the second fit throws away the statistics
    learned from the first, so the two halves end up scaled by different
    constants and the model is scored on a scale it never saw.

    Two shapes are deliberately *not* flagged, because both are correct:

    * Fitting a **different** object on the second half. That is the ordinary
      train / calibrate / test workflow, where a calibrator is supposed to be
      fitted on the middle split -- scikit-learn's own
      ``test_calibration.py`` does exactly this.
    * Fitting on the second output and scoring on the first. That is reversed
      naming, not leakage: the two sets are still disjoint.

    ``train_test_split`` returns ``2 * len(arrays)`` values interleaved
    train, test, train, test…, so even indices are training halves and odd
    indices are held out, whatever the script chose to call them. Names are
    resolved per scope -- a ``test_data`` local in one function must not be
    matched against a split in another.
    """
    findings: list[Finding] = []

    for local in _scope_local_nodes(tree):
        assigns: list[tuple[ast.Assign, ast.Call]] = []
        for candidate in local:
            if isinstance(candidate, ast.Assign) and isinstance(candidate.value, ast.Call):
                assigns.append((candidate, candidate.value))
        assigns.sort(key=lambda pair: (pair[0].lineno, pair[0].col_offset))

        halves: dict[str, int] = {}
        for node, split_call in assigns:
            if not _call_matches(split_call, *_SPLIT_CALL):
                continue
            for target in node.targets:
                if not isinstance(target, (ast.Tuple, ast.List)):
                    continue
                for index, element in enumerate(target.elts):
                    if isinstance(element, ast.Name) and not element.id.startswith("_"):
                        halves[element.id] = index % 2

        if not halves:
            continue

        # receiver -> half -> (line, attr, argument name), first fit of each half
        fits: dict[str, dict[int, tuple[int, str, str]]] = {}
        calls = sorted(
            (n for n in local if isinstance(n, ast.Call)),
            key=lambda c: (c.lineno, c.col_offset),
        )
        for call in calls:
            func = call.func
            if not isinstance(func, ast.Attribute) or func.attr not in _FIT_ATTRS:
                continue
            # A bare name is the only receiver whose identity we can track.
            if not isinstance(func.value, ast.Name):
                continue
            if not call.args or not isinstance(call.args[0], ast.Name):
                continue
            half = halves.get(call.args[0].id)
            if half is None:
                continue
            fits.setdefault(func.value.id, {}).setdefault(
                half, (call.lineno, func.attr, call.args[0].id)
            )

        for receiver, by_half in sorted(fits.items()):
            if 0 not in by_half or 1 not in by_half:
                continue
            train_line, _, train_arg = by_half[0]
            hold_line, hold_attr, hold_arg = by_half[1]
            findings.append(
                Finding(
                    rule_id="refit_across_split",
                    severity="error",
                    message=(
                        f"`{receiver}` is fitted on `{train_arg}` (line {train_line}) and "
                        f"fitted again on `{hold_arg}` (line {hold_line}) — the two halves "
                        "of one `train_test_split`. The second fit discards what the first "
                        "learned, so the halves are transformed by different constants."
                    ),
                    suggestion=(
                        f"Fit once on the training half, then call "
                        f"`{receiver}.transform({hold_arg})` instead of "
                        f"`{receiver}.{hold_attr}({hold_arg})`."
                    ),
                    line=hold_line,
                )
            )

    return findings


# --------------------------------------------------------------------------- #
# Rule: target_leak                                                           #
# --------------------------------------------------------------------------- #


def _bare_frame_alias(value: ast.expr) -> str | None:
    """Frame name if ``value`` is that whole frame, un-narrowed."""
    if isinstance(value, ast.Name):
        return value.id
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr in {"copy", "to_numpy"}
        and isinstance(value.func.value, ast.Name)
    ):
        return value.func.value.id
    if (
        isinstance(value, ast.Attribute)
        and value.attr == "values"
        and isinstance(value.value, ast.Name)
    ):
        return value.value.id
    return None


def _string_column_of(value: ast.expr) -> tuple[str, str] | None:
    """``(frame, column)`` if ``value`` is ``frame["column"]``."""
    if (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Name)
        and isinstance(value.slice, ast.Constant)
        and isinstance(value.slice.value, str)
    ):
        return value.value.id, value.slice.value
    return None


def _check_target_leak(tree: ast.AST, frameworks: set[str]) -> list[Finding]:
    """The feature matrix is the whole frame the target column came from."""
    # Any frame that is dropped from or popped is out of scope: we cannot tell
    # which columns survive, so we say nothing.
    narrowed: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"drop", "pop"}
            and isinstance(node.func.value, ast.Name)
        ):
            narrowed.add(node.func.value.id)

    alias_of: dict[str, str] = {}
    column_of: dict[str, tuple[str, str]] = {}
    for node in _assignments_in_source_order(tree):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id

        frame = _bare_frame_alias(node.value)
        if frame is not None:
            alias_of[name] = frame
        else:
            # Re-bound to something narrower; the alias no longer holds.
            alias_of.pop(name, None)

        column = _string_column_of(node.value)
        if column is not None:
            column_of[name] = column
        elif frame is None:
            column_of.pop(name, None)

    if not alias_of or not column_of:
        return []

    findings: list[Finding] = []
    seen: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        is_split = _call_matches(node, *_SPLIT_CALL)
        is_fit = isinstance(node.func, ast.Attribute) and node.func.attr == "fit"
        if not (is_split or is_fit):
            continue

        features, target = node.args[0], node.args[1]
        if not isinstance(features, ast.Name) or not isinstance(target, ast.Name):
            continue

        frame = alias_of.get(features.id)
        column = column_of.get(target.id)
        if frame is None or column is None or column[0] != frame:
            continue
        if frame in narrowed or features.id in narrowed:
            continue
        if node.lineno in seen:
            continue

        seen.add(node.lineno)
        findings.append(
            Finding(
                rule_id="target_leak",
                severity="error",
                message=(
                    f"`{features.id}` is the whole of `{frame}`, and the target "
                    f"`{target.id}` is `{frame}['{column[1]}']` — so the target column "
                    "is still one of the features and the model can read the answer."
                ),
                suggestion=(
                    f"Build the feature matrix as "
                    f"`{features.id} = {frame}.drop(columns=['{column[1]}'])`, or take "
                    f"the target with `{target.id} = {frame}.pop('{column[1]}')` before "
                    f"assigning `{features.id}`."
                ),
                line=node.lineno,
            )
        )

    return findings


# --------------------------------------------------------------------------- #
# Rule: random_state                                                          #
# --------------------------------------------------------------------------- #

# scikit-learn classes that are stochastic with their *default* parameters.
# Anything only stochastic under a non-default argument is left out, because
# firing on `LogisticRegression()` (deterministic under the default lbfgs
# solver) is exactly the kind of noise that teaches users to ignore the tool.
# Left out for that reason: LogisticRegression, SVC, LinearSVC, Ridge, Lasso,
# ElasticNet, KNeighbors*, GaussianNB, and PCA (svd_solver='auto' only picks
# the randomized solver for large inputs).
_STOCHASTIC_SKLEARN = frozenset(
    {
        # Model selection
        "ShuffleSplit",
        "StratifiedShuffleSplit",
        "GroupShuffleSplit",
        "RepeatedKFold",
        "RepeatedStratifiedKFold",
        "RandomizedSearchCV",
        # Ensembles and trees
        "RandomForestClassifier",
        "RandomForestRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "ExtraTreeClassifier",
        "ExtraTreeRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
        "BaggingClassifier",
        "BaggingRegressor",
        "RandomTreesEmbedding",
        "IsolationForest",
        # Iterative / stochastic fitters
        "MLPClassifier",
        "MLPRegressor",
        "SGDClassifier",
        "SGDRegressor",
        "Perceptron",
        "PassiveAggressiveClassifier",
        "PassiveAggressiveRegressor",
        # Unsupervised with random initialisation
        "KMeans",
        "MiniBatchKMeans",
        "BisectingKMeans",
        "GaussianMixture",
        "BayesianGaussianMixture",
        "TruncatedSVD",
        "FastICA",
        "TSNE",
    }
)

# Deterministic unless ``shuffle=True``; sklearn raises if you pass
# random_state without it.
_SHUFFLE_GATED_SPLITTERS = frozenset({"KFold", "StratifiedKFold", "GroupKFold"})

# Deterministic when handed explicit starting centroids rather than an
# initialisation strategy name.
_INIT_GATED_CLUSTERERS = frozenset({"KMeans", "MiniBatchKMeans", "BisectingKMeans"})


def _has_explicit_init_array(call: ast.Call) -> bool:
    """True if ``init=`` is something other than a strategy name like ``"k-means++"``."""
    init = _has_keyword(call, "init")
    if init is None:
        return False
    return not (isinstance(init.value, ast.Constant) and isinstance(init.value.value, str))


# Meta-estimators whose ``_make_estimator`` stamps its own ``random_state`` onto
# every clone of the inner template. ``AdaBoostClassifier(estimator=
# DecisionTreeClassifier(max_depth=1), random_state=0)`` -- the canonical
# decision-stump example -- is fully seeded, so the inner constructor must not
# be flagged.
#
# Pipeline, GridSearchCV, OneVsRestClassifier and VotingClassifier are
# deliberately absent: they clone without seeding, so an unseeded estimator
# inside one of those really is unseeded.
_SEEDING_META_ESTIMATORS = frozenset(
    {
        "BaggingClassifier",
        "BaggingRegressor",
        "AdaBoostClassifier",
        "AdaBoostRegressor",
    }
)

_ESTIMATOR_SLOTS = ("estimator", "base_estimator")


def _templates_seeded_by_parent(tree: ast.AST) -> set[int]:
    """Node ids of estimator templates that an enclosing meta-estimator seeds."""
    seeded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _has_keyword(node, "random_state") is None:
            continue
        chain = _attr_chain(node.func)
        if not chain or chain[-1] not in _SEEDING_META_ESTIMATORS:
            continue

        candidates: list[ast.expr] = list(node.args[:1])
        for slot in _ESTIMATOR_SLOTS:
            kw = _has_keyword(node, slot)
            if kw is not None:
                candidates.append(kw.value)
        for candidate in candidates:
            if isinstance(candidate, ast.Call):
                seeded.add(id(candidate))
    return seeded


_GLOBAL_NUMPY_SEEDS = ("numpy.random.seed", "np.random.seed")


def _seeds_global_numpy_rng(tree: ast.AST) -> bool:
    """True if the script seeds the global numpy RNG.

    scikit-learn's ``random_state=None`` draws from exactly that generator, so
    a script that seeds it does reproduce run to run and must not be flagged.
    ``np.random.default_rng(...)`` does *not* count: it returns an independent
    generator that scikit-learn never sees.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_matches(node, *_GLOBAL_NUMPY_SEEDS):
            return True
    return False


def _check_random_state(tree: ast.AST, frameworks: set[str]) -> list[Finding]:
    """scikit-learn's answer to the ``seed`` rule: unset ``random_state=``."""
    if _seeds_global_numpy_rng(tree):
        return []

    findings: list[Finding] = []
    seen: set[tuple[str, int]] = set()
    seeded_templates = _templates_seeded_by_parent(tree)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _attr_chain(node.func)
        if not chain:
            continue
        name = chain[-1]

        # ``**params`` may well carry random_state; the AST cannot tell.
        if any(kw.arg is None for kw in node.keywords):
            continue
        if _has_keyword(node, "random_state") is not None:
            continue
        if id(node) in seeded_templates:
            continue

        if name == "train_test_split":
            shuffle = _has_keyword(node, "shuffle")
            if (
                shuffle is not None
                and isinstance(shuffle.value, ast.Constant)
                and shuffle.value.value is False
            ):
                continue
        elif name in _SHUFFLE_GATED_SPLITTERS:
            shuffle = _has_keyword(node, "shuffle")
            shuffled = (
                shuffle is not None
                and isinstance(shuffle.value, ast.Constant)
                and shuffle.value.value is True
            )
            if not shuffled:
                continue
        elif name in _STOCHASTIC_SKLEARN:
            if name in _INIT_GATED_CLUSTERERS and _has_explicit_init_array(node):
                continue
        else:
            continue

        key = (name, node.lineno)
        if key in seen:
            continue
        seen.add(key)
        findings.append(
            Finding(
                rule_id="random_state",
                severity="warning",
                message=(
                    f"`{name}` is called without `random_state=`; it is stochastic "
                    "with its default parameters, so this run cannot be reproduced."
                ),
                suggestion=(
                    f"Pass `random_state=42` to `{name}` (every split, splitter and "
                    "estimator that takes one), or seed the global generator once with "
                    "`np.random.seed(42)`."
                ),
                line=node.lineno,
            )
        )

    return findings


# --------------------------------------------------------------------------- #
# Rule: unused_holdout                                                        #
# --------------------------------------------------------------------------- #


def _check_unused_holdout(tree: ast.AST, frameworks: set[str]) -> list[Finding]:
    """A split was made and one of its outputs is never read again."""
    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }

    findings: list[Finding] = []
    for call, targets in _split_assignments(tree):
        unused = [
            t.id
            for t in targets
            if isinstance(t, ast.Name) and not t.id.startswith("_") and t.id not in loaded
        ]
        if not unused:
            continue
        names = ", ".join(f"`{n}`" for n in unused)
        findings.append(
            Finding(
                rule_id="unused_holdout",
                severity="warning",
                message=(
                    f"`train_test_split` assigns {names}, which is never read again; "
                    "the data was held out and then never scored."
                ),
                suggestion=(
                    "Score the held-out half before saving the model, or name the "
                    "outputs you genuinely do not want `_`."
                ),
                line=call.lineno,
            )
        )

    return findings


# --------------------------------------------------------------------------- #
# Rule: metric_choice                                                         #
# --------------------------------------------------------------------------- #

_OTHER_CLF_METRICS = frozenset(
    {
        "roc_auc_score",
        "average_precision_score",
        "f1_score",
        "fbeta_score",
        "precision_score",
        "recall_score",
        "precision_recall_fscore_support",
        "precision_recall_curve",
        "roc_curve",
        "classification_report",
        "confusion_matrix",
        "balanced_accuracy_score",
        "matthews_corrcoef",
        "cohen_kappa_score",
        "jaccard_score",
        "log_loss",
        "brier_score_loss",
        "top_k_accuracy_score",
    }
)


def _check_metric_choice(tree: ast.AST, frameworks: set[str]) -> list[Finding]:
    """``accuracy_score`` is the only classification metric in the script.

    This is a caution, not an assertion. A static read cannot see the label
    distribution, so the finding is worded conditionally and kept at ``info``.
    """
    accuracy_line: int | None = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # A non-accuracy ``scoring=`` argument counts as a second metric.
        scoring = _has_keyword(node, "scoring")
        if scoring is not None:
            if not isinstance(scoring.value, ast.Constant):
                return []
            if "accuracy" not in str(scoring.value.value):
                return []

        chain = _attr_chain(node.func)
        if not chain:
            continue
        name = chain[-1]
        if name in _OTHER_CLF_METRICS:
            return []
        if name == "accuracy_score" and accuracy_line is None:
            accuracy_line = node.lineno

    if accuracy_line is None:
        return []

    return [
        Finding(
            rule_id="metric_choice",
            severity="info",
            message=(
                "`accuracy_score` is the only classification metric this script "
                "computes. A static read cannot see the label distribution, so this "
                "is a caution rather than a verdict: if the target is imbalanced, "
                "accuracy stays high while the minority class is never predicted."
            ),
            suggestion=(
                "Check the class balance, and report `roc_auc_score`, `f1_score` or "
                "`classification_report` next to accuracy if it is skewed."
            ),
            line=accuracy_line,
        )
    ]
