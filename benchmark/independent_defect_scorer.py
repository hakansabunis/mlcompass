"""A second opinion on the six-item checklist, by a different method.

`run_ab.check_defects` decides the six rules with regular expressions over the
script's text. This module decides them by walking the parsed syntax tree. The
point is not that ast is better -- it is that the two fail differently, so a
disagreement is informative in a way a second regex implementation would not
be.

Two rules are deliberately inverted. `check_defects` scores
`target_in_features` and `no_validation` by the ABSENCE of every enumerated
correct form; its own comments record that this is why one of the defects in
Section 10 existed, because every form nobody thought of becomes a false
positive on correct code. Here they fire only on POSITIVE evidence of the
defect. That makes this scorer strictly more conservative on those two rules by
construction, and the disagreements it produces are the enumeration's blind
spots rather than noise.

What this is not: an independent replication. Both scorers are the same
authors', written days apart. It is a differential audit and the paper says so.

Nothing here imports from run_ab. Restating is the whole point.
"""

from __future__ import annotations

import ast
from typing import Any

RULES = (
    "no_seed",
    "leak_fit_before_split",
    "leak_duplicate_rows",
    "wrong_metric_for_imbalance",
    "no_validation",
    "target_in_features",
)

# Frameworks whose absence of seeding is a reproducibility defect, and the
# calls or keywords that count as seeding one.
_SEEDABLE = {
    "random": {"seed"},
    "numpy": {"seed", "default_rng", "RandomState"},
    "sklearn": set(),  # seeded through random_state=, handled below
    "torch": {"manual_seed", "seed"},
    "tensorflow": {"set_seed", "set_random_seed"},
    "lightgbm": set(),
    "xgboost": set(),
}
_SEED_KEYWORDS = {"random_state", "seed", "random_seed"}

_SPLITTERS = {
    "train_test_split",
    "StratifiedShuffleSplit",
    "ShuffleSplit",
    "GroupShuffleSplit",
}
_CV = {
    "cross_val_score",
    "cross_validate",
    "cross_val_predict",
    "KFold",
    "StratifiedKFold",
    "GroupKFold",
    "RepeatedKFold",
    "TimeSeriesSplit",
    "GridSearchCV",
    "RandomizedSearchCV",
    "HalvingGridSearchCV",
}
_ACCURACY = {"accuracy_score"}
_FITTERS = {"fit", "fit_transform", "fit_resample"}


def _root_module(name: str) -> str:
    return (name or "").split(".")[0]


class _Walk(ast.NodeVisitor):
    """One pass, collecting everything the six rules need."""

    def __init__(self) -> None:
        self.imports: set[str] = set()
        self.seed_calls: list[tuple[str, str]] = []  # (module, attr)
        self.seed_kwargs: list[str] = []  # keyword names given a literal
        self.split_lines: list[int] = []
        self.split_first_args: list[ast.expr] = []
        self.cv_lines: list[int] = []
        self.fit_lines: list[tuple[int, ast.Call]] = []
        self.dedupe_lines: list[int] = []
        self.accuracy_hits: list[int] = []
        self.accuracy_strings: list[int] = []
        self.assigns: list[tuple[int, list[str], ast.expr]] = []
        self.score_calls: list[int] = []

    # -- imports ---------------------------------------------------------- #
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self.imports.add(_root_module(a.name))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(_root_module(node.module))
        self.generic_visit(node)

    # -- assignments ------------------------------------------------------ #
    def visit_Assign(self, node: ast.Assign) -> None:
        names: list[str] = []
        for t in node.targets:
            if isinstance(t, ast.Name):
                names.append(t.id)
            elif isinstance(t, ast.Tuple):
                names.extend(e.id for e in t.elts if isinstance(e, ast.Name))
        self.assigns.append((node.lineno, names, node.value))
        self.generic_visit(node)

    # -- calls ------------------------------------------------------------ #
    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        attr = fn.attr if isinstance(fn, ast.Attribute) else None
        plain = fn.id if isinstance(fn, ast.Name) else attr

        if attr:
            owner = fn.value
            chain = []
            while isinstance(owner, ast.Attribute):
                chain.append(owner.attr)
                owner = owner.value
            if isinstance(owner, ast.Name):
                chain.append(owner.id)
            root = chain[-1] if chain else ""
            for mod, attrs in _SEEDABLE.items():
                if attrs and root.startswith(mod[:2]) and attr in attrs:
                    self.seed_calls.append((mod, attr))
            if root in {"np", "numpy"} and attr in _SEEDABLE["numpy"]:
                self.seed_calls.append(("numpy", attr))
            if root in {"torch"} and attr in _SEEDABLE["torch"]:
                self.seed_calls.append(("torch", attr))
            if root in {"random"} and attr == "seed":
                self.seed_calls.append(("random", attr))
            if root in {"tf", "tensorflow"} and attr in _SEEDABLE["tensorflow"]:
                self.seed_calls.append(("tensorflow", attr))

        if plain in _SPLITTERS:
            self.split_lines.append(node.lineno)
            if node.args:
                self.split_first_args.append(node.args[0])
        if plain in _CV:
            self.cv_lines.append(node.lineno)
        if plain in _ACCURACY:
            self.accuracy_hits.append(node.lineno)
        if attr in _FITTERS:
            self.fit_lines.append((node.lineno, node))
        if attr == "drop_duplicates":
            self.dedupe_lines.append(node.lineno)
        if attr == "score":
            self.score_calls.append(node.lineno)

        for kw in node.keywords:
            if kw.arg in _SEED_KEYWORDS and _is_bound_literal(kw.value):
                self.seed_kwargs.append(kw.arg)

        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and node.value.strip().lower() == "accuracy":
            self.accuracy_strings.append(node.lineno)
        self.generic_visit(node)


def _is_bound_literal(node: ast.expr) -> bool:
    """A seed is bound if it is a number, or a Name (a hoisted constant).

    Accepting a bare Name is the deliberate conservative choice: resolving it
    would reimplement the very constant-expansion whose regex version was the
    defect reported in Section 10, and an unresolvable name is far more often a
    hoisted seed than an unseeded call.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return True
    return isinstance(node, (ast.Name, ast.Attribute))


def _names_in(node: ast.expr) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _strings_in(node: ast.expr) -> set[str]:
    return {
        n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def _target_forms(target: str) -> set[str]:
    t = (target or "").strip()
    return {t, t.lower(), t.upper()} - {""}


def score(
    source: str,
    *,
    target: str,
    duplicate_rows: int,
    minority_fraction: float | None,
    target_is_last_column: bool = False,
) -> dict[str, Any]:
    """Six booleans, decided from the syntax tree. Same signature as the first
    scorer so the two can be run side by side on identical inputs."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        # A script that does not parse gets no opinion from this scorer. The
        # caller must treat that as "no second opinion", never as "clean".
        return {"flags": None, "parse_error": f"{exc.__class__.__name__}: {exc}"}

    w = _Walk()
    w.visit(tree)

    # ---- no_seed: no seeding evidence of any kind ------------------------ #
    seedable_imported = {m for m in w.imports if m in _SEEDABLE}
    has_seed = bool(w.seed_calls or w.seed_kwargs)
    no_seed = bool(seedable_imported) and not has_seed
    if not seedable_imported:
        # Nothing seedable was imported. The first scorer calls this a defect;
        # this one does not, because there is no randomness to pin.
        no_seed = False

    # ---- leak_fit_before_split ------------------------------------------ #
    split_at = min(w.split_lines) if w.split_lines else None
    pre_split_frames: set[str] = set()
    for arg in w.split_first_args:
        pre_split_frames |= _names_in(arg)
    fitted_before = False
    fitted_on_full = False
    for lineno, call in w.fit_lines:
        if split_at is not None and lineno < split_at:
            fitted_before = True
        if (
            call.args
            and (_names_in(call.args[0]) & pre_split_frames)
            and (split_at is None or lineno <= split_at)
        ):
            fitted_on_full = True
    leak_fit_before_split = fitted_before or fitted_on_full

    # ---- leak_duplicate_rows -------------------------------------------- #
    if duplicate_rows == 0:
        leak_duplicate_rows = False
    else:
        # Deadline: the line that first binds a name the split consumes. A
        # dedupe after that point cannot reach the data being split.
        deadline = None
        for lineno, names, _value in w.assigns:
            if set(names) & pre_split_frames:
                deadline = lineno if deadline is None else min(deadline, lineno)
        if deadline is None:
            deadline = split_at
        if deadline is None:
            leak_duplicate_rows = not w.dedupe_lines
        else:
            leak_duplicate_rows = not any(i <= deadline for i in w.dedupe_lines)

    # ---- wrong_metric_for_imbalance ------------------------------------- #
    imbalanced = minority_fraction is not None and minority_fraction < 0.20
    wrong_metric_for_imbalance = bool(imbalanced and (w.accuracy_hits or w.accuracy_strings))

    # ---- no_validation: INVERTED, fires on positive evidence ------------- #
    # The first scorer flags the absence of every enumerated validation form.
    # This one flags only a script that trains and reports without ever
    # separating data: a fit exists, and no splitter, no CV object and no
    # .score() call appears anywhere.
    no_validation = bool(w.fit_lines) and not (w.split_lines or w.cv_lines or w.score_calls)

    # ---- target_in_features: INVERTED ----------------------------------- #
    # Fires only where the target is demonstrably still in the matrix that is
    # split: either the split consumes a frame with no filtering applied at
    # all, or a literal column list names the target.
    forms = _target_forms(target)
    target_in_features = False
    for arg in w.split_first_args:
        if isinstance(arg, ast.Name):
            # Trace the binding once. No filtering call anywhere in the
            # defining expression means the whole frame went in.
            for _lineno, names, value in w.assigns:
                if arg.id in names:
                    calls = {
                        n.func.attr
                        for n in ast.walk(value)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    }
                    subscripted = any(isinstance(n, ast.Subscript) for n in ast.walk(value))
                    filtered = (
                        bool(calls & {"drop", "pop", "difference", "select_dtypes"}) or subscripted
                    )
                    if not filtered and isinstance(value, ast.Name):
                        target_in_features = True
                    if _strings_in(value) & forms and subscripted:
                        # An explicit column list that names the target.
                        listed = any(
                            isinstance(n, (ast.List, ast.Tuple)) and (_strings_in(n) & forms)
                            for n in ast.walk(value)
                        )
                        dropped = bool(calls & {"drop", "pop", "difference"})
                        if listed and not dropped:
                            target_in_features = True
        elif isinstance(arg, (ast.Attribute, ast.Call)):
            pass  # an expression built inline; no positive evidence either way

    flags = {
        "no_seed": no_seed,
        "leak_fit_before_split": leak_fit_before_split,
        "leak_duplicate_rows": leak_duplicate_rows,
        "wrong_metric_for_imbalance": wrong_metric_for_imbalance,
        "no_validation": no_validation,
        "target_in_features": target_in_features,
    }
    return {
        "flags": flags,
        "defect_count": sum(1 for v in flags.values() if v),
        "imports": sorted(w.imports),
        "seeded_by": sorted({m for m, _ in w.seed_calls} | set(w.seed_kwargs)),
        "split_line": split_at,
        "dedupe_lines": w.dedupe_lines,
        "fit_lines": [line for line, _ in w.fit_lines],
    }
