"""Train and validate the Pre-Race (post-qualifying) and Live In-Race models.

For each stage two candidate models are compared with **walk-forward season
validation** (train on earlier seasons, validate on the next one):

* ``logreg`` - imputed + standardised logistic regression (robust on the small
  pre-race dataset);
* ``hgb``    - histogram gradient boosting (captures non-linear race dynamics).

Raw winner scores are normalised within each race (pre-race) or each
race-lap (in-race) so a field's win probabilities sum to 1, and models are
selected on the resulting **winner log-loss**.  The selected candidate is then
refit on all data and saved to ``backend/ml/artifacts/`` with its validation
metrics.

Usage (from ``backend/``)::

    python -m ml.train                  # both stages
    python -m ml.train --stage pre_race
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from ml.features import (
    IN_RACE_FEATURES,
    PRE_RACE_FEATURES,
    build_in_race_features,
    build_pre_race_features,
    make_in_race_x,
    make_pre_race_x,
    normalize_within_groups,
    recency_weights,
)
from ml.ingest_history import load_in_race_raw, load_pre_race_raw

LOGGER = logging.getLogger(__name__)

PRE_GROUPS = ["season", "round"]
IN_GROUPS = ["season", "round", "lap"]
CANDIDATES = ("logreg", "hgb")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def make_model(kind: str, stage: str, seed: int = config.RANDOM_SEED) -> Pipeline:
    """Fresh, unfitted candidate pipeline (final step is always named ``clf``)."""
    if kind == "logreg":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.3, max_iter=3000)),
            ]
        )
    if kind == "hgb":
        if stage == "pre_race":
            params = dict(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=30)
        else:
            params = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=50)
        return Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        l2_regularization=1.0,
                        early_stopping=False,
                        random_state=seed,
                        **params,
                    ),
                )
            ]
        )
    raise ValueError(f"Unknown model kind: {kind!r}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _py(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(v) for v in value]
    return value


def race_level_metrics(
    meta: pd.DataFrame,
    y: np.ndarray,
    scores: np.ndarray,
    group_cols: Sequence[str],
    baseline_col: str,
) -> Dict[str, Any]:
    """Winner-centric metrics on within-group normalised probabilities."""
    data = meta[list(group_cols) + [baseline_col]].copy().reset_index(drop=True)
    data["y"] = np.asarray(y, dtype=int)
    data["p"] = normalize_within_groups(data, group_cols, scores)
    data = data[data.groupby(list(group_cols), sort=False)["y"].transform("max") == 1]
    if data.empty:
        return {"n_groups": 0}

    data["rank"] = data.groupby(list(group_cols), sort=False)["p"].rank(ascending=False, method="first")
    winners = data[data["y"] == 1]
    sizes = data.groupby(list(group_cols), sort=False)["y"].transform("count")
    leader = data.loc[data[baseline_col] == 1]
    return _py(
        {
            "n_groups": int(len(winners)),
            "logloss": float(-np.log(winners["p"].clip(lower=1e-6)).mean()),
            "uniform_logloss": float(np.log(sizes[winners.index]).mean()),
            "top1_accuracy": float((winners["rank"] == 1).mean()),
            "top3_accuracy": float((winners["rank"] <= 3).mean()),
            "mean_winner_prob": float(winners["p"].mean()),
            "brier": float(((data["p"] - data["y"]) ** 2).mean()),
            "baseline_top1_accuracy": float(leader["y"].mean()) if len(leader) else None,
        }
    )


def make_splits(
    meta: pd.DataFrame, n_splits: int, min_train_seasons: int
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """Walk-forward season splits, with a chronological fallback for tiny data."""
    seasons = sorted(meta["season"].unique())
    eligible = [s for i, s in enumerate(seasons) if i >= min_train_seasons]
    splits: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for season in eligible[-n_splits:]:
        train_idx = np.flatnonzero((meta["season"] < season).to_numpy())
        val_idx = np.flatnonzero((meta["season"] == season).to_numpy())
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx, f"season_{int(season)}"))
    if splits:
        return splits

    keys = meta[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    if len(keys) < 5:
        return []
    cut = max(1, int(len(keys) * 0.8))
    val_keys = set(map(tuple, keys.iloc[cut:].to_numpy().tolist()))
    is_val = np.asarray(
        [(s, r) in val_keys for s, r in zip(meta["season"].tolist(), meta["round"].tolist())]
    )
    return [(np.flatnonzero(~is_val), np.flatnonzero(is_val), "last_20pct_of_races")]


def _fit(model: Pipeline, X: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> Pipeline:
    model.fit(X, y, clf__sample_weight=weights)
    return model


def _mean_metric(folds: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [f[key] for f in folds if f.get(key) is not None]
    return float(np.mean(values)) if values else None


def select_and_fit(
    stage: str,
    X: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,
    group_cols: Sequence[str],
    baseline_col: str,
    *,
    candidates: Sequence[str] = CANDIDATES,
    n_splits: int = config.WALK_FORWARD_SPLITS,
    min_train_seasons: int = config.MIN_TRAIN_SEASONS,
    half_life: float = config.RECENCY_HALF_LIFE_SEASONS,
    seed: int = config.RANDOM_SEED,
) -> Tuple[Pipeline, str, Dict[str, Any]]:
    """Compare candidates with walk-forward validation, refit the winner."""
    weights = recency_weights(meta["season"].to_numpy(), half_life)
    splits = make_splits(meta, n_splits, min_train_seasons)
    report: Dict[str, Any] = {}
    for kind in candidates:
        folds: List[Dict[str, Any]] = []
        for train_idx, val_idx, label in splits:
            if y[train_idx].sum() == 0 or y[val_idx].sum() == 0:
                continue
            model = _fit(make_model(kind, stage, seed), X.iloc[train_idx], y[train_idx], weights[train_idx])
            scores = model.predict_proba(X.iloc[val_idx])[:, 1]
            fold = race_level_metrics(meta.iloc[val_idx], y[val_idx], scores, group_cols, baseline_col)
            fold["fold"] = label
            folds.append(fold)
            LOGGER.info(
                "[%s/%s] %s: logloss=%.3f top1=%.3f (uniform %.3f)",
                stage, kind, label, fold.get("logloss", float("nan")),
                fold.get("top1_accuracy", float("nan")), fold.get("uniform_logloss", float("nan")),
            )
        report[kind] = {
            "folds": folds,
            "mean_logloss": _mean_metric(folds, "logloss"),
            "mean_top1_accuracy": _mean_metric(folds, "top1_accuracy"),
            "mean_top3_accuracy": _mean_metric(folds, "top3_accuracy"),
            "mean_brier": _mean_metric(folds, "brier"),
            "mean_uniform_logloss": _mean_metric(folds, "uniform_logloss"),
            "mean_baseline_top1_accuracy": _mean_metric(folds, "baseline_top1_accuracy"),
        }

    scored = {k: v["mean_logloss"] for k, v in report.items() if v["mean_logloss"] is not None}
    best = min(scored, key=scored.get) if scored else candidates[0]
    LOGGER.info("[%s] selected model: %s", stage, best)

    final = _fit(make_model(best, stage, seed), X, y, weights)
    summary = {
        "selected_model": best,
        "candidates": report,
        "n_rows": int(len(X)),
        "n_groups": int(meta[list(group_cols)].drop_duplicates().shape[0]),
        "seasons": [int(meta["season"].min()), int(meta["season"].max())],
        "validation_splits": [label for _, _, label in splits],
    }
    if best in scored:
        chosen = report[best]
        summary["validation"] = {
            "logloss": chosen["mean_logloss"],
            "top1_accuracy": chosen["mean_top1_accuracy"],
            "top3_accuracy": chosen["mean_top3_accuracy"],
            "brier": chosen["mean_brier"],
            "uniform_logloss": chosen["mean_uniform_logloss"],
            "baseline_top1_accuracy": chosen["mean_baseline_top1_accuracy"],
        }
    else:
        summary["validation"] = {}
    return final, best, _py(summary)


# --------------------------------------------------------------------------- #
# Stage runners
# --------------------------------------------------------------------------- #
def _artifact(stage: str, model: Pipeline, kind: str, features: Sequence[str], summary: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "stage": stage,
        "model": model,
        "features": list(features),
        "model_name": kind,
        "version": f"{stage}-{kind}-{now.strftime('%Y%m%d%H%M%S')}",
        "trained_at": now.isoformat(timespec="milliseconds"),
        "validation": summary.get("validation", {}),
        "summary": summary,
    }


def train_pre_race(feature_table: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """Train the post-qualifying model from ``build_pre_race_features`` output."""
    df = feature_table[feature_table["completed"].astype(bool) & feature_table["grid_position"].notna()]
    df = df.reset_index(drop=True)
    y = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(int).to_numpy()
    if y.sum() < 5:
        raise ValueError("Pre-race training needs at least 5 race winners.")
    X = make_pre_race_x(df)
    model, kind, summary = select_and_fit(
        "pre_race", X, y, df[["season", "round", "grid_position"]], PRE_GROUPS, "grid_position", **kwargs
    )
    return _artifact("pre_race", model, kind, PRE_RACE_FEATURES, summary)


def train_in_race(in_race_features: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """Train the live model from ``build_in_race_features`` output."""
    df = in_race_features.dropna(subset=["won", "position"]).reset_index(drop=True)
    y = df["won"].astype(int).to_numpy()
    if y.sum() < 5:
        raise ValueError("In-race training needs at least 5 labelled winner snapshots.")
    X = make_in_race_x(df)
    model, kind, summary = select_and_fit(
        "in_race", X, y, df[["season", "round", "lap", "position"]], IN_GROUPS, "position", **kwargs
    )
    return _artifact("in_race", model, kind, IN_RACE_FEATURES, summary)


def save_artifact(artifact: Dict[str, Any], path: Any) -> None:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, target)
    metrics: Dict[str, Any] = {}
    if config.METRICS_PATH.exists():
        try:
            metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
        except ValueError:
            metrics = {}
    metrics[artifact["stage"]] = {
        "version": artifact["version"],
        "trained_at": artifact["trained_at"],
        "model_name": artifact["model_name"],
        **artifact["summary"],
    }
    config.METRICS_PATH.write_text(json.dumps(_py(metrics), indent=2), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the F1 Pit-Wall winner models.")
    parser.add_argument("--stage", choices=["both", "pre_race", "in_race"], default="both")
    parser.add_argument("--splits", type=int, default=config.WALK_FORWARD_SPLITS)
    parser.add_argument("--min-train-seasons", type=int, default=config.MIN_TRAIN_SEASONS)
    parser.add_argument("--half-life", type=float, default=config.RECENCY_HALF_LIFE_SEASONS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    config.configure_logging(args.verbose)
    config.ensure_directories()
    options = dict(
        n_splits=args.splits,
        min_train_seasons=args.min_train_seasons,
        half_life=args.half_life,
        seed=args.seed,
    )

    trained = 0
    pre_features: Optional[pd.DataFrame] = None
    try:
        pre_features = build_pre_race_features(load_pre_race_raw())
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)

    if args.stage in ("both", "pre_race") and pre_features is not None:
        artifact = train_pre_race(pre_features, **options)
        save_artifact(artifact, config.PRE_RACE_MODEL_PATH)
        LOGGER.info("Saved %s -> %s", artifact["version"], config.PRE_RACE_MODEL_PATH)
        trained += 1

    if args.stage in ("both", "in_race"):
        try:
            snapshots = load_in_race_raw()
        except FileNotFoundError as exc:
            LOGGER.warning("Skipping in-race model: %s", exc)
        else:
            if snapshots.empty:
                LOGGER.warning("Skipping in-race model: dataset is empty.")
            else:
                features = build_in_race_features(snapshots, pre_features)
                artifact = train_in_race(features, **options)
                save_artifact(artifact, config.IN_RACE_MODEL_PATH)
                LOGGER.info("Saved %s -> %s", artifact["version"], config.IN_RACE_MODEL_PATH)
                trained += 1
    return 0 if trained else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
