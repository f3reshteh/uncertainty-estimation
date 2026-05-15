from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


RESULTS_DIR = Path(".")
THRESHOLDS = np.linspace(0.0, 1.0, 201)

BASE_MODELS = ["GEMINI_2.5_flashlite", "GPT20B", "MISTRAL_24B", "QWEN_32B"]
CHECK_KEY = ["project_id", "test_code", "check_code"]
TEST_KEY = ["project_id", "test_code"]
EPS = 1e-6
Q_LOW = 0.05
Q_HIGH = 0.95

EXCLUDE_SCORE_COLS = {
    "source",
    "source_file",
    "judge_backend",
    "judge_model",
    "judge_raw",
    "parsed_test_name",
    "project_id",
    "test_code",
    "check_code",
    "gt",
    "pred",
    "match",
    "label",
    "response",
    "input_tokens",
    "output_tokens",
    "cost_usd",
}

RANKING_KEEP_COLS = [
    "model",
    "metric",
    "model_error_rate",
    "average_precision",
    "ap_lift_over_baseline",
    "average_precision_error",
    "ap_error_lift_over_baseline",
    "roc_auc_error",
    "brier_error",
    "flipped",
    "q05",
    "q95",
]


def parse_wb_bb(name, prefix):
    stem = name[len(prefix):-4]
    for bm in BASE_MODELS:
        suffix = f"_{bm}"
        if stem.endswith(suffix):
            return stem[:-len(suffix)], bm
    return None, None


def to_binary_flag(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce") > 0).astype(float)
    return s.astype(str).str.strip().str.lower().map(
        {
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
            "yes": 1.0,
            "no": 0.0,
        }
    )


def norm_score(s):
    x = pd.to_numeric(s, errors="coerce").astype(float)
    if x.notna().any():
        q99, q01 = x.dropna().quantile(0.99), x.dropna().quantile(0.01)
        if q99 > 1.5 and q99 <= 100.5 and q01 >= 0:
            x = x / 100.0
    return x.clip(EPS, 1 - EPS)


def soft_or(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    arr = np.clip(arr, EPS, 1 - EPS)
    return float(1.0 - np.prod(1.0 - arr))


def max_agg(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(arr.max())


def mean_agg(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(arr.mean())


def make_gated_soft_or(floor):
    """Soft-OR that zeroes out checks below `floor` before OR-ing.
    Kills noise compounding from quiet checks while preserving the
    'multiple alarms reinforce' semantics when several checks are loud."""
    def gated(values):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        arr = np.where(arr < floor, 0.0, arr)
        arr = np.clip(arr, 0.0, 1 - EPS)
        return float(1.0 - np.prod(1.0 - arr))
    gated.__name__ = f"gated_soft_or_floor_{floor}"
    return gated


def scale_to_error(values, q_low=Q_LOW, q_high=Q_HIGH):
    arr = np.asarray(values, dtype=float)
    lo, hi = np.quantile(arr, [q_low, q_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        scaled = np.full_like(arr, 0.5, dtype=float)
    else:
        scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.clip(scaled, EPS, 1 - EPS), float(lo), float(hi)


def add_model_error_target(df, match_col):
    out = df.copy()
    out["label"] = to_binary_flag(out[match_col])
    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].round().astype(int)
    out["error_label"] = 1 - out["label"]
    return out


def require_constant_within_tests(df, label_col, context, group_cols=None):
    if df.empty:
        return
    group_cols = list(group_cols or [])
    label_nunique = df.groupby(group_cols + TEST_KEY, dropna=False)[label_col].nunique(dropna=False)
    bad = label_nunique[label_nunique > 1]
    if not bad.empty:
        sample = [tuple(idx) if isinstance(idx, tuple) else idx for idx in bad.index[:5]]
        raise ValueError(f"Inconsistent test labels found for {context}: {sample}")


def summarize_ranking(y_correct, correct_score):
    y_correct = np.asarray(y_correct).astype(int)
    correct_score = np.clip(np.asarray(correct_score).astype(float), EPS, 1 - EPS)
    y_error = 1 - y_correct

    correct_rate = float(y_correct.mean())
    model_error_rate = float(y_error.mean())

    average_precision = (
        float(average_precision_score(y_correct, correct_score))
        if len(np.unique(y_correct)) > 1
        else np.nan
    )
    average_precision_error = (
        float(average_precision_score(y_error, 1 - correct_score))
        if len(np.unique(y_error)) > 1
        else np.nan
    )

    return {
        "correct_rate": correct_rate,
        "model_error_rate": model_error_rate,
        "average_precision": average_precision,
        "ap_lift_over_baseline": (
            float(average_precision / correct_rate)
            if correct_rate > 0 and not np.isnan(average_precision)
            else np.nan
        ),
        "average_precision_error": average_precision_error,
        "ap_error_lift_over_baseline": (
            float(average_precision_error / model_error_rate)
            if model_error_rate > 0 and not np.isnan(average_precision_error)
            else np.nan
        ),
    }


def build_metric_frame(df, metric, match_col="match"):
    required = TEST_KEY + [match_col, metric]
    if any(col not in df.columns for col in required):
        return pd.DataFrame(columns=TEST_KEY + ["label", "error_label", "raw_score"])

    metric_frame = add_model_error_target(df[required].copy(), match_col)
    metric_frame["raw_score"] = norm_score(metric_frame[metric])
    metric_frame = metric_frame.dropna(subset=["raw_score"]).copy()
    if metric_frame.empty:
        return metric_frame

    require_constant_within_tests(metric_frame, "label", f"metric={metric}")
    return metric_frame[TEST_KEY + ["label", "error_label", "raw_score"]]


def evaluate_metric(df, metric, match_col="match", auto_flip=True, agg_fn=soft_or, scale=True,
                    q_low=Q_LOW, q_high=Q_HIGH):
    metric_frame = build_metric_frame(df, metric, match_col=match_col)
    if metric_frame.empty:
        return None

    best = None
    for flipped in ([False, True] if auto_flip else [False]):
        raw = metric_frame["raw_score"].to_numpy(dtype=float)
        if scale:
            oriented = -raw if flipped else raw
            check_error_score, lo, hi = scale_to_error(oriented, q_low=q_low, q_high=q_high)
        else:
            check_error_score = (1.0 - raw) if flipped else raw
            lo, hi = float("nan"), float("nan")

        temp = metric_frame[TEST_KEY + ["label", "error_label"]].copy()
        temp["check_error_score"] = check_error_score
        test_df = (
            temp.groupby(TEST_KEY, as_index=False)
            .agg(
                label=("label", "first"),
                error_label=("error_label", "first"),
                test_error_score=("check_error_score", agg_fn),
            )
            .dropna(subset=["test_error_score"])
        )

        y_error = test_df["error_label"].astype(int).to_numpy()
        y_correct = test_df["label"].astype(int).to_numpy()
        error_scores = test_df["test_error_score"].astype(float).to_numpy()
        correct_scores = 1.0 - error_scores

        if len(y_error) < 10 or np.unique(y_error).size < 2:
            return None

        summary = summarize_ranking(y_correct, correct_scores)
        result = {
            **summary,
            "roc_auc_error": float(roc_auc_score(y_error, error_scores)),
            "brier_error": float(brier_score_loss(y_error, error_scores)),
            "flipped": bool(flipped),
            "q05": lo,
            "q95": hi,
        }
        if best is None or result["average_precision_error"] > best["average_precision_error"]:
            best = result

    return best


def evaluate_metric_files(file_glob, prefix, metrics, auto_flip=True, agg_fn=soft_or, scale=True,
                          q_low=Q_LOW, q_high=Q_HIGH):
    files = sorted(p for p in RESULTS_DIR.glob(file_glob) if ".bak_" not in p.name)
    if not files:
        raise FileNotFoundError(f"No CSVs found matching {file_glob}")

    grouped = {}
    for path in files:
        _, model = parse_wb_bb(path.name, prefix)
        if model is None:
            continue
        grouped.setdefault(model, []).append(pd.read_csv(path))

    rows = []
    for model, dfs in grouped.items():
        df = pd.concat(dfs, ignore_index=True)
        if "match" not in df.columns:
            print(f"[WARN] model={model}: no 'match' column; skipped.")
            continue

        metrics_here = [metric for metric in metrics if metric in df.columns]
        if not metrics_here:
            print(f"[WARN] model={model}: none of the configured metrics found; skipped.")
            continue

        for metric in metrics_here:
            result = evaluate_metric(df, metric, match_col="match", auto_flip=auto_flip,
                                     agg_fn=agg_fn, scale=scale, q_low=q_low, q_high=q_high)
            if result is not None:
                rows.append({"model": model, "metric": metric, **result})

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        raise RuntimeError("No valid model/metric evaluations were produced.")

    results_df = results_df.sort_values(
        ["model", "average_precision_error", "ap_error_lift_over_baseline", "roc_auc_error", "brier_error"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    return results_df[RANKING_KEEP_COLS]


# ----------------------------------------------------------------------
# Test-level ensemble helpers (used by the GPT20B calibration notebook)
# ----------------------------------------------------------------------

def aggregate_check_error_to_tests(df, score_col, agg_fn=soft_or):
    return (
        df[TEST_KEY + ["label", "error_label", score_col]]
        .groupby(TEST_KEY, as_index=False)
        .agg(
            label=("label", "first"),
            error_label=("error_label", "first"),
            test_error_score=(score_col, agg_fn),
        )
    )


def scale_with_fixed_bounds(values, q05, q95):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(q05) or not np.isfinite(q95) or q95 <= q05:
        return np.full(values.shape, 0.5, dtype=float)
    scaled = (values - q05) / (q95 - q05)
    return np.clip(scaled, 0.0, 1.0)


def threshold_metrics(y_true, scores, tau):
    pred = (scores >= tau).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    f2 = (5 * precision * recall) / max(4 * precision + recall, 1e-12)
    tpr = recall
    tnr = tn / max(tn + fp, 1)
    bal_acc = (tpr + tnr) / 2

    return {
        "threshold": float(tau),
        "balanced_accuracy": float(bal_acc),
        "accuracy": float((pred == y_true).mean()),
        "precision_error": float(precision),
        "recall_error": float(recall),
        "f1_error": float(f1),
        "f2_error": float(f2),
        "flagged_rate": float(pred.mean()),
        "n_flagged": int(pred.sum()),
        "tp_error": tp,
        "fp_error": fp,
        "fn_error": fn,
        "tn_correct": tn,
    }


def choose_shared_threshold(summary_df, primary_metric, recall_floor=None):
    """Pick a threshold from the per-threshold summary.

    primary_metric:
        "f1"               -> max f1, then recall, then precision, then lower threshold
        "f2"               -> max f2, then recall, then precision, then lower threshold
        "balanced_accuracy"-> max bal acc, then f2, then precision, then lower threshold
        "precision_floor"  -> highest threshold whose mean recall >= recall_floor
    """
    if primary_metric == "balanced_accuracy":
        return summary_df.sort_values(
            ["balanced_accuracy_mean", "f2_error_mean", "precision_error_mean", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]

    if primary_metric == "f2":
        return summary_df.sort_values(
            ["f2_error_mean", "recall_error_mean", "precision_error_mean", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]

    if primary_metric == "f1":
        return summary_df.sort_values(
            ["f1_error_mean", "recall_error_mean", "precision_error_mean", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]

    if primary_metric == "precision_floor":
        if recall_floor is None:
            raise ValueError("precision_floor mode requires recall_floor")
        eligible = summary_df[summary_df["recall_error_mean"] >= recall_floor]
        if eligible.empty:
            # fall back to the threshold with the highest recall
            return summary_df.sort_values(
                ["recall_error_mean", "precision_error_mean", "threshold"],
                ascending=[False, False, True],
            ).iloc[0]
        return eligible.sort_values(
            ["precision_error_mean", "threshold"],
            ascending=[False, False],
        ).iloc[0]

    raise ValueError(
        "primary_metric must be one of 'f1', 'f2', 'balanced_accuracy', 'precision_floor'"
    )


def learn_global_orientation_and_bounds(df_all, method, q_low=Q_LOW, q_high=Q_HIGH, agg_fn=soft_or, scale=True):
    """Pick orientation (flip yes/no) and per-check q05/q95 by maximising
    test-level AP_error on the pooled ensembles. When scale=False, q05/q95
    are not used (returned as NaN) and orientation flips via 1-raw instead
    of clipping to quantile bounds; this assumes raw scores are bounded in
    [0, 1]."""
    raw_values = df_all[method].to_numpy(dtype=float)

    candidates = []
    for flipped in (False, True):
        if scale:
            oriented = -raw_values if flipped else raw_values
            q05 = float(np.nanquantile(oriented, q_low))
            q95 = float(np.nanquantile(oriented, q_high))
            check_score = scale_with_fixed_bounds(oriented, q05, q95)
        else:
            check_score = (1.0 - raw_values) if flipped else raw_values
            q05 = float("nan")
            q95 = float("nan")

        temp = df_all[TEST_KEY + ["label", "error_label"]].copy()
        temp["check_error_score"] = check_score

        test_df = (
            aggregate_check_error_to_tests(temp, "check_error_score", agg_fn=agg_fn)
            .sort_values(TEST_KEY)
            .reset_index(drop=True)
        )

        y_test = test_df["error_label"].astype(int).to_numpy()
        if np.unique(y_test).size < 2:
            ap_error = np.nan
        else:
            ap_error = float(
                average_precision_score(
                    y_test,
                    test_df["test_error_score"].to_numpy(dtype=float),
                )
            )

        candidates.append({
            "method": method,
            "flipped": bool(flipped),
            "q05": q05,
            "q95": q95,
            "ap_error": ap_error,
        })

    return max(candidates, key=lambda x: (x["ap_error"] if x["ap_error"] == x["ap_error"] else -1.0))


def build_test_work(df, global_params, methods, agg_fn=soft_or, scale=True):
    """Apply global orientation (and optional q05/q95 scaling) to each method,
    mean across methods at the check level, then aggregate across checks at
    the test level with agg_fn (default soft_or). When scale=False, raw scores
    are taken as-is (with 1-raw on flipped scorers); this assumes raw values
    are in [0, 1]."""
    work = df[CHECK_KEY + ["label", "error_label"] + list(methods)].copy()
    params_map = global_params.set_index("method").to_dict(orient="index")

    norm_cols = []
    for method in methods:
        p = params_map[method]
        raw = work[method].to_numpy(dtype=float)

        col = f"{method}__errnorm"
        if scale:
            oriented = -raw if p["flipped"] else raw
            work[col] = scale_with_fixed_bounds(oriented, p["q05"], p["q95"])
        else:
            work[col] = (1.0 - raw) if p["flipped"] else raw
        norm_cols.append(col)

    work["check_ensemble_error_score"] = work[norm_cols].mean(axis=1)

    return (
        aggregate_check_error_to_tests(work, "check_ensemble_error_score", agg_fn=agg_fn)
        .sort_values(TEST_KEY)
        .reset_index(drop=True)
        .rename(columns={"test_error_score": "test_ensemble_error_score"})
    )


def load_one_ensemble(data_dir, ensemble_id, methods, target_base_model=None):
    """Load all CSVs for one ensemble run, attach error labels, optionally
    filter by base_model. Used by the calibration notebook."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(f"GPT20_Ensemble{ensemble_id}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No files found for Ensemble {ensemble_id}")

    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

    if target_base_model is not None and "base_model" in df.columns:
        df = df[df["base_model"] == target_base_model].copy()

    required_cols = [*CHECK_KEY, "match", *methods]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Ensemble {ensemble_id}: missing required columns: {missing}")

    df = add_model_error_target(df, "match")
    require_constant_within_tests(df, "label", f"ensemble {ensemble_id}")
    df["ensemble"] = ensemble_id
    return df


def per_threshold_summary(threshold_rows_df):
    """Aggregate per-(ensemble, threshold) rows into a per-threshold summary."""
    return (
        threshold_rows_df
        .groupby("threshold", as_index=False)
        .agg(
            n_versions=("ensemble", "nunique"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_error_mean=("precision_error", "mean"),
            recall_error_mean=("recall_error", "mean"),
            f1_error_mean=("f1_error", "mean"),
            f2_error_mean=("f2_error", "mean"),
            flagged_rate_mean=("flagged_rate", "mean"),
            n_flagged_mean=("n_flagged", "mean"),
            tp_error_mean=("tp_error", "mean"),
            fp_error_mean=("fp_error", "mean"),
            fn_error_mean=("fn_error", "mean"),
            tn_correct_mean=("tn_correct", "mean"),
        )
    )


def evaluate_subset_at_threshold(ensemble_dfs, methods_subset, threshold, q_low=Q_LOW, q_high=Q_HIGH, agg_fn=soft_or):
    """Fit global params on the pooled ensembles, then report mean threshold
    metrics across the per-ensemble evaluations. Used for ablation."""
    df_all = pd.concat(ensemble_dfs, ignore_index=True)
    global_params = pd.DataFrame(
        [
            learn_global_orientation_and_bounds(df_all, m, q_low=q_low, q_high=q_high, agg_fn=agg_fn)
            for m in methods_subset
        ]
    )

    rows = []
    for df in ensemble_dfs:
        test_work = build_test_work(df, global_params, methods_subset, agg_fn=agg_fn)
        y_test = test_work["error_label"].astype(int).to_numpy()
        scores = test_work["test_ensemble_error_score"].to_numpy(dtype=float)
        rows.append(threshold_metrics(y_test, scores, threshold))

    out = pd.DataFrame(rows)
    return {
        "methods": ", ".join(methods_subset),
        "n_methods": len(methods_subset),
        "threshold": float(threshold),
        "precision_error_mean": float(out["precision_error"].mean()),
        "recall_error_mean": float(out["recall_error"].mean()),
        "f1_error_mean": float(out["f1_error"].mean()),
        "f2_error_mean": float(out["f2_error"].mean()),
        "tp_error_mean": float(out["tp_error"].mean()),
        "fp_error_mean": float(out["fp_error"].mean()),
        "fn_error_mean": float(out["fn_error"].mean()),
        "tn_correct_mean": float(out["tn_correct"].mean()),
        "n_flagged_mean": float(out["n_flagged"].mean()),
    }
