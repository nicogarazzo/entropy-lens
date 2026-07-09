"""Generate JSON reports from analysis results."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def save_report(
    model_name: str,
    records: list,
    fit_results: dict,
    go_nogo: str,
    output_dir: str,
) -> str:
    """Save analysis results as a JSON report.

    Args:
        model_name: HuggingFace model identifier.
        records: list of dicts with per-matrix results.
        fit_results: dict mapping epsilon label to FitResult.
        go_nogo: go/no-go verdict string.
        output_dir: directory to write the report.

    Returns:
        Path to the saved report file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_matrices": len(records),
        "go_nogo": go_nogo,
        "fits": {},
        "matrices": records,
    }

    for label, fit in fit_results.items():
        report["fits"][label] = {
            "slope": fit.slope,
            "intercept": fit.intercept,
            "r_squared": fit.r_squared,
            "p_value": fit.p_value,
            "n": fit.n,
            "c_constrained": fit.c_constrained,
            "rmse_log": fit.rmse_log,
        }

    # Convert numpy types to native Python for JSON serialization
    report = _convert_numpy(report)

    safe_name = model_name.replace("/", "_")
    fpath = out / f"report_{safe_name}.json"
    with open(fpath, "w") as f:
        json.dump(report, f, indent=2)

    return str(fpath)


def _convert_numpy(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
