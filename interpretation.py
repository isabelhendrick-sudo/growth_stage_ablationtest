"""Report decision rules; independent of model fitting and numerical dependencies."""

BASELINE_LOWER_BOUND = 0.60
MIN_DELTA = 0.01
GATED_RUN = "leaf_crop"


def interval_verdict(lo, hi, margin=MIN_DELTA):
    """Describe a paired interval without overriding the baseline gate."""
    if lo > margin:
        return "HELPS (gain exceeds practical margin)"
    if lo > 0:
        return "POSITIVE (practical gain not established)"
    if hi < -margin:
        return "HURTS (loss exceeds practical margin)"
    if lo >= -margin and hi <= margin:
        return "EQUIVALENT (within practical margin)"
    return "INCONCLUSIVE (interval spans competing conclusions)"


def interpret_results(overall, deltas, run_name):
    """Accept record lists; gate run 2 before interpreting any stage comparison."""
    if run_name not in ("full_image", "leaf_crop"):
        raise ValueError("run_name must be full_image or leaf_crop")
    baseline = next(row for row in overall if row["arm"] == "no_stage")
    lower = float(baseline["ci_lo"])
    gate_applies = run_name == GATED_RUN
    gate_passed = lower > BASELINE_LOWER_BOUND
    status = "not_interpreted" if gate_applies and not gate_passed else "interpreted"
    result = {
        "run": run_name,
        "baseline_macro_f1": float(baseline["macro_f1"]),
        "baseline_ci_lo": lower,
        "baseline_ci_hi": float(baseline["ci_hi"]),
        "baseline_ci_level": 0.95,
        "baseline_lower_bound_threshold": BASELINE_LOWER_BOUND,
        "baseline_gate_applies": gate_applies,
        "baseline_above_threshold": gate_passed,
        "gate_chronology": "Added after run 1, before run 2; not prespecified for run 1.",
        "minimum_meaningful_delta": MIN_DELTA,
        "stage_comparison_status": status,
    }
    if status == "not_interpreted":
        result["decision"] = (
            "NOT INTERPRETED: the image-only lower 95% CI bound "
            f"({lower:.3f}) does not exceed {BASELINE_LOWER_BOUND:.2f}. "
            "Report stage comparisons for completeness only; "
            "do not keep or drop stage conditioning."
        )
        result["verdicts"] = ["NOT INTERPRETED (baseline gate failed)"] * len(deltas)
        return result
    result["verdicts"] = [
        interval_verdict(float(row["ci_lo"]), float(row["ci_hi"])) for row in deltas
    ]
    main = [row for row in deltas if row["comparison"].endswith(" vs no_stage")]
    if len(main) != 2:
        raise ValueError("Expected the two main stage-versus-image-only comparisons")
    if any(float(row["ci_lo"]) > MIN_DELTA for row in main):
        decision = "A main comparison supports a gain exceeding 0.01 macro-F1."
    elif all(-MIN_DELTA <= float(row["ci_lo"]) <= float(row["ci_hi"]) <= MIN_DELTA for row in main):
        decision = "Both main comparisons fall within the practical-equivalence margin."
    else:
        decision = "INCONCLUSIVE: no basis for keeping or dropping stage conditioning."
    if not gate_applies:
        decision += " Run 1 is descriptive; the baseline gate was added afterwards."
    result["decision"] = decision
    return result
