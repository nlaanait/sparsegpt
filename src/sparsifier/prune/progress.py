"""Small, opt-in numerical progress events shared by calibration and solvers."""
from contextlib import contextmanager
import json
import math
import time


def scoped(progress, **context):
    if progress is None:
        return None
    return lambda record: progress(dict(context, **record))


def compact_diagnostics(values):
    """Keep numerical summaries, excluding masks, IDs and per-sample tensors."""
    result = {}
    arrays = {"chart_fit_counts", "chart_eval_counts", "chart_retained_ranks",
              "scale_quantiles", "scale_quantile_probabilities", "unsupported_factors"}
    for key, value in values.items():
        if key.endswith(("_sha256", "_ids", "_indices")):
            continue
        if isinstance(value, dict):
            result[key] = compact_diagnostics(value)
        elif isinstance(value, (str, bool, int, float)) or value is None:
            result[key] = str(value) if isinstance(value, float) and not math.isfinite(value) else value
        elif key in arrays:
            result[key] = value
    return result


class Progress:
    def __init__(self, **context):
        self.context = context
        self.started = time.monotonic()

    def __call__(self, record):
        print(json.dumps(dict(self.context, elapsed_seconds=time.monotonic()-self.started,
                              **record), allow_nan=False), flush=True)

    @contextmanager
    def phase(self, name):
        start = time.monotonic()
        self({"event": "phase_started", "phase": name})
        try:
            yield
        except Exception as error:
            self({"event": "phase_failed", "phase": name,
                  "phase_seconds": time.monotonic()-start,
                  "error_type": type(error).__name__, "error": str(error)})
            raise
        self({"event": "phase_completed", "phase": name,
              "phase_seconds": time.monotonic()-start})
