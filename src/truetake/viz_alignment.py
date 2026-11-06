"""Save headless diagnostic plots using the same signals as the aligner."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from truetake.verify_gyro import prepare_signals, _interpolate, VIDEO_AXES
from truetake.calibration import OFFSET_AGREEMENT_S


PLOT_DPI = 120
MIN_PLOT_STD = 0.000001


def plot_alignment(table: pd.DataFrame, output_path: str) -> None:
    """Plot aligned traces, fixed-mapping lag curves, and validation windows."""
    diagnostics = table.iloc[0]["gyro_diagnostics"]
    baseline = diagnostics["baseline"]
    mapping = baseline["axis_mapping"]
    figure = Figure(figsize=(14, 12) if mapping else (10, 3), layout="constrained")
    FigureCanvasAgg(figure)
    if mapping is None:
        figure.text(0.05, 0.65, "Gyro alignment diagnostics", fontsize=18)
        figure.text(0.05, 0.4, baseline["reason"], fontsize=13)
        figure.text(0.05, 0.2, "No candidate calibration profile was generated.", fontsize=11)
    else:
        axes = figure.subplots(4, 2)
        grid, videos = prepare_signals(table)
        offset = baseline["offset_s"]
        global_result = diagnostics["global_fixed_mapping"]
        curve = global_result["lag_curve"]
        for index, axis in enumerate(VIDEO_AXES):
            pair = mapping[axis]
            gyro = _interpolate(table["t"].to_numpy(), table[pair["axis"]].to_numpy(), grid + offset) * pair["sign"]
            video = videos[index].copy()
            common = np.isfinite(video) & np.isfinite(gyro)
            for values, label in ((video, "Video"), (gyro, "Gyro at t + reference offset")):
                values[~common] = np.nan
                if common.sum() and np.std(values[common]) > MIN_PLOT_STD:
                    values[common] = (values[common] - np.mean(values[common])) / np.std(values[common])
                    axes[index, 0].plot(grid, values, label=label, linewidth=0.8)
            axes[index, 0].axvspan(*diagnostics["calibration_interval_s"], color="gray", alpha=0.15)
            axes[index, 0].set(title=f"{axis} ↔ {pair['sign']:+d} × {pair['axis']}",
                               xlabel="Clip time (s)", ylabel="Standard deviations")
            axes[index, 0].legend(fontsize=8)
            axes[index, 1].plot([point["lag_s"] for point in curve],
                                [point["axis_scores"][axis] for point in curve])
            axes[index, 1].axvline(offset, color="black", linestyle="--", label="Calibration reference")
            axes[index, 1].set(title="Whole-clip lag curve, mapping held fixed",
                               xlabel="Offset (s): compare v(t) with g(t + offset)", ylabel="Correlation", ylim=(-1.05, 1.05))
        windows = diagnostics["windows"]
        centers = [(window["start_s"] + window["end_s"]) / 2 for window in windows]
        axes[3, 0].plot(centers, [w["reference_score"] for w in windows], "o-", markersize=3, label="Fixed reference offset")
        axes[3, 0].plot(centers, [w["best_score"] for w in windows], ".--", label="Best offset per window")
        axes[3, 0].set(title="All windows: mapping held fixed", xlabel="Window midpoint (s)", ylabel="Correlation", ylim=(-1.05, 1.05))
        axes[3, 0].legend(fontsize=8)
        axes[3, 1].plot(centers, [w["best_offset_s"] for w in windows], "o-", markersize=3)
        axes[3, 1].axhline(offset, color="black", linestyle="--")
        axes[3, 1].axhspan(offset - OFFSET_AGREEMENT_S, offset + OFFSET_AGREEMENT_S, color="gray", alpha=0.15)
        axes[3, 1].set(title="Local offsets versus calibration reference", xlabel="Window midpoint (s)", ylabel="Best offset (s)", ylim=(-2.1, 2.1))
        for axis in axes.flat:
            axis.grid(alpha=0.2)
        figure.suptitle(f"Provisional alignment — whole-clip reference {offset:+.1f} s; strong axes only\n"
                       "Overlapping windows are diagnostics, not independent validation or an authenticity verdict", fontsize=12)
    figure.savefig(Path(output_path), dpi=PLOT_DPI)
