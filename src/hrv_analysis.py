from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_ecg import load_ecg_segment
from preprocess_ecg import bandpass_filter
from detect_r_peaks import (
    calculate_detection_signal,
    detect_qrs_candidates,
    refine_r_peaks,
)
from heart_rate_analysis import calculate_heart_rate


# ============================================================
# PROJECT SETTINGS
# ============================================================

DATABASE = "mitdb"
RECORD_NAME = "100"
CHANNEL_INDEX = 0

START_TIME_SECONDS = 0.0
DURATION_SECONDS = 10.0

LOW_CUTOFF_HZ = 0.5
HIGH_CUTOFF_HZ = 40.0
FILTER_ORDER = 4


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# HRV CALCULATIONS
# ============================================================

def calculate_hrv_metrics(rr_intervals):
    """
    Calculate basic time-domain heart-rate-variability metrics.

    Parameters
    ----------
    rr_intervals:
        Consecutive RR intervals in seconds.

    Returns
    -------
    mean_rr_ms:
        Mean RR interval in milliseconds.

    sdnn_ms:
        Standard deviation of RR intervals in milliseconds.

    rmssd_ms:
        Root mean square of successive RR-interval differences
        in milliseconds.
    """

    if len(rr_intervals) < 2:
        raise ValueError(
            "At least two RR intervals are required "
            "for HRV calculations."
        )

    # --------------------------------------------------------
    # MEAN RR
    # --------------------------------------------------------

    mean_rr_seconds = np.mean(
        rr_intervals
    )

    # --------------------------------------------------------
    # SDNN
    # --------------------------------------------------------
    #
    # ddof=1 calculates the sample standard deviation.

    sdnn_seconds = np.std(
        rr_intervals,
        ddof=1,
    )

    # --------------------------------------------------------
    # RMSSD
    # --------------------------------------------------------
    #
    # First calculate the difference between consecutive
    # RR intervals:
    #
    # RR2 - RR1
    # RR3 - RR2
    # ...

    successive_differences = np.diff(
        rr_intervals
    )

    # Square each difference, calculate the mean,
    # then take the square root.

    rmssd_seconds = np.sqrt(
        np.mean(
            successive_differences ** 2
        )
    )

    # Convert seconds to milliseconds.

    mean_rr_ms = (
        mean_rr_seconds * 1000.0
    )

    sdnn_ms = (
        sdnn_seconds * 1000.0
    )

    rmssd_ms = (
        rmssd_seconds * 1000.0
    )

    return (
        mean_rr_ms,
        sdnn_ms,
        rmssd_ms,
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_hrv_summary(
    rr_intervals,
    mean_rr_ms,
    sdnn_ms,
    rmssd_ms,
):
    """Print the calculated HRV metrics."""

    print("\n" + "=" * 60)
    print("BASIC HEART RATE VARIABILITY ANALYSIS")
    print("=" * 60)

    print(
        f"RR intervals analyzed:    "
        f"{len(rr_intervals)}"
    )

    print(
        f"Mean RR interval:          "
        f"{mean_rr_ms:.2f} ms"
    )

    print(
        f"SDNN:                      "
        f"{sdnn_ms:.2f} ms"
    )

    print(
        f"RMSSD:                     "
        f"{rmssd_ms:.2f} ms"
    )

    print()
    print(
        "Note: These values are calculated from a short "
        "ECG segment and are used for engineering analysis only."
    )

    print("=" * 60)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_hrv_results(
    mean_rr_ms,
    sdnn_ms,
    rmssd_ms,
    output_path,
):
    """Save HRV metrics to a CSV file."""

    results = pd.DataFrame(
        [
            {
                "record": RECORD_NAME,
                "segment_duration_s": DURATION_SECONDS,
                "mean_rr_ms": mean_rr_ms,
                "sdnn_ms": sdnn_ms,
                "rmssd_ms": rmssd_ms,
            }
        ]
    )

    results.to_csv(
        output_path,
        index=False,
    )

    return results


# ============================================================
# RR INTERVAL PLOT
# ============================================================

def plot_rr_intervals(
    heart_rate_times,
    rr_intervals,
    output_path,
):
    """
    Plot RR intervals over time.
    """

    rr_intervals_ms = (
        rr_intervals * 1000.0
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        heart_rate_times,
        rr_intervals_ms,
        marker="o",
        linewidth=1.5,
    )

    ax.set_title(
        f"MIT-BIH Record {RECORD_NAME} — "
        "RR Interval Series"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "RR Interval (ms)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading ECG signal...")

    (
        record,
        annotations,
        raw_signal,
        time_s,
    ) = load_ecg_segment(
        record_name=RECORD_NAME,
        database=DATABASE,
        channel_index=CHANNEL_INDEX,
        start_time_s=START_TIME_SECONDS,
        duration_s=DURATION_SECONDS,
    )

    sampling_frequency = float(
        record.fs
    )

    # --------------------------------------------------------
    # FILTER ECG
    # --------------------------------------------------------

    print("Filtering ECG...")

    filtered_signal = bandpass_filter(
        signal=raw_signal,
        sampling_frequency=sampling_frequency,
        low_cutoff_hz=LOW_CUTOFF_HZ,
        high_cutoff_hz=HIGH_CUTOFF_HZ,
        order=FILTER_ORDER,
    )

    # --------------------------------------------------------
    # DETECT R-PEAKS
    # --------------------------------------------------------

    (
        derivative,
        squared_signal,
        integrated_signal,
    ) = calculate_detection_signal(
        filtered_signal=filtered_signal,
        sampling_frequency=sampling_frequency,
    )

    (
        candidate_peaks,
        threshold,
        properties,
    ) = detect_qrs_candidates(
        integrated_signal=integrated_signal,
        sampling_frequency=sampling_frequency,
    )

    r_peaks = refine_r_peaks(
        candidate_peaks=candidate_peaks,
        filtered_signal=filtered_signal,
        sampling_frequency=sampling_frequency,
    )

    # --------------------------------------------------------
    # CALCULATE RR INTERVALS
    # --------------------------------------------------------

    (
        beat_times,
        rr_intervals,
        heart_rate_bpm,
        heart_rate_times,
    ) = calculate_heart_rate(
        r_peaks=r_peaks,
        sampling_frequency=sampling_frequency,
        start_time_seconds=START_TIME_SECONDS,
    )

    # --------------------------------------------------------
    # CALCULATE HRV
    # --------------------------------------------------------

    (
        mean_rr_ms,
        sdnn_ms,
        rmssd_ms,
    ) = calculate_hrv_metrics(
        rr_intervals=rr_intervals,
    )

    print_hrv_summary(
        rr_intervals=rr_intervals,
        mean_rr_ms=mean_rr_ms,
        sdnn_ms=sdnn_ms,
        rmssd_ms=rmssd_ms,
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_path = (
        RESULTS_DIR
        / f"record_{RECORD_NAME}_hrv.csv"
    )

    save_hrv_results(
        mean_rr_ms=mean_rr_ms,
        sdnn_ms=sdnn_ms,
        rmssd_ms=rmssd_ms,
        output_path=results_path,
    )

    # --------------------------------------------------------
    # SAVE RR INTERVAL FIGURE
    # --------------------------------------------------------

    figure_path = (
        FIGURES_DIR
        / f"record_{RECORD_NAME}_rr_intervals.png"
    )

    plot_rr_intervals(
        heart_rate_times=heart_rate_times,
        rr_intervals=rr_intervals,
        output_path=figure_path,
    )

    print(
        f"\nResults saved to:\n{results_path}"
    )

    print(
        f"\nFigure saved to:\n{figure_path}"
    )


if __name__ == "__main__":
    main()