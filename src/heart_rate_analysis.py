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
# HEART RATE CALCULATIONS
# ============================================================

def calculate_heart_rate(
    r_peaks,
    sampling_frequency,
    start_time_seconds=0.0,
):
    """
    Calculate heartbeat times, RR intervals,
    and instantaneous heart rate.

    Returns
    -------
    beat_times:
        Time of every detected R-peak.

    rr_intervals:
        Time between consecutive R-peaks.

    heart_rate_bpm:
        Instantaneous heart rate calculated from each RR interval.

    heart_rate_times:
        Time associated with each instantaneous heart-rate value.
    """

    if len(r_peaks) < 2:
        raise ValueError(
            "At least two R-peaks are required "
            "to calculate RR intervals."
        )

    # --------------------------------------------------------
    # R-PEAK SAMPLE NUMBER -> TIME
    # --------------------------------------------------------

    beat_times = (
        start_time_seconds
        + r_peaks / sampling_frequency
    )

    # --------------------------------------------------------
    # RR INTERVALS
    # --------------------------------------------------------
    #
    # np.diff calculates:
    #
    # R2 - R1
    # R3 - R2
    # R4 - R3
    # ...
    #
    # Dividing by sampling frequency converts
    # sample differences into seconds.

    rr_intervals = (
        np.diff(r_peaks)
        / sampling_frequency
    )

    # --------------------------------------------------------
    # INSTANTANEOUS HEART RATE
    # --------------------------------------------------------
    #
    # Heart rate in beats/minute:
    #
    # HR = 60 / RR

    heart_rate_bpm = (
        60.0 / rr_intervals
    )

    # Each RR interval ends at the second heartbeat,
    # so associate each HR value with that R-peak time.
    heart_rate_times = beat_times[1:]

    return (
        beat_times,
        rr_intervals,
        heart_rate_bpm,
        heart_rate_times,
    )


# ============================================================
# SUMMARY
# ============================================================

def print_heart_rate_summary(
    beat_times,
    rr_intervals,
    heart_rate_bpm,
):
    """Print calculated RR and heart-rate results."""

    mean_rr = np.mean(rr_intervals)

    # Heart rate corresponding to the overall mean RR interval.
    average_heart_rate = (
        60.0 / mean_rr
    )

    minimum_heart_rate = np.min(
        heart_rate_bpm
    )

    maximum_heart_rate = np.max(
        heart_rate_bpm
    )

    print("\n" + "=" * 60)
    print("HEART RATE ANALYSIS")
    print("=" * 60)

    print(
        f"Detected heartbeats:      "
        f"{len(beat_times)}"
    )

    print(
        f"RR intervals calculated:  "
        f"{len(rr_intervals)}"
    )

    print(
        f"Mean RR interval:          "
        f"{mean_rr:.3f} s"
    )

    print(
        f"Average heart rate:        "
        f"{average_heart_rate:.1f} bpm"
    )

    print(
        f"Minimum heart rate:        "
        f"{minimum_heart_rate:.1f} bpm"
    )

    print(
        f"Maximum heart rate:        "
        f"{maximum_heart_rate:.1f} bpm"
    )

    print("\nIndividual RR intervals:")

    for i, rr in enumerate(
        rr_intervals,
        start=1,
    ):

        print(
            f"  RR {i:2d}: "
            f"{rr:.3f} s | "
            f"{heart_rate_bpm[i - 1]:.1f} bpm"
        )

    print("=" * 60)


# ============================================================
# SAVE RESULTS TABLE
# ============================================================

def save_heart_rate_results(
    heart_rate_times,
    rr_intervals,
    heart_rate_bpm,
    output_path,
):
    """
    Save RR intervals and heart-rate values to a CSV file.
    """

    results = pd.DataFrame(
        {
            "time_s": heart_rate_times,
            "rr_interval_s": rr_intervals,
            "heart_rate_bpm": heart_rate_bpm,
        }
    )

    results.to_csv(
        output_path,
        index=False,
    )

    return results


# ============================================================
# HEART RATE PLOT
# ============================================================

def plot_heart_rate(
    heart_rate_times,
    heart_rate_bpm,
    output_path,
):
    """Plot instantaneous heart rate over time."""

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        heart_rate_times,
        heart_rate_bpm,
        marker="o",
        linewidth=1.5,
    )

    ax.set_title(
        f"MIT-BIH Record {RECORD_NAME} — "
        "Instantaneous Heart Rate"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Heart Rate (bpm)"
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
    # CREATE DETECTION SIGNAL
    # --------------------------------------------------------

    (
        derivative,
        squared_signal,
        integrated_signal,
    ) = calculate_detection_signal(
        filtered_signal=filtered_signal,
        sampling_frequency=sampling_frequency,
    )

    # --------------------------------------------------------
    # DETECT HEARTBEATS
    # --------------------------------------------------------

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
    # HEART RATE CALCULATIONS
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

    print_heart_rate_summary(
        beat_times=beat_times,
        rr_intervals=rr_intervals,
        heart_rate_bpm=heart_rate_bpm,
    )

    # --------------------------------------------------------
    # SAVE NUMERICAL RESULTS
    # --------------------------------------------------------

    results_path = (
        RESULTS_DIR
        / f"record_{RECORD_NAME}_heart_rate.csv"
    )

    save_heart_rate_results(
        heart_rate_times=heart_rate_times,
        rr_intervals=rr_intervals,
        heart_rate_bpm=heart_rate_bpm,
        output_path=results_path,
    )

    # --------------------------------------------------------
    # SAVE HEART-RATE FIGURE
    # --------------------------------------------------------

    figure_path = (
        FIGURES_DIR
        / f"record_{RECORD_NAME}_heart_rate.png"
    )

    plot_heart_rate(
        heart_rate_times=heart_rate_times,
        heart_rate_bpm=heart_rate_bpm,
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