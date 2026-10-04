from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from load_ecg import load_ecg_segment
from preprocess_ecg import bandpass_filter
from detect_r_peaks import (
    calculate_detection_signal,
    detect_qrs_candidates,
    refine_r_peaks,
)
from validate_detector import (
    extract_reference_beats,
    match_detections,
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

DATABASE = "mitdb"
RECORD_NAME = "207"
CHANNEL_INDEX = 0

START_TIME_SECONDS = 0.0
DURATION_SECONDS = 60.0

LOW_CUTOFF_HZ = 0.5
HIGH_CUTOFF_HZ = 40.0
FILTER_ORDER = 4

MATCH_TOLERANCE_SECONDS = 0.050

# Automatically inspect the 10-second section
# containing the most false-positive detections.
ERROR_WINDOW_SECONDS = 10.0


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"


# ============================================================
# FIND WINDOW WITH MOST FALSE POSITIVES
# ============================================================

def find_false_positive_window(
    false_positive_peaks,
    signal_length,
    sampling_frequency,
    window_seconds=10.0,
):
    """
    Find the time window containing the largest number
    of false-positive detections.
    """

    window_samples = int(
        round(
            window_seconds * sampling_frequency
        )
    )

    if len(false_positive_peaks) == 0:
        return (
            0,
            min(window_samples, signal_length),
        )

    best_start = 0
    best_count = -1

    for false_peak in false_positive_peaks:

        candidate_start = int(
            false_peak - window_samples // 2
        )

        candidate_start = max(
            candidate_start,
            0,
        )

        candidate_stop = (
            candidate_start + window_samples
        )

        if candidate_stop > signal_length:

            candidate_stop = signal_length

            candidate_start = max(
                0,
                candidate_stop - window_samples,
            )

        count = np.sum(
            (
                false_positive_peaks
                >= candidate_start
            )
            &
            (
                false_positive_peaks
                < candidate_stop
            )
        )

        if count > best_count:

            best_count = count
            best_start = candidate_start

    best_stop = min(
        best_start + window_samples,
        signal_length,
    )

    return (
        best_start,
        best_stop,
    )


# ============================================================
# PRINT ALL EXPERT ANNOTATIONS IN WINDOW
# ============================================================

def print_annotations_in_window(
    annotations,
    sampling_frequency,
    window_start,
    window_stop,
):
    """
    Print every expert annotation symbol inside the
    selected inspection window.

    This helps determine whether unusual rhythm/event
    annotations are present in the failure region.
    """

    print("\nEXPERT ANNOTATIONS IN INSPECTION WINDOW")
    print("-" * 60)

    annotation_count = 0

    for sample, symbol in zip(
        annotations.sample,
        annotations.symbol,
    ):

        if window_start <= sample < window_stop:

            annotation_time = (
                START_TIME_SECONDS
                + sample / sampling_frequency
            )

            print(
                f"Time {annotation_time:7.3f} s | "
                f"Sample {sample:5d} | "
                f"Symbol {symbol}"
            )

            annotation_count += 1

    if annotation_count == 0:

        print(
            "No expert annotations found "
            "inside this window."
        )

    print("-" * 60)


# ============================================================
# PRINT ERROR SUMMARY
# ============================================================

def print_error_summary(
    reference_peaks,
    detected_peaks,
    false_positive_peaks,
    missed_reference_peaks,
    integrated_signal,
    threshold,
    filtered_signal,
    sampling_frequency,
    window_start,
    window_stop,
):
    """
    Print diagnostic information for Record 207.
    """

    start_time = (
        START_TIME_SECONDS
        + window_start / sampling_frequency
    )

    stop_time = (
        START_TIME_SECONDS
        + window_stop / sampling_frequency
    )

    false_in_window = false_positive_peaks[
        (
            false_positive_peaks
            >= window_start
        )
        &
        (
            false_positive_peaks
            < window_stop
        )
    ]

    if len(false_positive_peaks) > 0:

        false_peak_amplitudes = np.abs(
            filtered_signal[
                false_positive_peaks
            ]
        )

        median_false_amplitude = np.median(
            false_peak_amplitudes
        )

    else:

        median_false_amplitude = np.nan

    print("\n" + "=" * 65)
    print("RECORD 207 FALSE-POSITIVE ERROR ANALYSIS")
    print("=" * 65)

    print(
        f"Reference heartbeats:          "
        f"{len(reference_peaks)}"
    )

    print(
        f"Detected heartbeats:           "
        f"{len(detected_peaks)}"
    )

    print(
        f"False-positive detections:     "
        f"{len(false_positive_peaks)}"
    )

    print(
        f"Missed reference beats:        "
        f"{len(missed_reference_peaks)}"
    )

    print()

    print(
        f"Detection threshold:           "
        f"{threshold:.2f}"
    )

    print(
        f"Median QRS energy:             "
        f"{np.median(integrated_signal):.2f}"
    )

    print(
        f"95th percentile QRS energy:    "
        f"{np.percentile(integrated_signal, 95):.2f}"
    )

    print(
        f"99th percentile QRS energy:    "
        f"{np.percentile(integrated_signal, 99):.2f}"
    )

    print()

    print(
        f"Median |amplitude| of false "
        f"detections: "
        f"{median_false_amplitude:.3f} mV"
    )

    print()

    print(
        f"Inspection window:             "
        f"{start_time:.2f}–{stop_time:.2f} s"
    )

    print(
        f"False positives in window:     "
        f"{len(false_in_window)}"
    )

    print("=" * 65)


# ============================================================
# ERROR ANALYSIS PLOT
# ============================================================

def plot_error_analysis(
    time_s,
    raw_signal,
    filtered_signal,
    integrated_signal,
    threshold,
    reference_peaks,
    detected_peaks,
    false_positive_peaks,
    missed_reference_peaks,
    sampling_frequency,
    window_start,
    window_stop,
    output_path,
):
    """
    Plot the region containing the greatest concentration
    of false-positive detections.
    """

    window_slice = slice(
        window_start,
        window_stop,
    )

    window_time = time_s[
        window_slice
    ]

    # --------------------------------------------------------
    # SELECT EVENTS WITHIN WINDOW
    # --------------------------------------------------------

    reference_in_window = reference_peaks[
        (
            reference_peaks >= window_start
        )
        &
        (
            reference_peaks < window_stop
        )
    ]

    detected_in_window = detected_peaks[
        (
            detected_peaks >= window_start
        )
        &
        (
            detected_peaks < window_stop
        )
    ]

    false_in_window = false_positive_peaks[
        (
            false_positive_peaks >= window_start
        )
        &
        (
            false_positive_peaks < window_stop
        )
    ]

    missed_in_window = missed_reference_peaks[
        (
            missed_reference_peaks >= window_start
        )
        &
        (
            missed_reference_peaks < window_stop
        )
    ]

    # Convert sample indices to seconds.

    reference_times = (
        START_TIME_SECONDS
        + reference_in_window
        / sampling_frequency
    )

    detected_times = (
        START_TIME_SECONDS
        + detected_in_window
        / sampling_frequency
    )

    false_times = (
        START_TIME_SECONDS
        + false_in_window
        / sampling_frequency
    )

    missed_times = (
        START_TIME_SECONDS
        + missed_in_window
        / sampling_frequency
    )

    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        sharex=True,
    )

    # --------------------------------------------------------
    # 1. RAW ECG
    # --------------------------------------------------------

    axes[0].plot(
        window_time,
        raw_signal[window_slice],
        linewidth=1.0,
    )

    axes[0].set_title(
        "1. Raw ECG"
    )

    axes[0].set_ylabel(
        "Amplitude (mV)"
    )

    # --------------------------------------------------------
    # 2. FILTERED ECG + DETECTIONS
    # --------------------------------------------------------

    axes[1].plot(
        window_time,
        filtered_signal[window_slice],
        linewidth=1.0,
        label="Filtered ECG",
    )

    if len(reference_in_window) > 0:

        axes[1].scatter(
            reference_times,
            filtered_signal[
                reference_in_window
            ],
            marker="o",
            s=80,
            facecolors="none",
            linewidths=1.5,
            label="Expert reference beats",
            zorder=3,
        )

    if len(detected_in_window) > 0:

        axes[1].scatter(
            detected_times,
            filtered_signal[
                detected_in_window
            ],
            marker="x",
            s=55,
            linewidths=1.5,
            label="All detector outputs",
            zorder=4,
        )

    # False positives
    for peak_time in false_times:

        axes[1].axvline(
            peak_time,
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
        )

    # Missed beats
    for peak_time in missed_times:

        axes[1].axvline(
            peak_time,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )

    axes[1].set_title(
        "2. Filtered ECG — Reference Beats vs Detector Outputs"
    )

    axes[1].set_ylabel(
        "Amplitude (mV)"
    )

    axes[1].legend()

    # --------------------------------------------------------
    # 3. QRS ENERGY
    # --------------------------------------------------------

    axes[2].plot(
        window_time,
        integrated_signal[
            window_slice
        ],
        linewidth=1.1,
        label="QRS detection signal",
    )

    axes[2].axhline(
        threshold,
        linestyle="--",
        linewidth=1.3,
        label="Detection threshold",
    )

    for peak_time in false_times:

        axes[2].axvline(
            peak_time,
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
        )

    axes[2].set_title(
        "3. QRS Energy — False Positive Locations"
    )

    axes[2].set_xlabel(
        "Time (s)"
    )

    axes[2].set_ylabel(
        "QRS Energy"
    )

    axes[2].legend()

    # --------------------------------------------------------
    # FORMATTING
    # --------------------------------------------------------

    for ax in axes:

        ax.grid(
            True,
            alpha=0.3,
        )

    axes[2].set_xlim(
        window_time[0],
        window_time[-1],
    )

    fig.suptitle(
        "MIT-BIH Record 207 — False-Positive Error Analysis",
        fontsize=14,
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

    print(
        "\nLoading Record 207..."
    )

    # --------------------------------------------------------
    # LOAD ECG + ALL EXPERT ANNOTATIONS
    # --------------------------------------------------------

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

    filtered_signal = bandpass_filter(
        signal=raw_signal,
        sampling_frequency=sampling_frequency,
        low_cutoff_hz=LOW_CUTOFF_HZ,
        high_cutoff_hz=HIGH_CUTOFF_HZ,
        order=FILTER_ORDER,
    )

    # --------------------------------------------------------
    # BUILD QRS DETECTION SIGNAL
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
    # RUN CURRENT DETECTOR
    # --------------------------------------------------------

    (
        candidate_peaks,
        threshold,
        properties,
    ) = detect_qrs_candidates(
        integrated_signal=integrated_signal,
        sampling_frequency=sampling_frequency,
    )

    detected_peaks = refine_r_peaks(
        candidate_peaks=candidate_peaks,
        filtered_signal=filtered_signal,
        sampling_frequency=sampling_frequency,
    )

    # --------------------------------------------------------
    # EXTRACT BEAT-TYPE EXPERT ANNOTATIONS
    # --------------------------------------------------------

    reference_peaks = extract_reference_beats(
        annotations
    )

    # --------------------------------------------------------
    # MATCH DETECTIONS TO REFERENCE BEATS
    # --------------------------------------------------------

    (
        matches,
        true_positives,
        false_positives,
        false_negatives,
        missed_reference_peaks,
        false_positive_peaks,
    ) = match_detections(
        reference_peaks=reference_peaks,
        detected_peaks=detected_peaks,
        sampling_frequency=sampling_frequency,
        tolerance_seconds=MATCH_TOLERANCE_SECONDS,
    )

    # --------------------------------------------------------
    # FIND REGION WITH MOST FALSE POSITIVES
    # --------------------------------------------------------

    (
        window_start,
        window_stop,
    ) = find_false_positive_window(
        false_positive_peaks=false_positive_peaks,
        signal_length=len(raw_signal),
        sampling_frequency=sampling_frequency,
        window_seconds=ERROR_WINDOW_SECONDS,
    )

    # --------------------------------------------------------
    # PRINT ERROR SUMMARY
    # --------------------------------------------------------

    print_error_summary(
        reference_peaks=reference_peaks,
        detected_peaks=detected_peaks,
        false_positive_peaks=false_positive_peaks,
        missed_reference_peaks=missed_reference_peaks,
        integrated_signal=integrated_signal,
        threshold=threshold,
        filtered_signal=filtered_signal,
        sampling_frequency=sampling_frequency,
        window_start=window_start,
        window_stop=window_stop,
    )

    # --------------------------------------------------------
    # PRINT ALL EXPERT ANNOTATION TYPES
    # --------------------------------------------------------

    print_annotations_in_window(
        annotations=annotations,
        sampling_frequency=sampling_frequency,
        window_start=window_start,
        window_stop=window_stop,
    )

    # --------------------------------------------------------
    # CREATE ERROR-ANALYSIS FIGURE
    # --------------------------------------------------------

    figure_path = (
        FIGURES_DIR
        / "record_207_error_analysis.png"
    )

    plot_error_analysis(
        time_s=time_s,
        raw_signal=raw_signal,
        filtered_signal=filtered_signal,
        integrated_signal=integrated_signal,
        threshold=threshold,
        reference_peaks=reference_peaks,
        detected_peaks=detected_peaks,
        false_positive_peaks=false_positive_peaks,
        missed_reference_peaks=missed_reference_peaks,
        sampling_frequency=sampling_frequency,
        window_start=window_start,
        window_stop=window_stop,
        output_path=figure_path,
    )

    print(
        f"\nFigure saved to:\n"
        f"{figure_path}"
    )


if __name__ == "__main__":
    main()