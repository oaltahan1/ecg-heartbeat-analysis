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
from validate_detector import (
    extract_reference_beats,
    match_detections,
    calculate_metrics,
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

DATABASE = "mitdb"

RECORD_NAMES = [
    "100",
    "105",
    "108",
    "200",
    "207",
]

CHANNEL_INDEX = 0

START_TIME_SECONDS = 0.0
DURATION_SECONDS = 60.0

LOW_CUTOFF_HZ = 0.5
HIGH_CUTOFF_HZ = 40.0
FILTER_ORDER = 4

MATCH_TOLERANCE_SECONDS = 0.050


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


# ============================================================
# VENTRICULAR-FLUTTER INTERVAL EXTRACTION
# ============================================================

def extract_flutter_intervals(
    annotations,
    signal_length,
):
    """
    Extract ventricular-flutter intervals marked by
    '[' and ']' annotations.

    These regions are excluded from discrete heartbeat
    detection scoring because individual R-peaks may not be
    meaningfully defined during ventricular flutter.

    Returns
    -------
    intervals:
        List of (start_sample, stop_sample) pairs.

        stop_sample is treated as exclusive.
    """

    intervals = []

    active_start = None

    for sample, symbol in zip(
        annotations.sample,
        annotations.symbol,
    ):

        sample = int(sample)

        # Start of ventricular flutter/fibrillation interval.
        if symbol == "[":

            if active_start is None:
                active_start = sample

        # End of ventricular flutter/fibrillation interval.
        elif symbol == "]":

            # If the recording segment began while already
            # inside a flutter interval, exclude from sample 0.
            if active_start is None:
                start = 0
            else:
                start = active_start

            stop = min(
                sample + 1,
                signal_length,
            )

            intervals.append(
                (
                    max(start, 0),
                    stop,
                )
            )

            active_start = None

    # If a flutter interval begins but does not end before
    # our 60-second analysis segment finishes, exclude through
    # the end of the loaded signal.
    if active_start is not None:

        intervals.append(
            (
                max(active_start, 0),
                signal_length,
            )
        )

    # --------------------------------------------------------
    # MERGE OVERLAPPING INTERVALS
    # --------------------------------------------------------

    if len(intervals) == 0:
        return []

    intervals.sort(
        key=lambda interval: interval[0]
    )

    merged = [
        list(intervals[0])
    ]

    for start, stop in intervals[1:]:

        previous_start, previous_stop = merged[-1]

        if start <= previous_stop:

            merged[-1][1] = max(
                previous_stop,
                stop,
            )

        else:

            merged.append(
                [start, stop]
            )

    return [
        (start, stop)
        for start, stop in merged
    ]


# ============================================================
# EXCLUDE SAMPLES INSIDE NON-EVALUABLE REGIONS
# ============================================================

def remove_samples_inside_intervals(
    samples,
    excluded_intervals,
):
    """
    Remove peak locations that occur inside excluded intervals.
    """

    samples = np.asarray(
        samples,
        dtype=int,
    )

    if (
        len(samples) == 0
        or len(excluded_intervals) == 0
    ):
        return samples.copy()

    keep_mask = np.ones(
        len(samples),
        dtype=bool,
    )

    for start, stop in excluded_intervals:

        inside_interval = (
            (samples >= start)
            &
            (samples < stop)
        )

        keep_mask &= ~inside_interval

    return samples[
        keep_mask
    ]


# ============================================================
# EXCLUDED DURATION
# ============================================================

def calculate_excluded_duration(
    excluded_intervals,
    sampling_frequency,
):
    """
    Calculate the total duration of excluded ECG regions.
    """

    excluded_samples = sum(
        stop - start
        for start, stop in excluded_intervals
    )

    excluded_seconds = (
        excluded_samples
        / sampling_frequency
    )

    return excluded_seconds


# ============================================================
# VALIDATE ONE RECORD
# ============================================================

def validate_record(
    record_name,
):
    """
    Run the complete ECG detection and validation workflow
    for one MIT-BIH record.
    """

    print("\n" + "=" * 70)
    print(
        f"PROCESSING RECORD {record_name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD ECG + ANNOTATIONS
    # --------------------------------------------------------

    (
        record,
        annotations,
        raw_signal,
        time_s,
    ) = load_ecg_segment(
        record_name=record_name,
        database=DATABASE,
        channel_index=CHANNEL_INDEX,
        start_time_s=START_TIME_SECONDS,
        duration_s=DURATION_SECONDS,
    )

    sampling_frequency = float(
        record.fs
    )

    signal_length = len(
        raw_signal
    )

    lead_name = record.sig_name[0]

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
    # DETECT QRS CANDIDATES
    # --------------------------------------------------------

    (
        candidate_peaks,
        threshold,
        properties,
    ) = detect_qrs_candidates(
        integrated_signal=integrated_signal,
        sampling_frequency=sampling_frequency,
    )

    # --------------------------------------------------------
    # REFINE TO R-PEAKS
    # --------------------------------------------------------

    detected_peaks_all = refine_r_peaks(
        candidate_peaks=candidate_peaks,
        filtered_signal=filtered_signal,
        sampling_frequency=sampling_frequency,
    )

    # --------------------------------------------------------
    # EXPERT REFERENCE HEARTBEATS
    # --------------------------------------------------------

    reference_peaks_all = extract_reference_beats(
        annotations
    )

    # --------------------------------------------------------
    # FIND NON-EVALUABLE FLUTTER INTERVALS
    # --------------------------------------------------------

    excluded_intervals = extract_flutter_intervals(
        annotations=annotations,
        signal_length=signal_length,
    )

    excluded_duration = calculate_excluded_duration(
        excluded_intervals=excluded_intervals,
        sampling_frequency=sampling_frequency,
    )

    evaluable_duration = (
        DURATION_SECONDS
        - excluded_duration
    )

    # --------------------------------------------------------
    # REMOVE EVENTS INSIDE EXCLUDED REGIONS
    # --------------------------------------------------------

    reference_peaks = remove_samples_inside_intervals(
        samples=reference_peaks_all,
        excluded_intervals=excluded_intervals,
    )

    detected_peaks = remove_samples_inside_intervals(
        samples=detected_peaks_all,
        excluded_intervals=excluded_intervals,
    )

    # --------------------------------------------------------
    # MATCH DETECTED VS REFERENCE BEATS
    # --------------------------------------------------------

    (
        matches,
        true_positives,
        false_positives,
        false_negatives,
        unmatched_reference_peaks,
        unmatched_detected_peaks,
    ) = match_detections(
        reference_peaks=reference_peaks,
        detected_peaks=detected_peaks,
        sampling_frequency=sampling_frequency,
        tolerance_seconds=MATCH_TOLERANCE_SECONDS,
    )

    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------

    (
        sensitivity,
        precision,
        f1_score,
    ) = calculate_metrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Lead:                         "
        f"{lead_name}"
    )

    print(
        f"Total analyzed duration:      "
        f"{DURATION_SECONDS:.2f} s"
    )

    print(
        f"Excluded flutter duration:    "
        f"{excluded_duration:.2f} s"
    )

    print(
        f"Evaluable duration:           "
        f"{evaluable_duration:.2f} s"
    )

    print(
        f"Flutter intervals excluded:   "
        f"{len(excluded_intervals)}"
    )

    print()

    print(
        f"Reference beats (evaluable):  "
        f"{len(reference_peaks)}"
    )

    print(
        f"Detected beats (evaluable):   "
        f"{len(detected_peaks)}"
    )

    print()

    print(
        f"True positives:               "
        f"{true_positives}"
    )

    print(
        f"False positives:              "
        f"{false_positives}"
    )

    print(
        f"False negatives:              "
        f"{false_negatives}"
    )

    print()

    print(
        f"Sensitivity:                  "
        f"{sensitivity * 100:.2f}%"
    )

    print(
        f"Precision:                    "
        f"{precision * 100:.2f}%"
    )

    print(
        f"F1 score:                     "
        f"{f1_score:.4f}"
    )

    # --------------------------------------------------------
    # PRINT EXCLUDED INTERVALS
    # --------------------------------------------------------

    if len(excluded_intervals) > 0:

        print(
            "\nExcluded ventricular-flutter intervals:"
        )

        for start, stop in excluded_intervals:

            start_time = (
                START_TIME_SECONDS
                + start / sampling_frequency
            )

            stop_time = (
                START_TIME_SECONDS
                + stop / sampling_frequency
            )

            print(
                f"  {start_time:.3f} s "
                f"to {stop_time:.3f} s"
            )

    # --------------------------------------------------------
    # BUILD ROW FOR RESULTS TABLE
    # --------------------------------------------------------

    result = {
        "record": record_name,
        "lead": lead_name,

        "total_duration_s":
            DURATION_SECONDS,

        "excluded_flutter_duration_s":
            excluded_duration,

        "evaluable_duration_s":
            evaluable_duration,

        "excluded_intervals":
            len(excluded_intervals),

        "reference_beats_total":
            len(reference_peaks_all),

        "detected_beats_total":
            len(detected_peaks_all),

        "reference_beats_evaluable":
            len(reference_peaks),

        "detected_beats_evaluable":
            len(detected_peaks),

        "true_positives":
            true_positives,

        "false_positives":
            false_positives,

        "false_negatives":
            false_negatives,

        "sensitivity":
            sensitivity,

        "precision":
            precision,

        "f1_score":
            f1_score,
    }

    # Create rows describing each excluded interval.
    excluded_rows = []

    for start, stop in excluded_intervals:

        interval_start_s = (
            START_TIME_SECONDS
            + start / sampling_frequency
        )

        interval_stop_s = (
            START_TIME_SECONDS
            + stop / sampling_frequency
        )

        excluded_rows.append(
            {
                "record": record_name,

                "start_sample":
                    start,

                "stop_sample":
                    stop,

                "start_time_s":
                    interval_start_s,

                "stop_time_s":
                    interval_stop_s,

                "duration_s":
                    interval_stop_s
                    - interval_start_s,
            }
        )

    return (
        result,
        excluded_rows,
    )


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

def calculate_overall_performance(
    results,
):
    """
    Calculate aggregate performance using total TP, FP,
    and FN across all records.
    """

    total_tp = int(
        results[
            "true_positives"
        ].sum()
    )

    total_fp = int(
        results[
            "false_positives"
        ].sum()
    )

    total_fn = int(
        results[
            "false_negatives"
        ].sum()
    )

    (
        sensitivity,
        precision,
        f1_score,
    ) = calculate_metrics(
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
    )

    return (
        total_tp,
        total_fp,
        total_fn,
        sensitivity,
        precision,
        f1_score,
    )


# ============================================================
# PERFORMANCE FIGURE
# ============================================================

def plot_performance(
    results,
    output_path,
):
    """
    Plot final sensitivity, precision, and F1 performance.
    """

    record_labels = results[
        "record"
    ].astype(str)

    x = np.arange(
        len(record_labels)
    )

    bar_width = 0.25

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        x - bar_width,
        results["sensitivity"] * 100,
        width=bar_width,
        label="Sensitivity",
    )

    ax.bar(
        x,
        results["precision"] * 100,
        width=bar_width,
        label="Precision",
    )

    ax.bar(
        x + bar_width,
        results["f1_score"] * 100,
        width=bar_width,
        label="F1 Score",
    )

    ax.set_title(
        "Final Heartbeat Detector Validation Across MIT-BIH Records"
    )

    ax.set_xlabel(
        "MIT-BIH Record"
    )

    ax.set_ylabel(
        "Performance (%)"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        record_labels
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    ax.legend()

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

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []
    all_excluded_intervals = []

    print(
        "\nFINAL MULTI-RECORD ECG DETECTOR VALIDATION"
    )

    print(
        f"Testing {len(RECORD_NAMES)} records "
        f"for {DURATION_SECONDS:.0f} seconds each."
    )

    # --------------------------------------------------------
    # PROCESS EVERY RECORD
    # --------------------------------------------------------

    for record_name in RECORD_NAMES:

        (
            result,
            excluded_rows,
        ) = validate_record(
            record_name
        )

        all_results.append(
            result
        )

        all_excluded_intervals.extend(
            excluded_rows
        )

    results = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # OVERALL PERFORMANCE
    # --------------------------------------------------------

    (
        total_tp,
        total_fp,
        total_fn,
        overall_sensitivity,
        overall_precision,
        overall_f1,
    ) = calculate_overall_performance(
        results
    )

    total_duration = results[
        "total_duration_s"
    ].sum()

    total_excluded_duration = results[
        "excluded_flutter_duration_s"
    ].sum()

    total_evaluable_duration = results[
        "evaluable_duration_s"
    ].sum()

    print("\n" + "=" * 70)
    print("FINAL OVERALL PERFORMANCE")
    print("=" * 70)

    print(
        f"Records tested:               "
        f"{len(results)}"
    )

    print(
        f"Total recording duration:     "
        f"{total_duration:.2f} s"
    )

    print(
        f"Excluded flutter duration:    "
        f"{total_excluded_duration:.2f} s"
    )

    print(
        f"Total evaluable duration:     "
        f"{total_evaluable_duration:.2f} s"
    )

    print()

    print(
        f"Total true positives:         "
        f"{total_tp}"
    )

    print(
        f"Total false positives:        "
        f"{total_fp}"
    )

    print(
        f"Total false negatives:        "
        f"{total_fn}"
    )

    print()

    print(
        f"Overall sensitivity:          "
        f"{overall_sensitivity * 100:.2f}%"
    )

    print(
        f"Overall precision:            "
        f"{overall_precision * 100:.2f}%"
    )

    print(
        f"Overall F1 score:             "
        f"{overall_f1:.4f}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # DISPLAY FINAL TABLE
    # --------------------------------------------------------

    display_columns = [
        "record",
        "evaluable_duration_s",
        "reference_beats_evaluable",
        "detected_beats_evaluable",
        "true_positives",
        "false_positives",
        "false_negatives",
        "sensitivity",
        "precision",
        "f1_score",
    ]

    print(
        "\nFINAL PER-RECORD RESULTS\n"
    )

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE FINAL RESULTS
    # --------------------------------------------------------

    results_path = (
        RESULTS_DIR
        / "final_multi_record_validation.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE EXCLUDED-INTERVAL RECORD
    # --------------------------------------------------------

    excluded_intervals_path = (
        RESULTS_DIR
        / "excluded_flutter_intervals.csv"
    )

    excluded_columns = [
        "record",
        "start_sample",
        "stop_sample",
        "start_time_s",
        "stop_time_s",
        "duration_s",
    ]

    excluded_results = pd.DataFrame(
        all_excluded_intervals,
        columns=excluded_columns,
    )

    excluded_results.to_csv(
        excluded_intervals_path,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL PERFORMANCE FIGURE
    # --------------------------------------------------------

    figure_path = (
        FIGURES_DIR
        / "final_multi_record_performance.png"
    )

    plot_performance(
        results=results,
        output_path=figure_path,
    )

    print(
        f"\nFinal results saved to:\n"
        f"{results_path}"
    )

    print(
        f"\nExcluded intervals saved to:\n"
        f"{excluded_intervals_path}"
    )

    print(
        f"\nFinal figure saved to:\n"
        f"{figure_path}"
    )


if __name__ == "__main__":
    main()