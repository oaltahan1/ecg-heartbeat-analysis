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

# Maximum allowed timing difference between a detected
# R-peak and an expert reference beat.
MATCH_TOLERANCE_SECONDS = 0.050


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# STANDARD BEAT ANNOTATION SYMBOLS
# ============================================================

BEAT_SYMBOLS = {
    "N",    # Normal beat
    "L",    # Left bundle branch block beat
    "R",    # Right bundle branch block beat
    "B",    # Bundle branch block beat
    "A",    # Atrial premature beat
    "a",    # Aberrated atrial premature beat
    "J",    # Junctional premature beat
    "S",    # Supraventricular premature beat
    "V",    # Premature ventricular contraction
    "r",    # R-on-T PVC
    "F",    # Fusion beat
    "e",    # Atrial escape beat
    "j",    # Junctional escape beat
    "n",    # Supraventricular escape beat
    "E",    # Ventricular escape beat
    "/",    # Paced beat
    "f",    # Fusion of paced and normal beat
    "Q",    # Unclassifiable beat
    "?",    # Beat not classified during learning
}


# ============================================================
# REFERENCE BEAT EXTRACTION
# ============================================================

def extract_reference_beats(annotations):
    """
    Keep only expert annotations that represent heartbeats.

    Non-beat annotations such as '+' for rhythm changes
    are excluded.
    """

    reference_samples = []

    for sample, symbol in zip(
        annotations.sample,
        annotations.symbol,
    ):
        if symbol in BEAT_SYMBOLS:
            reference_samples.append(
                int(sample)
            )

    return np.asarray(
        reference_samples,
        dtype=int,
    )


# ============================================================
# MATCH DETECTIONS TO REFERENCE BEATS
# ============================================================

def match_detections(
    reference_peaks,
    detected_peaks,
    sampling_frequency,
    tolerance_seconds=0.050,
):
    """
    Match detected R-peaks to expert reference beats.

    Each reference beat and each detected beat may be used
    at most once.
    """

    tolerance_samples = int(
        round(
            tolerance_seconds
            * sampling_frequency
        )
    )

    # Create every possible reference/detection pair that
    # lies within the allowed timing tolerance.
    possible_matches = []

    for reference_index, reference_peak in enumerate(
        reference_peaks
    ):

        for detected_index, detected_peak in enumerate(
            detected_peaks
        ):

            difference = abs(
                detected_peak
                - reference_peak
            )

            if difference <= tolerance_samples:

                possible_matches.append(
                    (
                        difference,
                        reference_index,
                        detected_index,
                    )
                )

    # Smallest timing errors are matched first.
    possible_matches.sort(
        key=lambda item: item[0]
    )

    matched_reference_indices = set()
    matched_detection_indices = set()

    matches = []

    for (
        difference,
        reference_index,
        detected_index,
    ) in possible_matches:

        if (
            reference_index
            not in matched_reference_indices
            and detected_index
            not in matched_detection_indices
        ):

            matched_reference_indices.add(
                reference_index
            )

            matched_detection_indices.add(
                detected_index
            )

            matches.append(
                (
                    reference_peaks[reference_index],
                    detected_peaks[detected_index],
                    difference,
                )
            )

    # --------------------------------------------------------
    # CLASSIFY RESULTS
    # --------------------------------------------------------

    true_positives = len(
        matches
    )

    false_negatives = (
        len(reference_peaks)
        - true_positives
    )

    false_positives = (
        len(detected_peaks)
        - true_positives
    )

    unmatched_reference_peaks = np.asarray(
        [
            reference_peaks[i]
            for i in range(len(reference_peaks))
            if i not in matched_reference_indices
        ],
        dtype=int,
    )

    unmatched_detected_peaks = np.asarray(
        [
            detected_peaks[i]
            for i in range(len(detected_peaks))
            if i not in matched_detection_indices
        ],
        dtype=int,
    )

    return (
        matches,
        true_positives,
        false_positives,
        false_negatives,
        unmatched_reference_peaks,
        unmatched_detected_peaks,
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    true_positives,
    false_positives,
    false_negatives,
):
    """
    Calculate sensitivity, precision, and F1 score.
    """

    sensitivity_denominator = (
        true_positives
        + false_negatives
    )

    precision_denominator = (
        true_positives
        + false_positives
    )

    if sensitivity_denominator > 0:

        sensitivity = (
            true_positives
            / sensitivity_denominator
        )

    else:
        sensitivity = np.nan


    if precision_denominator > 0:

        precision = (
            true_positives
            / precision_denominator
        )

    else:
        precision = np.nan


    if (
        np.isfinite(sensitivity)
        and np.isfinite(precision)
        and (sensitivity + precision) > 0
    ):

        f1_score = (
            2
            * sensitivity
            * precision
            / (sensitivity + precision)
        )

    else:
        f1_score = np.nan


    return (
        sensitivity,
        precision,
        f1_score,
    )


# ============================================================
# PRINT VALIDATION RESULTS
# ============================================================

def print_validation_summary(
    reference_peaks,
    detected_peaks,
    true_positives,
    false_positives,
    false_negatives,
    sensitivity,
    precision,
    f1_score,
    sampling_frequency,
):
    """Print detector validation results."""

    tolerance_samples = int(
        round(
            MATCH_TOLERANCE_SECONDS
            * sampling_frequency
        )
    )

    print("\n" + "=" * 60)
    print("R-PEAK DETECTOR VALIDATION")
    print("=" * 60)

    print(
        f"Reference heartbeats:     "
        f"{len(reference_peaks)}"
    )

    print(
        f"Detected heartbeats:      "
        f"{len(detected_peaks)}"
    )

    print(
        f"Matching tolerance:       "
        f"±{MATCH_TOLERANCE_SECONDS * 1000:.0f} ms "
        f"(±{tolerance_samples} samples)"
    )

    print()

    print(
        f"True positives (TP):      "
        f"{true_positives}"
    )

    print(
        f"False positives (FP):     "
        f"{false_positives}"
    )

    print(
        f"False negatives (FN):     "
        f"{false_negatives}"
    )

    print()

    print(
        f"Sensitivity:              "
        f"{sensitivity * 100:.2f}%"
    )

    print(
        f"Precision:                "
        f"{precision * 100:.2f}%"
    )

    print(
        f"F1 score:                 "
        f"{f1_score:.4f}"
    )

    print("=" * 60)


# ============================================================
# SAVE MATCH DETAILS
# ============================================================

def save_match_results(
    matches,
    sampling_frequency,
    output_path,
):
    """
    Save every matched reference/detection pair.
    """

    rows = []

    for (
        reference_sample,
        detected_sample,
        error_samples,
    ) in matches:

        error_seconds = (
            error_samples
            / sampling_frequency
        )

        rows.append(
            {
                "reference_sample":
                    reference_sample,

                "detected_sample":
                    detected_sample,

                "error_samples":
                    error_samples,

                "error_ms":
                    error_seconds * 1000,
            }
        )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        output_path,
        index=False,
    )

    return results


# ============================================================
# VALIDATION PLOT
# ============================================================

def plot_validation(
    time_s,
    filtered_signal,
    reference_peaks,
    detected_peaks,
    sampling_frequency,
    start_time_seconds,
    output_path,
):
    """
    Plot expert reference beats and algorithm detections
    on the same ECG waveform.
    """

    reference_times = (
        start_time_seconds
        + reference_peaks
        / sampling_frequency
    )

    detected_times = (
        start_time_seconds
        + detected_peaks
        / sampling_frequency
    )

    reference_amplitudes = (
        filtered_signal[
            reference_peaks
        ]
    )

    detected_amplitudes = (
        filtered_signal[
            detected_peaks
        ]
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        time_s,
        filtered_signal,
        linewidth=1.0,
        label="Filtered ECG",
    )

    ax.scatter(
        reference_times,
        reference_amplitudes,
        marker="o",
        s=75,
        facecolors="none",
        linewidths=1.5,
        label="Expert reference",
        zorder=3,
    )

    ax.scatter(
        detected_times,
        detected_amplitudes,
        marker="x",
        s=55,
        linewidths=1.5,
        label="Detected R-peaks",
        zorder=4,
    )

    ax.set_title(
        f"MIT-BIH Record {RECORD_NAME} — "
        "R-Peak Detection Validation"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Amplitude (mV)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    ax.set_xlim(
        time_s[0],
        time_s[-1],
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

    # --------------------------------------------------------
    # LOAD ECG + EXPERT ANNOTATIONS
    # --------------------------------------------------------

    print(
        "\nLoading ECG and expert annotations..."
    )

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
    # RUN OUR DETECTOR
    # --------------------------------------------------------

    filtered_signal = bandpass_filter(
        signal=raw_signal,
        sampling_frequency=sampling_frequency,
        low_cutoff_hz=LOW_CUTOFF_HZ,
        high_cutoff_hz=HIGH_CUTOFF_HZ,
        order=FILTER_ORDER,
    )

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

    detected_peaks = refine_r_peaks(
        candidate_peaks=candidate_peaks,
        filtered_signal=filtered_signal,
        sampling_frequency=sampling_frequency,
    )

    # --------------------------------------------------------
    # EXTRACT TRUE BEAT ANNOTATIONS
    # --------------------------------------------------------

    reference_peaks = extract_reference_beats(
        annotations
    )

    # --------------------------------------------------------
    # MATCH DETECTED VS REFERENCE
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
    # CALCULATE METRICS
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

    print_validation_summary(
        reference_peaks=reference_peaks,
        detected_peaks=detected_peaks,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        sensitivity=sensitivity,
        precision=precision,
        f1_score=f1_score,
        sampling_frequency=sampling_frequency,
    )

    # --------------------------------------------------------
    # SAVE MATCHING DETAILS
    # --------------------------------------------------------

    match_results_path = (
        RESULTS_DIR
        / f"record_{RECORD_NAME}_validation_matches.csv"
    )

    save_match_results(
        matches=matches,
        sampling_frequency=sampling_frequency,
        output_path=match_results_path,
    )

    # --------------------------------------------------------
    # VALIDATION FIGURE
    # --------------------------------------------------------

    figure_path = (
        FIGURES_DIR
        / f"record_{RECORD_NAME}_validation.png"
    )

    plot_validation(
        time_s=time_s,
        filtered_signal=filtered_signal,
        reference_peaks=reference_peaks,
        detected_peaks=detected_peaks,
        sampling_frequency=sampling_frequency,
        start_time_seconds=START_TIME_SECONDS,
        output_path=figure_path,
    )

    print(
        f"\nMatch data saved to:\n"
        f"{match_results_path}"
    )

    print(
        f"\nFigure saved to:\n"
        f"{figure_path}"
    )


if __name__ == "__main__":
    main()