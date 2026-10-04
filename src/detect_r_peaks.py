from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from load_ecg import load_ecg_segment
from preprocess_ecg import bandpass_filter


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

# Detection parameters
INTEGRATION_WINDOW_SECONDS = 0.150

# Two detected heartbeats must be at least this far apart.
# 0.20 s corresponds to a maximum theoretical rate of 300 bpm.
MIN_PEAK_DISTANCE_SECONDS = 0.20

# Once a candidate heartbeat is found, search this distance
# around it for the actual R-peak in the filtered ECG.
R_PEAK_SEARCH_SECONDS = 0.120

# Controls how high the detection threshold is.
THRESHOLD_FRACTION = 0.35


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"


# ============================================================
# QRS DETECTION SIGNAL
# ============================================================

def calculate_detection_signal(
    filtered_signal,
    sampling_frequency,
    integration_window_seconds=0.150,
):
    """
    Transform the filtered ECG into a signal that emphasizes
    QRS complexes.

    Processing steps:
        1. Differentiate
        2. Square
        3. Moving-window integration
    """

    # --------------------------------------------------------
    # STEP 1: DIFFERENTIATION
    # --------------------------------------------------------
    #
    # The derivative measures how quickly the ECG amplitude
    # changes between samples.
    #
    # QRS complexes normally contain steep slopes, so they
    # produce large derivative values.

    derivative = np.diff(
        filtered_signal,
        prepend=filtered_signal[0],
    )

    # Convert from change-per-sample to approximate
    # change-per-second.
    derivative = (
        derivative
        * sampling_frequency
    )

    # --------------------------------------------------------
    # STEP 2: SQUARING
    # --------------------------------------------------------
    #
    # Squaring makes every value positive and strongly
    # emphasizes large slopes.

    squared_signal = (
        derivative ** 2
    )

    # --------------------------------------------------------
    # STEP 3: MOVING-WINDOW INTEGRATION
    # --------------------------------------------------------
    #
    # Convert the integration window from seconds to samples.

    window_samples = int(
        round(
            integration_window_seconds
            * sampling_frequency
        )
    )

    window_samples = max(
        window_samples,
        1,
    )

    # Moving-average kernel
    integration_kernel = (
        np.ones(window_samples)
        / window_samples
    )

    integrated_signal = np.convolve(
        squared_signal,
        integration_kernel,
        mode="same",
    )

    return (
        derivative,
        squared_signal,
        integrated_signal,
    )


# ============================================================
# CANDIDATE HEARTBEAT DETECTION
# ============================================================

def detect_qrs_candidates(
    integrated_signal,
    sampling_frequency,
):
    """
    Detect candidate QRS locations from the integrated signal.

    A percentile-based reference level is used instead of the
    absolute maximum. This makes the threshold more resistant
    to isolated high-energy artifacts.
    """

    # Typical background QRS-energy level.
    baseline = np.median(
        integrated_signal
    )

    # Robust estimate of the upper energy level.
    #
    # The original detector used np.max(), which allowed one
    # extreme artifact to raise the threshold for an entire
    # ECG recording.
    #
    # The 99th percentile is much less sensitive to a small
    # number of extreme values.
    robust_high_level = np.percentile(
        integrated_signal,
        99,
    )

    # Signal-dependent detection threshold.
    threshold = (
        baseline
        + THRESHOLD_FRACTION
        * (
            robust_high_level
            - baseline
        )
    )

    # Convert minimum spacing between beats from seconds
    # into samples.
    minimum_distance_samples = int(
        round(
            MIN_PEAK_DISTANCE_SECONDS
            * sampling_frequency
        )
    )

    candidate_peaks, properties = find_peaks(
        integrated_signal,
        height=threshold,
        distance=minimum_distance_samples,
    )

    return (
        candidate_peaks,
        threshold,
        properties,
    )


# ============================================================
# R-PEAK REFINEMENT
# ============================================================

def refine_r_peaks(
    candidate_peaks,
    filtered_signal,
    sampling_frequency,
):
    """
    Refine each QRS candidate so the final detected point lies
    on the dominant R-peak location of the filtered ECG.
    """

    search_radius = int(
        round(
            R_PEAK_SEARCH_SECONDS
            * sampling_frequency
        )
    )

    refined_peaks = []

    for candidate in candidate_peaks:

        search_start = max(
            candidate - search_radius,
            0,
        )

        search_stop = min(
            candidate + search_radius + 1,
            len(filtered_signal),
        )

        local_segment = filtered_signal[
            search_start:search_stop
        ]

        # Use absolute amplitude so the detector can also
        # handle leads with an inverted dominant QRS complex.
        local_peak_index = np.argmax(
            np.abs(local_segment)
        )

        r_peak_index = (
            search_start
            + local_peak_index
        )

        refined_peaks.append(
            r_peak_index
        )

    # Remove duplicate detections that may refine to the
    # same ECG sample.
    refined_peaks = np.unique(
        refined_peaks
    )

    return refined_peaks


# ============================================================
# DETECTION SUMMARY
# ============================================================

def print_detection_summary(
    r_peaks,
    sampling_frequency,
    start_time_seconds,
):
    """
    Print detected R-peak locations and times.
    """

    print("\n" + "=" * 60)
    print("R-PEAK DETECTION SUMMARY")
    print("=" * 60)

    print(
        f"Detected R-peaks:        "
        f"{len(r_peaks)}"
    )

    print(
        f"Sampling frequency:      "
        f"{sampling_frequency:.1f} Hz"
    )

    print("\nDetected heartbeat times:")

    for number, peak in enumerate(
        r_peaks,
        start=1,
    ):

        peak_time = (
            start_time_seconds
            + peak / sampling_frequency
        )

        print(
            f"  Beat {number:2d}: "
            f"sample {peak:4d} | "
            f"time {peak_time:6.3f} s"
        )

    print("=" * 60)


# ============================================================
# DIAGNOSTIC PLOT
# ============================================================

def plot_detection_steps(
    time_s,
    filtered_signal,
    derivative,
    squared_signal,
    integrated_signal,
    threshold,
    output_path,
):
    """
    Show the major signal-processing stages used by the
    heartbeat detector.
    """

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 10),
        sharex=True,
    )

    # --------------------------------------------------------
    # 1. FILTERED ECG
    # --------------------------------------------------------

    axes[0].plot(
        time_s,
        filtered_signal,
        linewidth=1.0,
    )

    axes[0].set_title(
        "1. Filtered ECG"
    )

    axes[0].set_ylabel(
        "Amplitude (mV)"
    )

    # --------------------------------------------------------
    # 2. DERIVATIVE
    # --------------------------------------------------------

    axes[1].plot(
        time_s,
        derivative,
        linewidth=1.0,
    )

    axes[1].set_title(
        "2. ECG Derivative"
    )

    axes[1].set_ylabel(
        "Slope"
    )

    # --------------------------------------------------------
    # 3. SQUARED DERIVATIVE
    # --------------------------------------------------------

    axes[2].plot(
        time_s,
        squared_signal,
        linewidth=1.0,
    )

    axes[2].set_title(
        "3. Squared Derivative"
    )

    axes[2].set_ylabel(
        "Squared slope"
    )

    # --------------------------------------------------------
    # 4. MOVING-WINDOW INTEGRATION
    # --------------------------------------------------------

    axes[3].plot(
        time_s,
        integrated_signal,
        linewidth=1.0,
        label="Integrated signal",
    )

    axes[3].axhline(
        threshold,
        linestyle="--",
        linewidth=1.2,
        label="Detection threshold",
    )

    axes[3].set_title(
        "4. Moving-Window Integration"
    )

    axes[3].set_xlabel(
        "Time (s)"
    )

    axes[3].set_ylabel(
        "QRS energy"
    )

    axes[3].legend()

    # --------------------------------------------------------
    # FORMATTING
    # --------------------------------------------------------

    for ax in axes:

        ax.grid(
            True,
            alpha=0.3,
        )

    axes[3].set_xlim(
        time_s[0],
        time_s[-1],
    )

    fig.suptitle(
        "R-Peak Detection Processing Pipeline",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )


# ============================================================
# FINAL R-PEAK PLOT
# ============================================================

def plot_detected_r_peaks(
    time_s,
    filtered_signal,
    r_peaks,
    sampling_frequency,
    start_time_seconds,
    output_path,
):
    """
    Plot the filtered ECG with detected R-peaks marked.
    """

    r_peak_times = (
        start_time_seconds
        + r_peaks / sampling_frequency
    )

    r_peak_amplitudes = (
        filtered_signal[r_peaks]
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
        r_peak_times,
        r_peak_amplitudes,
        s=55,
        marker="o",
        zorder=3,
        label="Detected R-peaks",
    )

    ax.set_title(
        f"MIT-BIH Record {RECORD_NAME} — "
        "Detected R-Peaks"
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

    print(
        "\nLoading ECG signal..."
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
    # PREPROCESSING
    # --------------------------------------------------------

    print(
        "Filtering ECG..."
    )

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

    print(
        "Creating QRS detection signal..."
    )

    (
        derivative,
        squared_signal,
        integrated_signal,
    ) = calculate_detection_signal(
        filtered_signal=filtered_signal,
        sampling_frequency=sampling_frequency,
        integration_window_seconds=INTEGRATION_WINDOW_SECONDS,
    )

    # --------------------------------------------------------
    # FIND CANDIDATE HEARTBEATS
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
    # REFINE TO ACTUAL R-PEAKS
    # --------------------------------------------------------

    r_peaks = refine_r_peaks(
        candidate_peaks=candidate_peaks,
        filtered_signal=filtered_signal,
        sampling_frequency=sampling_frequency,
    )

    print_detection_summary(
        r_peaks=r_peaks,
        sampling_frequency=sampling_frequency,
        start_time_seconds=START_TIME_SECONDS,
    )

    # --------------------------------------------------------
    # SAVE FIGURES
    # --------------------------------------------------------

    diagnostic_figure_path = (
        FIGURES_DIR
        / f"record_{RECORD_NAME}_detection_pipeline.png"
    )

    final_figure_path = (
        FIGURES_DIR
        / f"record_{RECORD_NAME}_detected_r_peaks.png"
    )

    plot_detection_steps(
        time_s=time_s,
        filtered_signal=filtered_signal,
        derivative=derivative,
        squared_signal=squared_signal,
        integrated_signal=integrated_signal,
        threshold=threshold,
        output_path=diagnostic_figure_path,
    )

    plot_detected_r_peaks(
        time_s=time_s,
        filtered_signal=filtered_signal,
        r_peaks=r_peaks,
        sampling_frequency=sampling_frequency,
        start_time_seconds=START_TIME_SECONDS,
        output_path=final_figure_path,
    )

    print(
        "\nFigures saved to:"
    )

    print(
        diagnostic_figure_path
    )

    print(
        final_figure_path
    )


if __name__ == "__main__":
    main()