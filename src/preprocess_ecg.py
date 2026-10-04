from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from load_ecg import load_ecg_segment


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


# ============================================================
# ECG FILTERING
# ============================================================

def bandpass_filter(
    signal,
    sampling_frequency,
    low_cutoff_hz=0.5,
    high_cutoff_hz=40.0,
    order=4,
):
    """
    Apply a Butterworth band-pass filter to an ECG signal.

    Frequencies below the low cutoff and above the high cutoff
    are attenuated.

    Zero-phase filtering is used so the locations of ECG features,
    especially R-peaks, are not shifted in time.
    """

    nyquist_frequency = sampling_frequency / 2.0

    if low_cutoff_hz <= 0:
        raise ValueError("Low cutoff frequency must be greater than 0 Hz.")

    if high_cutoff_hz >= nyquist_frequency:
        raise ValueError(
            "High cutoff frequency must be below the Nyquist frequency."
        )

    if low_cutoff_hz >= high_cutoff_hz:
        raise ValueError(
            "Low cutoff frequency must be below high cutoff frequency."
        )

    # Create the Butterworth filter.
    #
    # SOS = second-order sections.
    # This representation is numerically stable and is preferred
    # over using polynomial filter coefficients directly.
    sos = butter(
        N=order,
        Wn=[low_cutoff_hz, high_cutoff_hz],
        btype="bandpass",
        fs=sampling_frequency,
        output="sos",
    )

    # Apply the filter forward and backward.
    #
    # Filtering in both directions removes phase shift.
    filtered_signal = sosfiltfilt(
        sos,
        signal,
    )

    return filtered_signal


# ============================================================
# SIGNAL INFORMATION
# ============================================================

def print_filter_summary(
    raw_signal,
    filtered_signal,
    sampling_frequency,
):
    """Print basic information about the filtering operation."""

    print("\n" + "=" * 60)
    print("ECG PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"Sampling frequency:        {sampling_frequency:.1f} Hz")
    print(f"Low cutoff frequency:      {LOW_CUTOFF_HZ:.1f} Hz")
    print(f"High cutoff frequency:     {HIGH_CUTOFF_HZ:.1f} Hz")
    print(f"Filter order:              {FILTER_ORDER}")

    print("\nRaw ECG:")
    print(f"  Mean amplitude:          {np.mean(raw_signal):.4f} mV")
    print(f"  Standard deviation:      {np.std(raw_signal):.4f} mV")

    print("\nFiltered ECG:")
    print(f"  Mean amplitude:          {np.mean(filtered_signal):.4f} mV")
    print(f"  Standard deviation:      {np.std(filtered_signal):.4f} mV")

    print("=" * 60)


# ============================================================
# PLOTTING
# ============================================================

def plot_raw_vs_filtered(
    time_s,
    raw_signal,
    filtered_signal,
    record,
    output_path,
):
    """Compare the raw ECG and filtered ECG using the same time axis."""

    lead_name = record.sig_name[0]
    amplitude_unit = record.units[0]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
    )

    # --------------------------------------------------------
    # Raw ECG
    # --------------------------------------------------------

    axes[0].plot(
        time_s,
        raw_signal,
        linewidth=1.0,
    )

    axes[0].set_title(
        f"Raw ECG — MIT-BIH Record {RECORD_NAME} ({lead_name})"
    )

    axes[0].set_ylabel(
        f"Amplitude ({amplitude_unit})"
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    # --------------------------------------------------------
    # Filtered ECG
    # --------------------------------------------------------

    axes[1].plot(
        time_s,
        filtered_signal,
        linewidth=1.0,
    )

    axes[1].set_title(
        f"Filtered ECG — {LOW_CUTOFF_HZ:.1f}–"
        f"{HIGH_CUTOFF_HZ:.1f} Hz Butterworth Band-pass"
    )

    axes[1].set_xlabel("Time (s)")

    axes[1].set_ylabel(
        f"Amplitude ({amplitude_unit})"
    )

    axes[1].grid(
        True,
        alpha=0.3,
    )

    axes[1].set_xlim(
        time_s[0],
        time_s[-1],
    )

    fig.suptitle(
        "ECG Preprocessing: Raw vs Filtered Signal",
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

    print("\nLoading ECG signal...")

    record, annotations, raw_signal, time_s = load_ecg_segment(
        record_name=RECORD_NAME,
        database=DATABASE,
        channel_index=CHANNEL_INDEX,
        start_time_s=START_TIME_SECONDS,
        duration_s=DURATION_SECONDS,
    )

    sampling_frequency = float(record.fs)

    print("Applying band-pass filter...")

    filtered_signal = bandpass_filter(
        signal=raw_signal,
        sampling_frequency=sampling_frequency,
        low_cutoff_hz=LOW_CUTOFF_HZ,
        high_cutoff_hz=HIGH_CUTOFF_HZ,
        order=FILTER_ORDER,
    )

    print_filter_summary(
        raw_signal=raw_signal,
        filtered_signal=filtered_signal,
        sampling_frequency=sampling_frequency,
    )

    figure_path = (
        FIGURES_DIR
        / f"record_{RECORD_NAME}_raw_vs_filtered.png"
    )

    plot_raw_vs_filtered(
        time_s=time_s,
        raw_signal=raw_signal,
        filtered_signal=filtered_signal,
        record=record,
        output_path=figure_path,
    )

    print(
        f"\nFigure saved to:\n{figure_path}"
    )


if __name__ == "__main__":
    main()