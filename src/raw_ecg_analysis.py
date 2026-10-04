from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from load_ecg import load_ecg_segment


# ============================================================
# SETTINGS
# ============================================================

DATABASE = "mitdb"
RECORD_NAME = "100"
CHANNEL_INDEX = 0

# A shorter segment makes individual ECG features easier to see.
START_TIME_SECONDS = 1.0
DURATION_SECONDS = 3.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"


# ============================================================
# RAW SIGNAL ANALYSIS
# ============================================================

def print_signal_statistics(ecg_signal, record):
    """Print basic numerical information about the raw ECG."""

    print("\n" + "=" * 55)
    print("RAW ECG SIGNAL ANALYSIS")
    print("=" * 55)

    print(f"ECG lead:              {record.sig_name[0]}")
    print(f"Sampling frequency:    {record.fs:.1f} Hz")
    print(f"Number of samples:     {len(ecg_signal)}")
    print(f"Minimum amplitude:     {np.min(ecg_signal):.3f} mV")
    print(f"Maximum amplitude:     {np.max(ecg_signal):.3f} mV")
    print(f"Mean amplitude:        {np.mean(ecg_signal):.3f} mV")
    print(f"Standard deviation:    {np.std(ecg_signal):.3f} mV")

    print("=" * 55)


def plot_zoomed_raw_ecg(
    time_s,
    ecg_signal,
    record,
    annotations,
    start_time_s,
    output_path,
):
    """
    Plot a short raw ECG segment and show expert beat locations.

    The annotation markers are only being used for visualization.
    They are NOT our heartbeat-detection algorithm.
    """

    sampling_frequency = float(record.fs)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Raw ECG waveform
    ax.plot(
        time_s,
        ecg_signal,
        linewidth=1.2,
        label="Raw ECG"
    )

    # --------------------------------------------------------
    # Show expert annotation locations
    # --------------------------------------------------------

    for sample, symbol in zip(
        annotations.sample,
        annotations.symbol
    ):
        # '+' is a rhythm/event annotation rather than a heartbeat.
        if symbol == "+":
            continue

        # Because shift_samps=True was used when loading,
        # annotation samples begin at 0 for this segment.
        sample_index = int(sample)

        if 0 <= sample_index < len(ecg_signal):

            annotation_time = (
                start_time_s
                + sample_index / sampling_frequency
            )

            annotation_amplitude = ecg_signal[sample_index]

            ax.scatter(
                annotation_time,
                annotation_amplitude,
                s=55,
                zorder=3
            )

            ax.annotate(
                symbol,
                (
                    annotation_time,
                    annotation_amplitude
                ),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=9
            )

    # --------------------------------------------------------
    # Figure formatting
    # --------------------------------------------------------

    lead_name = record.sig_name[0]
    amplitude_unit = record.units[0]

    ax.set_title(
        f"MIT-BIH Record {RECORD_NAME} — "
        f"Zoomed Raw ECG ({lead_name})"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Amplitude ({amplitude_unit})")

    ax.grid(
        True,
        which="major",
        alpha=0.35
    )

    ax.minorticks_on()

    ax.grid(
        True,
        which="minor",
        alpha=0.12
    )

    ax.set_xlim(
        time_s[0],
        time_s[-1]
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLoading raw ECG segment...")

    record, annotations, ecg_signal, time_s = load_ecg_segment(
        record_name=RECORD_NAME,
        database=DATABASE,
        channel_index=CHANNEL_INDEX,
        start_time_s=START_TIME_SECONDS,
        duration_s=DURATION_SECONDS
    )

    print_signal_statistics(
        ecg_signal,
        record
    )

    figure_path = (
        FIGURES_DIR
        / f"record_{RECORD_NAME}_raw_ecg_zoom.png"
    )

    plot_zoomed_raw_ecg(
        time_s=time_s,
        ecg_signal=ecg_signal,
        record=record,
        annotations=annotations,
        start_time_s=START_TIME_SECONDS,
        output_path=figure_path
    )

    print(
        f"\nFigure saved to:\n{figure_path}"
    )


if __name__ == "__main__":
    main()