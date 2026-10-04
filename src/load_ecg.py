from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb


# ============================================================
# PROJECT SETTINGS
# ============================================================

DATABASE = "mitdb"
RECORD_NAME = "100"
ANNOTATION_EXTENSION = "atr"

CHANNEL_INDEX = 0

START_TIME_SECONDS = 0.0
DURATION_SECONDS = 10.0


# Find the main project folder automatically
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"


# ============================================================
# ECG DATA LOADING
# ============================================================

def load_ecg_segment(
    record_name: str,
    database: str,
    channel_index: int,
    start_time_s: float,
    duration_s: float,
):
    """
    Load a short ECG segment and its expert annotations
    from PhysioNet.

    Parameters
    ----------
    record_name:
        MIT-BIH record number, for example "100".

    database:
        PhysioNet database directory.

    channel_index:
        Which ECG channel to load.

    start_time_s:
        Starting time of the segment in seconds.

    duration_s:
        Length of the segment in seconds.

    Returns
    -------
    record:
        WFDB record containing the ECG samples.

    annotations:
        Expert annotations for the same time segment.

    time_s:
        NumPy array containing the time of each ECG sample.
    """

    # Read the record header first.
    # The header contains information such as:
    # sampling frequency, number of channels, signal length, etc.
    header = wfdb.rdheader(
        record_name,
        pn_dir=database
    )

    sampling_frequency = float(header.fs)

    # Make sure the requested channel exists.
    if channel_index >= header.n_sig:
        raise ValueError(
            f"Channel {channel_index} does not exist. "
            f"This record contains {header.n_sig} channels."
        )

    # Convert time in seconds into sample numbers.
    start_sample = int(round(start_time_s * sampling_frequency))

    stop_sample = int(
        round((start_time_s + duration_s) * sampling_frequency)
    )

    # Prevent attempting to read beyond the end of the recording.
    stop_sample = min(stop_sample, header.sig_len)

    if start_sample >= stop_sample:
        raise ValueError("The requested ECG segment is invalid.")

    # Load one ECG channel in physical units.
    record = wfdb.rdrecord(
        record_name,
        sampfrom=start_sample,
        sampto=stop_sample,
        channels=[channel_index],
        physical=True,
        pn_dir=database
    )

    # Load the corresponding expert annotations.
    #
    # shift_samps=True means annotation sample numbers are
    # shifted so that sample 0 corresponds to the beginning
    # of our selected segment.
    annotations = wfdb.rdann(
        record_name,
        ANNOTATION_EXTENSION,
        sampfrom=start_sample,
        sampto=stop_sample,
        shift_samps=True,
        pn_dir=database
    )

    # Extract the ECG signal from the WFDB Record object.
    ecg_signal = record.p_signal[:, 0]

    # Build a time axis:
    #
    # time = sample number / sampling frequency
    #
    # np.arange(...) produces:
    #
    # 0, 1, 2, 3, 4, ...
    #
    # Dividing by the sampling frequency converts those
    # sample numbers into seconds.
    time_s = (
        np.arange(len(ecg_signal)) / sampling_frequency
        + start_time_s
    )

    return record, annotations, ecg_signal, time_s


# ============================================================
# INFORMATION SUMMARY
# ============================================================

def print_record_summary(
    record,
    annotations,
    ecg_signal,
    start_time_s: float,
):
    """Print useful information about the loaded ECG segment."""

    sampling_frequency = float(record.fs)

    lead_name = record.sig_name[0]
    amplitude_unit = record.units[0]

    duration = len(ecg_signal) / sampling_frequency

    print("\n" + "=" * 55)
    print("ECG RECORD SUMMARY")
    print("=" * 55)

    print(f"Record:               {RECORD_NAME}")
    print(f"Sampling frequency:    {sampling_frequency:.1f} Hz")
    print(f"ECG channel:           {lead_name}")
    print(f"Amplitude unit:         {amplitude_unit}")
    print(f"Samples loaded:         {len(ecg_signal)}")
    print(f"Segment start time:     {start_time_s:.2f} s")
    print(f"Segment duration:       {duration:.2f} s")
    print(f"Annotations loaded:     {len(annotations.sample)}")

    print("\nFirst few expert annotations:")

    number_to_show = min(5, len(annotations.sample))

    for i in range(number_to_show):

        sample_number = annotations.sample[i]

        annotation_time = (
            start_time_s
            + sample_number / sampling_frequency
        )

        symbol = annotations.symbol[i]

        print(
            f"  Sample {sample_number:4d} | "
            f"Time {annotation_time:6.3f} s | "
            f"Symbol {symbol}"
        )

    print("=" * 55)


# ============================================================
# ECG PLOTTING
# ============================================================

def plot_raw_ecg(
    time_s,
    ecg_signal,
    record,
    output_path: Path,
):
    """Create and save a professional raw ECG plot."""

    lead_name = record.sig_name[0]
    amplitude_unit = record.units[0]

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.plot(
        time_s,
        ecg_signal,
        linewidth=1.1
    )

    ax.set_title(
        f"MIT-BIH Record {RECORD_NAME} — Raw ECG ({lead_name})"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Amplitude ({amplitude_unit})")

    ax.grid(
        True,
        alpha=0.3
    )

    ax.set_xlim(
        time_s[0],
        time_s[-1]
    )

    fig.tight_layout()

    # Save a high-resolution version for GitHub/portfolio use.
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
    """Run the ECG loading and visualization workflow."""

    # Make sure the figures folder exists.
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLoading ECG data from PhysioNet...")

    record, annotations, ecg_signal, time_s = load_ecg_segment(
        record_name=RECORD_NAME,
        database=DATABASE,
        channel_index=CHANNEL_INDEX,
        start_time_s=START_TIME_SECONDS,
        duration_s=DURATION_SECONDS
    )

    print_record_summary(
        record=record,
        annotations=annotations,
        ecg_signal=ecg_signal,
        start_time_s=START_TIME_SECONDS
    )

    figure_path = (
        FIGURES_DIR
        / f"record_{RECORD_NAME}_raw_ecg.png"
    )

    plot_raw_ecg(
        time_s=time_s,
        ecg_signal=ecg_signal,
        record=record,
        output_path=figure_path
    )

    print(
        f"\nFigure saved to:\n{figure_path}"
    )


if __name__ == "__main__":
    main()