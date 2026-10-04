from pathlib import Path

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
from hrv_analysis import calculate_hrv_metrics


# ============================================================
# PROJECT SETTINGS
# ============================================================

DATABASE = "mitdb"
TEST_RECORD = "100"
CHANNEL_INDEX = 0

START_TIME_SECONDS = 0.0
TEST_DURATION_SECONDS = 10.0

LOW_CUTOFF_HZ = 0.5
HIGH_CUTOFF_HZ = 40.0
FILTER_ORDER = 4

# Engineering acceptance criteria for final detector validation
MIN_SENSITIVITY = 0.95
MIN_PRECISION = 0.95
MIN_F1_SCORE = 0.95

EXPECTED_RECORD_COUNT = 5


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

FINAL_VALIDATION_FILE = (
    RESULTS_DIR
    / "final_multi_record_validation.csv"
)

EXCLUDED_INTERVALS_FILE = (
    RESULTS_DIR
    / "excluded_flutter_intervals.csv"
)

VERIFICATION_OUTPUT_FILE = (
    RESULTS_DIR
    / "verification_table.csv"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def add_requirement(
    results,
    requirement_id,
    requirement,
    test_method,
    actual_result,
    passed,
):
    """
    Add one requirement and its verification result
    to the verification table.
    """

    results.append(
        {
            "requirement_id": requirement_id,
            "requirement": requirement,
            "test_method": test_method,
            "actual_result": actual_result,
            "status": "PASS" if passed else "FAIL",
        }
    )


# ============================================================
# VERIFICATION TESTS
# ============================================================

def run_verification():

    verification_results = []

    print("\n" + "=" * 70)
    print("ECG HEARTBEAT ANALYSIS — ENGINEERING VERIFICATION")
    print("=" * 70)

    # ========================================================
    # REQUIREMENT 1
    # LOAD REAL ECG DATA + EXPERT ANNOTATIONS
    # ========================================================

    (
        record,
        annotations,
        raw_signal,
        time_s,
    ) = load_ecg_segment(
        record_name=TEST_RECORD,
        database=DATABASE,
        channel_index=CHANNEL_INDEX,
        start_time_s=START_TIME_SECONDS,
        duration_s=TEST_DURATION_SECONDS,
    )

    load_pass = (
        len(raw_signal) > 0
        and len(annotations.sample) > 0
        and record.fs > 0
    )

    add_requirement(
        verification_results,
        requirement_id="REQ-01",
        requirement=(
            "The system shall load a real ECG recording "
            "and its expert annotations."
        ),
        test_method=(
            "Load MIT-BIH Record 100 using WFDB and confirm "
            "that ECG samples and annotations are present."
        ),
        actual_result=(
            f"{len(raw_signal)} ECG samples and "
            f"{len(annotations.sample)} annotations loaded "
            f"at {record.fs:.1f} Hz."
        ),
        passed=load_pass,
    )

    sampling_frequency = float(
        record.fs
    )

    # ========================================================
    # REQUIREMENT 2
    # PREPROCESS ECG
    # ========================================================

    filtered_signal = bandpass_filter(
        signal=raw_signal,
        sampling_frequency=sampling_frequency,
        low_cutoff_hz=LOW_CUTOFF_HZ,
        high_cutoff_hz=HIGH_CUTOFF_HZ,
        order=FILTER_ORDER,
    )

    preprocessing_pass = (
        len(filtered_signal) == len(raw_signal)
        and np.all(
            np.isfinite(filtered_signal)
        )
    )

    add_requirement(
        verification_results,
        requirement_id="REQ-02",
        requirement=(
            "The system shall automatically preprocess "
            "the ECG signal."
        ),
        test_method=(
            "Apply the 0.5–40 Hz Butterworth band-pass "
            "filter and verify that the output contains "
            "the same number of finite samples."
        ),
        actual_result=(
            f"Filtered {len(filtered_signal)} samples using "
            f"a {LOW_CUTOFF_HZ:.1f}–{HIGH_CUTOFF_HZ:.1f} Hz "
            f"band-pass filter."
        ),
        passed=preprocessing_pass,
    )

    # ========================================================
    # REQUIREMENT 3
    # AUTOMATIC R-PEAK DETECTION
    # ========================================================

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

    detection_pass = (
        len(r_peaks) >= 2
        and np.all(
            np.diff(r_peaks) > 0
        )
    )

    add_requirement(
        verification_results,
        requirement_id="REQ-03",
        requirement=(
            "The system shall automatically detect "
            "R-peaks without using expert annotations "
            "as detector inputs."
        ),
        test_method=(
            "Run the custom derivative, squaring, "
            "moving-window integration, thresholding, "
            "and R-peak refinement pipeline."
        ),
        actual_result=(
            f"{len(r_peaks)} R-peaks automatically detected "
            f"in the {TEST_DURATION_SECONDS:.0f}-second "
            f"Record {TEST_RECORD} test segment."
        ),
        passed=detection_pass,
    )

    # ========================================================
    # REQUIREMENT 4
    # RR INTERVALS
    # ========================================================

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

    rr_pass = (
        len(rr_intervals)
        == len(r_peaks) - 1
        and np.all(rr_intervals > 0)
        and np.all(
            np.isfinite(rr_intervals)
        )
    )

    add_requirement(
        verification_results,
        requirement_id="REQ-04",
        requirement=(
            "The system shall calculate RR intervals "
            "from consecutive detected R-peaks."
        ),
        test_method=(
            "Calculate sample differences between consecutive "
            "R-peaks and divide by sampling frequency."
        ),
        actual_result=(
            f"{len(rr_intervals)} valid RR intervals calculated; "
            f"mean RR = {np.mean(rr_intervals) * 1000:.2f} ms."
        ),
        passed=rr_pass,
    )

    # ========================================================
    # REQUIREMENT 5
    # HEART RATE
    # ========================================================

    heart_rate_pass = (
        len(heart_rate_bpm) > 0
        and np.all(
            np.isfinite(heart_rate_bpm)
        )
        and np.all(
            heart_rate_bpm > 0
        )
    )

    average_heart_rate = (
        60.0 / np.mean(rr_intervals)
    )

    add_requirement(
        verification_results,
        requirement_id="REQ-05",
        requirement=(
            "The system shall calculate heart rate "
            "from detected RR intervals."
        ),
        test_method=(
            "Calculate instantaneous heart rate using "
            "HR = 60 / RR."
        ),
        actual_result=(
            f"Average heart rate = "
            f"{average_heart_rate:.1f} bpm; "
            f"range = {np.min(heart_rate_bpm):.1f}–"
            f"{np.max(heart_rate_bpm):.1f} bpm."
        ),
        passed=heart_rate_pass,
    )

    # ========================================================
    # REQUIREMENT 6
    # BASIC HRV
    # ========================================================

    (
        mean_rr_ms,
        sdnn_ms,
        rmssd_ms,
    ) = calculate_hrv_metrics(
        rr_intervals
    )

    hrv_values = np.asarray(
        [
            mean_rr_ms,
            sdnn_ms,
            rmssd_ms,
        ]
    )

    hrv_pass = np.all(
        np.isfinite(hrv_values)
    )

    add_requirement(
        verification_results,
        requirement_id="REQ-06",
        requirement=(
            "The system shall calculate basic "
            "time-domain HRV metrics."
        ),
        test_method=(
            "Calculate Mean RR, SDNN, and RMSSD "
            "from detected RR intervals."
        ),
        actual_result=(
            f"Mean RR = {mean_rr_ms:.2f} ms; "
            f"SDNN = {sdnn_ms:.2f} ms; "
            f"RMSSD = {rmssd_ms:.2f} ms."
        ),
        passed=hrv_pass,
    )

    # ========================================================
    # REQUIREMENT 7
    # MULTI-RECORD TESTING
    # ========================================================

    if FINAL_VALIDATION_FILE.exists():

        validation_results = pd.read_csv(
            FINAL_VALIDATION_FILE
        )

        multi_record_pass = (
            len(validation_results)
            == EXPECTED_RECORD_COUNT
        )

        evaluable_duration = validation_results[
            "evaluable_duration_s"
        ].sum()

        add_requirement(
            verification_results,
            requirement_id="REQ-07",
            requirement=(
                "The detector shall be tested on multiple "
                "independent MIT-BIH ECG records."
            ),
            test_method=(
                "Run the same detector configuration on "
                "five 60-second ECG segments."
            ),
            actual_result=(
                f"{len(validation_results)} records tested; "
                f"{evaluable_duration:.2f} s of ECG used "
                f"for final beat-by-beat evaluation."
            ),
            passed=multi_record_pass,
        )

    else:

        validation_results = None

        add_requirement(
            verification_results,
            requirement_id="REQ-07",
            requirement=(
                "The detector shall be tested on multiple "
                "independent MIT-BIH ECG records."
            ),
            test_method=(
                "Check final multi-record validation output."
            ),
            actual_result=(
                "Final validation results file not found."
            ),
            passed=False,
        )

    # ========================================================
    # REQUIREMENT 8
    # QUANTITATIVE DETECTOR PERFORMANCE
    # ========================================================

    if validation_results is not None:

        total_tp = int(
            validation_results[
                "true_positives"
            ].sum()
        )

        total_fp = int(
            validation_results[
                "false_positives"
            ].sum()
        )

        total_fn = int(
            validation_results[
                "false_negatives"
            ].sum()
        )

        sensitivity = (
            total_tp
            / (total_tp + total_fn)
        )

        precision = (
            total_tp
            / (total_tp + total_fp)
        )

        f1_score = (
            2
            * sensitivity
            * precision
            / (sensitivity + precision)
        )

        performance_pass = (
            sensitivity >= MIN_SENSITIVITY
            and precision >= MIN_PRECISION
            and f1_score >= MIN_F1_SCORE
        )

        add_requirement(
            verification_results,
            requirement_id="REQ-08",
            requirement=(
                "The final detector shall achieve at least "
                "95% sensitivity, 95% precision, and "
                "95% F1 score across the evaluable "
                "multi-record dataset."
            ),
            test_method=(
                "Aggregate TP, FP, and FN from final "
                "multi-record validation and calculate "
                "sensitivity, precision, and F1 score."
            ),
            actual_result=(
                f"TP = {total_tp}, FP = {total_fp}, "
                f"FN = {total_fn}; "
                f"Sensitivity = {sensitivity * 100:.2f}%; "
                f"Precision = {precision * 100:.2f}%; "
                f"F1 = {f1_score * 100:.2f}%."
            ),
            passed=performance_pass,
        )

    else:

        add_requirement(
            verification_results,
            requirement_id="REQ-08",
            requirement=(
                "The final detector shall achieve the "
                "defined performance targets."
            ),
            test_method=(
                "Read final multi-record validation results."
            ),
            actual_result=(
                "Performance could not be calculated."
            ),
            passed=False,
        )

    # ========================================================
    # REQUIREMENT 9
    # DOCUMENT NON-EVALUABLE RHYTHM INTERVALS
    # ========================================================

    exclusion_pass = (
        EXCLUDED_INTERVALS_FILE.exists()
    )

    if exclusion_pass:

        excluded_intervals = pd.read_csv(
            EXCLUDED_INTERVALS_FILE
        )

        excluded_duration = (
            excluded_intervals[
                "duration_s"
            ].sum()
            if len(excluded_intervals) > 0
            else 0.0
        )

        exclusion_result = (
            f"{len(excluded_intervals)} excluded interval(s) "
            f"documented, totaling "
            f"{excluded_duration:.2f} s."
        )

    else:

        exclusion_result = (
            "Excluded interval record was not found."
        )

    add_requirement(
        verification_results,
        requirement_id="REQ-09",
        requirement=(
            "Non-evaluable ventricular-flutter intervals "
            "shall be explicitly documented rather than "
            "silently removed."
        ),
        test_method=(
            "Verify that excluded intervals are stored "
            "in a dedicated CSV file with start, stop, "
            "and duration information."
        ),
        actual_result=exclusion_result,
        passed=exclusion_pass,
    )

    # ========================================================
    # CREATE VERIFICATION TABLE
    # ========================================================

    verification_table = pd.DataFrame(
        verification_results
    )

    verification_table.to_csv(
        VERIFICATION_OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nVERIFICATION RESULTS\n")

    print(
        verification_table.to_string(
            index=False
        )
    )

    passed_requirements = (
        verification_table[
            "status"
        ]
        == "PASS"
    ).sum()

    total_requirements = len(
        verification_table
    )

    print("\n" + "=" * 70)

    print(
        f"Requirements passed: "
        f"{passed_requirements}/{total_requirements}"
    )

    if passed_requirements == total_requirements:

        print(
            "FINAL VERIFICATION STATUS: PASS"
        )

    else:

        print(
            "FINAL VERIFICATION STATUS: FAIL"
        )

    print("=" * 70)

    print(
        f"\nVerification table saved to:\n"
        f"{VERIFICATION_OUTPUT_FILE}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    run_verification()