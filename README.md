# ECG Signal Processing & Heartbeat Detection System

A Python-based biomedical signal-processing project that analyzes real electrocardiogram (ECG) recordings from the MIT-BIH Arrhythmia Database, detects R-peaks, calculates heart-rate metrics, and validates detection performance against expert annotations.

The project was developed as an engineering portfolio project to demonstrate biomedical signal processing, Python programming, algorithm development, verification, validation, and technical analysis.

> This project is intended for engineering and educational analysis only. It is not a medical diagnostic system and is not intended to replace clinical ECG equipment or professional medical interpretation.

---

## Project Overview

Electrocardiogram signals contain electrical information about cardiac activity. A major signal-processing task is identifying the QRS complex, particularly the R-peak, because the timing between consecutive R-peaks can be used to determine RR intervals, heart rate, and basic heart-rate-variability metrics.

This project implements a custom ECG processing pipeline rather than relying on a complete prebuilt heartbeat detector.

The system performs:

- ECG data loading
- Raw ECG visualization
- Digital filtering
- QRS enhancement
- Automated R-peak detection
- RR interval calculation
- Heart-rate calculation
- Basic HRV analysis
- Validation against expert annotations
- Multi-record performance evaluation
- Error analysis
- Engineering verification

---

## Dataset

The project uses ECG recordings from the **MIT-BIH Arrhythmia Database** available through PhysioNet.

The database contains real ECG recordings together with expert annotations that identify cardiac events and heartbeat locations.

The project currently evaluates five MIT-BIH records:

- Record 100
- Record 105
- Record 108
- Record 200
- Record 207

Each record was evaluated over a 60-second segment using the same detector configuration.

The ECG dataset itself is not stored in this repository. It is accessed using the Python `wfdb` package.

---

## Signal Processing Pipeline

The heartbeat detection pipeline follows the sequence:

```text
Raw ECG
    ↓
0.5–40 Hz Butterworth Band-Pass Filter
    ↓
Differentiation
    ↓
Squaring
    ↓
Moving-Window Integration
    ↓
Robust Thresholding
    ↓
Candidate QRS Detection
    ↓
R-Peak Refinement
    ↓
RR Intervals
    ↓
Heart Rate / HRV
    ↓
Expert Annotation Validation