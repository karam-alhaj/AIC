# MTC-AIC3 EEG Classification Models by MetaCode Team
## Table of Contents

- [Competition Overview](#-competition-overview)
- [ What is EEG?](#-what-is-eeg)
- [ Data Description](#-data-description)
  - [ Motor Imagery (MI)](#-motor-imagery-mi-data)
  - [ SSVEP](#-steady-state-visual-evoked-potentials-ssvep-data)
- [Data Preprocessing](#-preprocessing)
- [ Models](#-models)
- [ Results](#-results)
- [ Contributors](#-contributors)

## Competition Overview

The **AIC-3 Competition** focuses on the development of intelligent models for interpreting brain signals in the field of **non-invasive Brain-Computer Interfaces (BCIs)**. The challenge revolves around two widely studied BCI paradigms:

- **Steady-State Visual Evoked Potentials (SSVEP):** These rely on the brain's oscillatory responses to visual stimuli flickering at predefined frequencies.
- **Motor Imagery (MI):** This involves decoding neural patterns generated when a person imagines specific motor movements (hand or directional movement).

The dataset provided contains **multi-channel EEG recordings** collected during both SSVEP and MI sessions, annotated respectively with:
- Visual stimulus frequencies (for SSVEP), and  
- Motor imagery categories (for MI).

## What is EEG?

**Electroencephalography (EEG)** is a non-invasive technique used to record the electrical activity of the brain. It works by placing electrodes on the scalp, which detect voltage fluctuations resulting from the synchronous activity of large groups of neurons, especially in the cerebral cortex.
<div align="center">
  <img src="images/EEG-readings.png" alt="EEG Brain Electrode Setup" width="600"/>
</div>

### Key Features:

- **Multichannel**: EEG data is recorded from multiple electrodes (channels), each representing electrical signals from a specific brain region.

- **Time-series Data**: Signals are captured at high sampling rates (128–1024 Hz), making EEG highly suitable for real-time and temporal analysis.

- **High Temporal Resolution**: EEG can capture changes in brain activity within milliseconds.

- **Low Spatial Resolution**: It provides limited information about where exactly in the brain the activity is occurring.

- **Sensitive to Artifacts**: EEG signals can be affected by eye blinks, muscle movement, and electrical noise, which makes signal preprocessing crucial.


### What Does EEG Data Look Like?

EEG data is typically presented as a series of waveforms, where each waveform corresponds to the electrical activity from a specific electrode location on the scalp. These waveforms vary over time and are analyzed to detect patterns and features in the brain’s electrical activity.

<div align="center">
  <img src="images/eeg-brain-waves-time-series-1.jpg" alt="EEG Waveform" width="800"/>
</div>

A key part of EEG analysis involves breaking down these waveforms into **frequency bands**, including:


- **Delta (0.5–4 Hz)**: Deep sleep
- **Theta (4–8 Hz)**: Light sleep, meditation
- **Alpha (8–13 Hz)**: Relaxed, calm state
- **Beta (13–30 Hz)**: Active thinking, concentration
- **Gamma (>30 Hz)**: High-level cognitive functions

These frequency bands are associated with various mental and physiological states, and extracting features from them plays a vital role in EEG-based classification tasks like MI and SSVEP.

##  Data Description

The dataset used in this competition comes from the **MTC-AIC3 BCI Competition** and contains multi-channel EEG recordings from 40 male participants (average age: 20 years).

Each participant performed tasks under two different Brain-Computer Interface paradigms: **Motor Imagery (MI)** and **Steady-State Visual Evoked Potential (SSVEP)**.

- **EEG Data**: Recordings from 8 channels.
- **EEG Channels**: 8 (FZ, C3, CZ, C4, PZ, PO7, OZ, PO8).
- **Sampling Rate**: 250 Hz
- **Subjects**: 40 participants
- **Participants**: 40 male subjects, average age 20 years.
- **Trials**:
  - MI: 9 seconds per trial → 2250 samples
  - SSVEP: 7 seconds per trial → 1750 samples
- **Trials per Session**: 10 trials for each experimental session.
---
### Directory Structure
The dataset is organized into two main task directories **(MI/ and SSVEP/)** within the mtc-aic3_dataset folder. 
Each task directory contains three subdirectories for data splitting:

- **train/**: Data for model training (30 subjects, 8 trial sessions per subject, 4800 total trials).
- **validation/**: Data for model validation (5 subjects, 1 trial session per subject, 100 total trials).
- **test/**: Data for model testing (5 subjects, 1 trial session per subject, 100 total trials).

Each subject's directory **(e.g., S1/, S2/)** contains session directories **(e.g., 1/)**, representing experimental sessions.

Both **MI** and **SSVEP** tasks are stored under their respective directories inside the main dataset folder...
```
mtc-aic3_dataset/
├── MI/
│   ├── train/
│   │   ├── S1/
│   │   │   └── 1/
│   │   │       └── EEGdata.csv
│   │   ├── S2/
│   │   │   └── ...
│   │   └── ...
│   ├── validation/
│   │   ├── S31/
│   │   │   └── 1/
│   │   │       └── EEGdata.csv
│   │   ├── S32/
│   │   │   └── ...
│   │   └── ...
│   └── test/
│       ├── S36/
│       │   └── 1/
│       │       └── EEGdata.csv
│       ├── S37/
│       │   └── ...
│       └── ...
├── SSVEP/
│   ├── train/
│   │   ├── S1/
│   │   │   └── 1/
│   │   │       └── EEGdata.csv
│   │   ├── S2/
│   │   │   └── ...
│   │   └── ...
│   ├── validation/
│   │   ├── S31/
│   │   │   └── 1/
│   │   │       └── EEGdata.csv
│   │   ├── S32/
│   │   │   └── ...
│   │   └── ...
│   └── test/
│       ├── S36/
│       │   └── 1/
│       │       └── EEGdata.csv
│       ├── S37/
│       │   └── ...
│       └── ...
├── train.csv
├── validation.csv
├── test.csv
└── sample_submission.csv
```
---

### Data File Details

#### EEGdata.csv (Inside Each Session Directory)

Each session directory contains a single `EEGdata.csv` file that holds raw EEG recordings for **all 10 trials**, concatenated sequentially.

- **Columns**:
  - `Time`
  - 8 EEG Channels: FZ, C3, CZ, C4, PZ, PO7, OZ, PO8
  - Motion sensors: `AccX`, `AccY`, `AccZ`, `Gyro1`, `Gyro2`, `Gyro3`
  - `Battery`, `Counter`, and `Validation` flag

- **Samples per trial**:
  - MI: `9 seconds × 250 Hz = 2250 samples`
  - SSVEP: `7 seconds × 250 Hz = 1750 samples`

#### Root-Level CSV Files

These CSV files are located at the root of the dataset and are used to structure splits:

- **`train.csv`**: Labeled training data (4800 entries).
  - Columns: `id`, `subject_id`, `task` (MI or SSVEP), `trial_session`, `trial`, `label`.
  - `id` range: **1–4800**

- **`validation.csv`**: Labeled validation data (100 entries).
  - Same columns as `train.csv`.
  - `id` range: **4801–4900**

- **`test.csv`**: Unlabeled test data (100 entries).
  - Columns: `id`, `subject_id`, `task`, `trial_session`, `trial`.
  - `id` range: **4901–5000**

- **`sample_submission.csv`**: Submission template
  - Columns: `id`, `label`.
  - `label` should be replaced with predictions for each `id` in `test.csv`.

---

### 1. [Motor Imagery (MI)](#-motor-imagery-mi-data)

- **Task**: Participants imagined moving their left or right hand.
- **Classes**:
  - `Left`
  - `Right`
- **Trial Duration**: 9 seconds (2250 samples @ 250 Hz)
- **Total Trials**: 4800 in training (8 sessions × 30 subjects × 10 trials/session)




