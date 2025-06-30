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

- **EEG Channels**: 8 (FZ, C3, CZ, C4, PZ, PO7, OZ, PO8)
- **Sampling Rate**: 250 Hz
- **Subjects**: 40 participants
- **Trials**:
  - MI: 9 seconds per trial → 2250 samples
  - SSVEP: 7 seconds per trial → 1750 samples
- **Trials per Session**: 10 trials for each experimental session.
---
### General Dataset Structure
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
### 1. [Motor Imagery (MI)](#-motor-imagery-mi-data)

- **Task**: Participants imagined moving their left or right hand.
- **Classes**:
  - `Left`
  - `Right`
- **Trial Duration**: 9 seconds (2250 samples @ 250 Hz)
- **Total Trials**: 4800 in training (8 sessions × 30 subjects × 10 trials/session)

**Directory Structure**:



