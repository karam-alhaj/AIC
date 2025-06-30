# AIC3 EEG Classification Models by MetaCode Team
### 1. Competition Overview

The **AIC-3 Competition** focuses on the development of intelligent models for interpreting brain signals in the field of **non-invasive Brain-Computer Interfaces (BCIs)**. The challenge revolves around two widely studied BCI paradigms:

- **Steady-State Visual Evoked Potentials (SSVEP):** These rely on the brain's oscillatory responses to visual stimuli flickering at predefined frequencies.
- **Motor Imagery (MI):** This involves decoding neural patterns generated when a person imagines specific motor movements (hand or directional movement).

The dataset provided contains **multi-channel EEG recordings** collected during both SSVEP and MI sessions, annotated respectively with:
- Visual stimulus frequencies (for SSVEP), and  
- Motor imagery categories (for MI).

Our goal is to build two robust AI models:
1. A classification model for **SSVEP EEG signals** based on spectral features.
2. A classification model for **MI EEG signals** using time-domain and spatial features.

We implement effective **preprocessing, feature extraction, and modeling pipelines** for both tasks, aiming to maximize classification performance while addressing challenges like signal noise and inter-subject variability.

This project reflects a team-based effort to advance brain-signal decoding using a combination of **signal processing** and **machine learning techniques**.
