# Data Directory

Raw data is **not included** in this repository due to legal and
confidentiality agreements with the industrial partner organisations
where data collection was conducted.

## Expected folder structure

```
data/
├── thermal/
│   ├── Healthy/          ← thermal images (.jpg / .png)
│   ├── Mild_Fault/
│   └── Heavy_Fault/
│
├── audio/
│   ├── Healthy/          ← acoustic recordings (.wav)
│   ├── Mild_Fault/
│   └── Heavy_Fault/
│
└── noise_sources/
    ├── factory.mp3       ← see noise_sources/README.md for download instructions
    ├── cafeteria.mp3
    ├── traffic.mp3
    └── rain.mp3
```

## Data collection summary

| Modality | Device | Classes |
|---|---|---|
| Thermal imaging | Fluke Ti400 PRO infrared camera | Healthy / Mild Fault / Heavy Fault |
| Acoustic recording | INMP441 MEMS microphone | Healthy / Mild Fault / Heavy Fault |

- Total machines evaluated across Experiments: **31**
- Recording conditions: varying voltage, load, ambient noise, and machine age
- Each `.wav` file is segmented into 5-second non-overlapping clips
  (first 10 seconds discarded to remove startup transients)

## Reproducing with public data

For reproducibility without access to the original dataset, the
[CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter)
(also available on [Kaggle](https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets))
can be used as a structural proxy for the acoustic fault classification pipeline.

Note: The CWRU dataset contains vibration signals, not acoustic recordings.
Preprocessing parameters in `src/config.py` may need adjustment for
vibration-domain signals.
