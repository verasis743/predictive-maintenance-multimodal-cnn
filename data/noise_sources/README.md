# Noise Source Files

The noise robustness evaluation in Experiment 4 uses four real-world
environmental noise recordings. These files are **not included** in this
repository due to copyright restrictions.

## Required files

Place the following files in this directory before running `evaluate.py`:

| Filename | Noise type | Description |
|---|---|---|
| `factory.mp3` | Factory floor | Industrial machinery background noise |
| `cafeteria.mp3` | Cafeteria | Indoor crowd and ambient noise |
| `traffic.mp3` | Road traffic | Outdoor urban traffic noise |
| `rain.mp3` | Rain | Outdoor rain and weather noise |

White Gaussian noise is generated programmatically — no file needed.

## Where to download

All four noise types are freely available on [freesound.org](https://freesound.org)
under Creative Commons licences. Search for the noise type and filter by
licence (CC0 or CC BY are recommended for reproducibility).

Alternatively, the [DEMAND dataset](https://zenodo.org/record/1227121)
(Diverse Environments Multichannel Acoustic Noise Database) provides
high-quality recordings of factory, cafeteria, and traffic environments
and is freely available for research use.

## Noise addition method

Noise is added to clean acoustic segments at controlled Signal-to-Noise
Ratios (SNR) using the `add_noise()` function in `src/evaluate.py`.
The noise amplitude is scaled to achieve the target SNR in dB:

```
SNR (dB) = 10 × log10(signal_power / noise_power)
```

SNR levels evaluated: +20, +10, +5, 0, −5, −10 dB.
