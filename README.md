# OrchestraMSS

A local web app that separates one orchestra recording with two pretrained models — the **SynthSOD baseline** (X-UMX) and **MVSep Mega 53 Stems** (BS-RoFormer) — and lets you listen to the stems side by side and A/B them per instrument. No model was trained here; the work is in running both models on a 16 GB Mac and comparing their output.

## Problem

The two models come from different codebases with different inputs (mono vs stereo), different stem sets, and different inference code. Running the Mega 53 model with its shipped settings on a 16 GB Mac did not finish in practice: it fell into swap. Comparing the two meant running each separately and lining up the wav files by hand.

## Approach

- **Mega 53 chunk 20 s → 10 s.** With the yaml's `inference.chunk_size` (20 s), the time-axis attention matrix on MPS grows to several GB and the machine swaps. Inference uses the training chunk size (10 s) instead; 5 s can be picked in the UI.
- **Mega 53 computes only the selected stems.** Only the selected mask heads run (`active_stem_ids`), and the output buffers are sized to those stems. MSST's default `demix()` allocates two song-length buffers for all 53 stems, which by that arithmetic exceeds 10 GB for a 5-minute track.
- **SynthSOD inference without the eval stack.** `separate()` was taken from `eval.py` without museval. The Wiener filter runs in blocks of about 46 s, the output length matches the input, and checkpoints load with `weights_only` using an allow-list of numpy types. Input is downmixed to mono (training setting `nb_channels: 1`).
- **One page for the comparison.** Upload and trim an input (default: first 30 s), choose models and stems, watch per-model progress, then compare in an instrument × [SynthSOD | Mega 53] matrix: dB-scale waveforms with RMS dB, click-to-seek, synchronized A/B playback (`A`/`B` keys, `Space` to stop), instrument-family rows (sum of SynthSOD stems vs Mega 53 family stems), and zip download. Jobs run one at a time to protect memory.

## Results

Measured on a Mac mini M4, 16 GB.

| Model | Input | Time | Peak memory |
|---|---|---|---|
| SynthSOD (CPU) | 30 s | about 4 s | 3.4 GB |
| Mega 53 (MPS, 10 s chunk, 20 stems) | 30 s | 62 s | 9.5 GB |
| Mega 53 (MPS, 10 s chunk, 20 stems) | 150 s | 164 s | 10.0 GB |
| Mega 53 (MPS, 5 s chunk) | 30 s | 34 s | 8.3 GB |

Disk use after `setup.sh`:

| Item | Size |
|---|---|
| Python env `.venv` (torch, asteroid, librosa, fastapi, …) | 1.3 GB |
| 8 SynthSOD checkpoints `models/synthsod` (4 trained on SynthSOD + 4 on EnsembleSet) | 811 MB |
| Mega 53 checkpoint `models/mega53` | 1.3 GB |
| Code `third_party` (SynthSOD-Baseline, MSST v1.0.21) | 2.4 MB |
| **Total** | **about 3.4 GB** |

Separation quality was compared by listening only; no SDR or other objective metric was measured.

## Tech stack

- Python 3.11, PyTorch (CPU / Apple MPS)
- Models: SynthSOD X-UMX (asteroid, norbert), MVSep Mega 53 Stems BS-RoFormer (MSST v1.0.21)
- Audio: soundfile, librosa, scipy
- Server: FastAPI + uvicorn
- Front end: plain HTML / CSS / JavaScript

## Run

```bash
./setup.sh
```

```bash
.venv/bin/python main.py
```

Open http://127.0.0.1:8000. `setup.sh` creates `.venv`, installs dependencies, and downloads the model code into `third_party/` and the weights into `models/`.

If you don't need the EnsembleSet-trained models, delete `models/synthsod/ensembleset_*.pth` (−405 MB). Results go to `runs/<job id>/`; one 20 s input produces about 200 MB of float32 wav.

## Lessons learned

- On a 16 GB Mac, chunk size decides whether a transformer separator runs at all: the shipped 20 s inference chunk swapped, while the 10 s training chunk ran in 62 s for a 30 s input.
- Peak memory for Mega 53 was 8.3–10.0 GB in the runs above, so 16 GB is close to the limit; longer inputs than 150 s were not measured.
- The two models do not output the same stem set, so the comparison is only direct at the instrument-family level (summed SynthSOD stems vs Mega 53 family stems).
- Next step: add an objective metric on a dataset with ground-truth stems instead of listening only.

## License

This repository is **AGPL-3.0** ([LICENSE](LICENSE)). `app/separators/synthsod.py` is adapted from `eval.py` of [SynthSOD-Baseline](https://github.com/repertorium/SynthSOD-Baseline) (AGPL-3.0), so the whole repository uses the same license.

- `app/separators/mega53.py` is adapted from `demix()` in [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) (MIT, © 2024 Roman Solovyev). The original MIT notice is kept at the top of the file.
- External code (`third_party/`) and model weights (`models/`) are not in this repository. `setup.sh` downloads them from the original repositories and releases, and each keeps its own license.
- Because of AGPL-3.0, if you modify this app and serve it over a network (including the public link from `share.sh`), you must offer the modified source to its users.

## References

- J. Garcia-Martinez, D. Diaz-Guerra, A. Politis, T. Virtanen, J. J. Carabias-Orti, P. Vera-Candeas, "SynthSOD: Developing an Heterogeneous Dataset for Orchestra Music Source Separation," *IEEE Open Journal of Signal Processing* 6, pp. 129–137, 2025. arXiv:2409.10995 (dataset and X-UMX baseline, `third_party/SynthSOD-Baseline`)
- R. Sawata et al., "All for One and One for All: Improving Music Separation by Bridging Networks," *ICASSP*, 2021. (X-UMX)
- W.-T. Lu et al., "Music Source Separation with Band-Split RoPE Transformer," *ICASSP*, 2024. (BS-RoFormer)
- R. Solovyev, A. Stempkovskiy, T. Habruseva, "Benchmarks and Leaderboards for Sound Demixing Tasks," arXiv:2305.07489, 2023. (Music-Source-Separation-Training, `third_party/MSST`)
- MVSep "Mega 53 Stems" model. https://mvsep.com

---
한국어 문서: [README.ko.md](README.ko.md)
