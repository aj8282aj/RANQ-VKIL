# RANQ-VKIL — Paper 2 (V5.2)

**Reliability-Aware Variational Quantum Kernel Learning under IBM-calibrated noise**

This repository contains the Python implementation of the V5.2 repeated-outer evaluation used for the RANQ-VKIL Paper 2 pilot study. It evaluates a bank of five variational quantum kernels (VQKs), calibration-conditioned kernel similarities, reliability-aware adaptive fusion, and pure Hadamard degrees `D1`, `D2`, and `D3`, alongside classical SVM controls.

> **Scope:** This is a pilot study using IBM calibration properties to construct approximate noise models for Aer simulation. It is **not** a report of physical quantum processing unit (QPU) execution and does not establish a general quantum advantage.

## Contents

- `ranq_vkil_paper2_v5_2.py` — complete V5.2 experiment script.
- `requirements.txt` — Python package dependencies.
- `.gitignore` — excludes local outputs, caches, credentials, and Python artifacts.
- `CITATION.cff` — citation metadata for this software repository.
- `RELEASE_CHECKLIST.md` — reproducibility and release checks.

## Method at a glance

The script uses five depth-2, four-qubit VQKs with different rotation patterns and entanglement layouts (`VQK_A`–`VQK_E`). The base fusion is followed by pure Hadamard-power controls:

- `D1 = K_base`
- `D2 = K_base ⊙ K_base`
- `D3 = K_base ⊙ K_base ⊙ K_base`

The experiment runs six binary-classification datasets: four UCI datasets (Ionosphere, Spambase, QSAR Biodegradation, and Blood Transfusion) and two synthetic variants. The synthetic variants share a generated latent target with two-way and three-way terms; the clean variant uses low-noise correlated measurements, while the noisy variant adds redundant and nuisance features.

For each dataset, the default repeated outer seeds are 42–51. Each split reserves 30 stratified observations for the untouched outer test set and samples 90 stratified observations from the remaining pool for model development. Other observations are not used in that outer repeat. The same outer split is assessed under the available calibration conditions; those conditions are repeated measurements on the same test observations, not independent outer test replicates.

Within the 90-observation training subset, the code uses fold-specific preprocessing and VQC training, calibration-profile contribution summaries, leave-one-calibration-out (LOCO) model development, and degree selection among `D1`, `D2`, and `D3`, with `D1` as fallback. The script also evaluates individual VQKs, uniform fusion, adaptive fusion, and classical RBF/degree-2/degree-3 SVM baselines.

## Requirements

- Python 3.10 or later.
- Qiskit, Qiskit Aer, and Qiskit IBM Runtime.
- NumPy, pandas, SciPy, scikit-learn, and ucimlrepo.
- An IBM Quantum account with access to the configured backend and permission to retrieve backend properties.

Install the listed packages:

```bash
python -m pip install -r requirements.txt
```

The script uses the IBM Runtime account configuration already available in your environment. Do not put API tokens or instance credentials in this repository. Configure credentials using IBM's supported local account-saving or environment configuration process.

## Run

From the repository directory:

```bash
python ranq_vkil_paper2_v5_2.py
```

The default backend is `ibm_fez` and the default IBM instance string is `open-instance`. These can be overridden without editing the script:

```bash
# Linux/macOS
export IBM_BACKEND_NAME="ibm_fez"
export IBM_INSTANCE="your-instance"
python ranq_vkil_paper2_v5_2.py
```

On PowerShell:

```powershell
$env:IBM_BACKEND_NAME = "ibm_fez"
$env:IBM_INSTANCE = "your-instance"
python .\ranq_vkil_paper2_v5_2.py
```

The four historical calibration-property requests are defined in the source code. The current calibration properties are fetched at runtime, so a rerun at a later date may not reproduce the exact same current-profile result. The script prints the returned timestamps and writes profile metadata into each result bundle. Historical properties may be unavailable depending on backend/API access.

The UCI data are fetched at runtime. QSAR Biodegradation has an official UCI ZIP fallback if the `ucimlrepo` import route is unavailable. An internet connection is therefore needed for first-time dataset retrieval unless the data are already cached or supplied locally as supported by the script.

## Outputs

By default, results are written beneath:

```text
ranq_vkil_paper2_v5_2_repeated_outer_output/
```

The output includes per-dataset/per-seed result bundles, a run manifest, kernel PSD diagnostics, kernel-drift diagnostics, contribution and synergy summaries, LOCO degree-selection records, individual/uniform/adaptive degree controls, classical baseline metrics, and repeated-outer aggregate CSV/JSON reports. These generated files are intentionally excluded from version control by `.gitignore`.

The resume option is enabled by default. Existing completed result bundles may be reused. To conduct a clean run, first archive or remove the relevant output directory and manifest, then run the script again.

## Important implementation and interpretation notes

1. **Approximate noise simulation:** calibration properties are converted into Aer noise channels. The code does not submit these circuits for physical QPU execution.
2. **Readout-error limitation:** a readout-error model is attached to the simulator, but the density-matrix extraction saves the state before measurement. Consequently, readout-assignment errors do not directly enter the premeasurement density matrices used to construct the density-state-fidelity kernels. Do not interpret the current density-kernel results as a full measurement-readout-noise study.
3. **Mixed-state kernel treatment:** noisy similarities are calculated using Qiskit's `state_fidelity`. The training Gram matrix is symmetrized, projected onto the PSD cone, and diagonally rescaled. The test-to-training kernel is not subjected to that training-Gram projection. The projection diagnostics are saved by the script.
4. **Pilot sample size:** the 90/30 design is a deliberately small-sample pilot protocol. It does not use all available observations in each dataset and should not be presented as a full-data benchmark.
5. **Classical comparison:** the included classical baselines are RBF, polynomial degree 2, and polynomial degree 3 SVMs. They are the baselines implemented in this V5.2 script; they should not be described as an exhaustive or fully tuned classical benchmark suite.
6. **No universal advantage claim:** the code is designed to compare kernel construction, fusion, calibration sensitivity, and Hadamard degree effects. Conclusions should be limited to the datasets, splits, calibration profiles, and baselines actually evaluated.

## Reproducibility recommendations

For a reported run, retain the complete output directory and record the Python/package versions, operating system, IBM backend, calibration-property timestamps, fixed physical-qubit mapping, and source revision. Before publishing numerical claims, check that all expected dataset/seed runs completed and that reported means and standard deviations are calculated across outer seeds within each dataset. Calibration conditions within a seed must not be counted as independent outer replicates.

## Citation

Citation metadata are provided in `CITATION.cff`. If you use this code in academic work, cite the associated RANQ-VKIL manuscript when its bibliographic details are finalized, and cite this repository release with its version or commit identifier.

## License

No software license is currently included. Unless the repository owner adds a license, the source remains subject to the default copyright terms; public visibility alone does not grant permission to reuse, modify, or redistribute it.
