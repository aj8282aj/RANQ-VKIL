# Release and reproducibility checklist

Before using this public repository as the definitive code companion to a manuscript:

- [ ] Confirm the source revision is the exact version used to generate the reported results.
- [ ] Retain the full per-dataset/per-seed output bundles and aggregate reports.
- [ ] Confirm all 6 datasets and all 10 outer seeds (42–51) completed; inspect the run manifest.
- [ ] Record package versions, Python version, operating system, backend name, and returned calibration-property timestamps.
- [ ] Verify the historical calibration snapshots that were successfully retrieved; do not silently assume all four are available on every run.
- [ ] Keep IBM credentials and tokens outside the repository.
- [ ] Describe the study as IBM calibration-derived Aer simulation, not physical QPU execution.
- [ ] Disclose that readout assignment errors do not directly affect the saved premeasurement density matrices used by the current density-kernel calculation.
- [ ] Explain the training-Gram PSD projection and its distinction from the unprojected test-to-training kernel.
- [ ] Describe the 90-training/30-test pilot sampling protocol and the observations unused in each split.
- [ ] Present classical comparisons as the implemented RBF, polynomial-degree-2, and polynomial-degree-3 SVM baselines; do not imply a comprehensive tuned benchmark suite.
- [ ] Add complete manuscript citation details to `CITATION.cff` when the paper is formally published or assigned a preprint DOI.
- [ ] Add an explicit software license only after the repository owner has chosen one.
