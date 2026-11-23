"""
RANQ-VKIL Paper 2 -- Repeated Outer V5.2 Multi-Dataset
Reliability-Aware Variational Quantum Kernel Learning under IBM-calibrated noise.

V5.2 retains the V5.1 polynomial-Hadamard formulation while adding repeated outer evaluation with a low-dimensional
polynomial-Hadamard expansion of the reliability-weighted base fusion:

    D1 = K_base
    D2 = K_base ⊙ K_base
    D3 = K_base ⊙ K_base ⊙ K_base

Degree mixtures are selected through nested cross-calibration evidence. Degree 1
is always available as the fallback, so the method never forces a higher-order
term to be active. All degree components use training-side Frobenius normalization.

The pilot retains fold-specific preprocessing, leakage-free VQC training, a fixed
IBM backend and fixed physical qubits across calibration snapshots, IBM
calibration-derived approximate Aer noise, distributional contribution, and
leave-one-calibration-out evaluation with an untouched outer test set.

This is a pilot validation study; IBM calibration-derived simulation is not
physical QPU execution.
"""

from __future__ import annotations

import json
import os
import time
import warnings
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, FrozenSet

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.noise.device import (
        basic_device_gate_errors,
        basic_device_readout_errors,
    )
    from qiskit_ibm_runtime import QiskitRuntimeService
except Exception as exc:
    raise ImportError(
        "Qiskit dependencies are unavailable.\n"
        f"Original import error: {exc}"
    ) from exc


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

# V5.2 repeated outer evaluation. Each seed defines a fresh outer split, fresh
# fold-specific preprocessing, fresh VQC training, fresh contribution/reliability
# estimation, and fresh degree selection. The same five IBM calibration conditions
# are then evaluated on that seed-specific held-out test set.
V52_OUTER_SEEDS = tuple(range(42, 52))  # 10 repeated outer seeds
V52_RESUME = True
V52_KEEP_INDIVIDUAL_BUNDLES = True
V52_RUN_CLASSICAL_BASELINES_PER_REPEAT = True

# Mutable per-repeat seed context. All stochastic operations inside the V5.1
# pipeline are derived from this seed so the existing implementation can be
# reused without duplicating the full V5.1 algorithm.
ACTIVE_RUN_SEED = RANDOM_SEED
PILOT_TRAIN_SAMPLES = 90
PILOT_TEST_SAMPLES = 30
N_QUBITS = 4

# Paper 2 multi-dataset evaluation. These datasets do not repeat the five
# datasets used in Paper 1 (Breast Cancer, Heart Disease, Diabetes, Banknote, Sonar).
DATASETS_TO_RUN = (
    "ionosphere",
    "spambase",
    "qsar_biodegradation",
    "blood_transfusion",
    "synthetic_3way_interaction_clean",
    "synthetic_3way_interaction",
)

PAPER1_DATASETS = {
    "breast_cancer",
    "heart_disease",
    "diabetes",
    "banknote",
    "sonar",
}

DATASET_UCI_IDS = {
    "ionosphere": 52,
    "spambase": 94,
    "qsar_biodegradation": 254,
    "blood_transfusion": 176,
}

DATASET_DISPLAY_NAMES = {
    "ionosphere": "Ionosphere",
    "spambase": "Spambase",
    "qsar_biodegradation": "QSAR Biodegradation",
    "blood_transfusion": "Blood Transfusion Service Center",
    "synthetic_3way_interaction_clean": "Synthetic-3WayInteraction (Clean)",
    "synthetic_3way_interaction": "Synthetic-3WayInteraction (Noisy)",
}

DATASET_DOIS = {
    "ionosphere": "10.24432/C5W01B",
    "spambase": "10.24432/C53G6X",
    "qsar_biodegradation": "10.24432/C5H60M",
    "blood_transfusion": "10.24432/C5GS39",
    "synthetic_3way_interaction": None,
}

OUTPUT_ROOT_DIR = Path("ranq_vkil_paper2_v5_2_repeated_outer_output")
OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = OUTPUT_ROOT_DIR

# Every VQK is depth 2 with data re-uploading.
VQC_DEPTH = 2
ACTIVE_KERNEL_IDS = ["VQK_A", "VQK_B", "VQK_C", "VQK_D", "VQK_E"]

# Pilot repetition settings.
ATTRIBUTION_REPEATS = 4
ATTRIBUTION_FOLDS = 3
ROBUSTNESS_FOLDS = 3
RELIABILITY_QUANTILE = 0.10
RELIABILITY_Q25 = 0.25
RELIABILITY_SCORE_ALPHA = 0.50
RELIABILITY_SOFTMAX_TEMPERATURE = 1.00
INTERACTION_WEIGHT_CAP = 0.30
INTERACTION_ETA_GRID = (0.00, 0.10, 0.25, 0.40, 0.60, 0.80, 1.00)
INTERACTION_ETA_SELECTION_FOLDS = 3
ROBUSTNESS_LAMBDA = 0.25

# Attribution uses a continuous OOF utility (ROC-AUC). Held-out classification
# reports include ROC-AUC, F1, balanced accuracy, MCC, accuracy, precision, and recall.
ATTRIBUTION_UTILITY = "roc_auc"
INTERACTION_GAIN_MODE = "adaptive_polynomial_hadamard_degree_fusion"
ROBUST_OPTIMIZATION_UTILITY = "roc_auc"

# Continuous reliability is expressed as a soft prior rather than a hard keep/remove
# threshold. The prior is derived from q10/q25 reliability summaries and is used as
# an optimizer start and tie-break only; robust ROC-AUC remains the data-driven objective.
VQC_MAXITER = 80
FUSION_MAXITER = 30
FUSION_MULTI_STARTS = 12
FUSION_DIRICHLET_ALPHA = 0.70
FUSION_FTOL = 1e-5
OPTIMIZER_TIE_TOLERANCE = 1e-6

# IBM account/backend. The user has already validated this instance/backend.
IBM_INSTANCE = os.getenv("IBM_INSTANCE", "open-instance")
IBM_BACKEND_NAME = os.getenv("IBM_BACKEND_NAME", "ibm_fez")

# Fixed qubits selected in the previous calibration-aware pilot using the
# reference calibration. They are intentionally frozen to avoid coupling
# qubit-selection changes to calibration robustness analysis.
REFERENCE_CALIBRATION_LABEL = "HIST_20260827"
FIXED_PHYSICAL_QUBITS = [16, 23, 22, 21]

# Exact historical requests used in the user's validated ibm_fez run.
# IBM returns the closest property set at or before the requested timestamp.
HISTORICAL_REQUESTS = [
    ("HIST_20260827", "2026-08-27T11:28:26+03:30"),
    ("HIST_20260903", "2026-09-03T11:28:26+03:30"),
    ("HIST_20260910", "2026-09-10T11:28:26+03:30"),
    ("HIST_20260916", "2026-09-16T11:28:26+03:30"),
]



# ============================================================
# Data classes
# ============================================================

@dataclass(frozen=True)
class KernelSpec:
    kernel_id: str
    rotation_pattern: Tuple[str, ...]
    entanglement: str
    depth: int = VQC_DEPTH


@dataclass
class CalibrationProfile:
    profile_id: str
    timestamp: str
    properties: object | None
    noise_model: NoiseModel | None
    simulator: AerSimulator | None
    physical_qubits: List[int]


KERNEL_SPECS: Dict[str, KernelSpec] = {
    "VQK_A": KernelSpec("VQK_A", ("ry", "rz"), "linear"),
    "VQK_B": KernelSpec("VQK_B", ("rx", "ry"), "reverse_linear"),
    "VQK_C": KernelSpec("VQK_C", ("rx", "ry", "rz"), "brickwall"),
    "VQK_D": KernelSpec("VQK_D", ("ry", "rz"), "staggered"),
    "VQK_E": KernelSpec("VQK_E", ("rx", "ry"), "linear"),
}


# ============================================================
# Generic helpers
# ============================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)


def fmt_float(x: float | None, digits: int = 6) -> str:
    if x is None:
        return "None"
    return f"{float(x):.{digits}f}"


def summarize(values):
    """Return robust descriptive statistics for a contribution/gain sample."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "q10": float("nan"),
            "q25": float("nan"),
            "q50": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
            "n": 0,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "n": int(arr.size),
    }


def select_base_kernels(base_summaries, threshold):
    """Select reliable base kernels; fall back to the highest q10 kernel."""
    selected = [
        kid
        for kid in ACTIVE_KERNEL_IDS
        if base_summaries[kid]["q10"] > float(threshold)
    ]
    if selected:
        return selected, "positive_q10"
    fallback = max(
        ACTIVE_KERNEL_IDS,
        key=lambda k: base_summaries[k]["q10"],
    )
    return [fallback], "fallback_argmax_q10"


def json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(type(obj).__name__)


# ============================================================
# Dataset registry + leakage-controlled outer split
# ============================================================


def _normalize_binary_target(y_raw, dataset_id):
    """Normalize known UCI binary targets to integer labels {0, 1}."""
    if isinstance(y_raw, pd.DataFrame):
        if y_raw.shape[1] < 1:
            raise ValueError(f"{dataset_id}: target dataframe has no columns.")
        s = y_raw.iloc[:, 0]
    elif isinstance(y_raw, pd.Series):
        s = y_raw
    else:
        s = pd.Series(np.asarray(y_raw).reshape(-1))

    # First try direct numeric conversion. This covers 0/1 targets.
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all():
        vals = np.asarray(numeric, dtype=float)
        unique = np.unique(vals)
        if unique.size == 2:
            if np.array_equal(unique, np.array([0.0, 1.0])):
                return vals.astype(int)
            mapping = {unique[0]: 0, unique[1]: 1}
            return np.asarray([mapping[v] for v in vals], dtype=int)

    labels = s.astype(str).str.strip().str.lower()
    explicit = {
        "g": 1, "good": 1, "1": 1, "yes": 1, "true": 1,
        "b": 0, "bad": 0, "0": 0, "no": 0, "false": 0,
        "rb": 1, "ready": 1, "ready biodegradable": 1,
        "nrb": 0, "not ready": 0, "not ready biodegradable": 0,
    }
    mapped = labels.map(explicit)
    if mapped.notna().all():
        return mapped.to_numpy(dtype=int)

    unique = sorted(labels.dropna().unique().tolist())
    if len(unique) != 2:
        raise ValueError(
            f"{dataset_id}: expected binary target, found labels {unique}."
        )
    fallback = {unique[0]: 0, unique[1]: 1}
    return labels.map(fallback).to_numpy(dtype=int)


QSAR_BIODEG_UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/254/"
    "qsar%2Bbiodegradation.zip"
)
QSAR_BIODEG_CACHE_DIR = OUTPUT_ROOT_DIR / "dataset_cache"
QSAR_BIODEG_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_qsar_biodegradation_direct():
    """Load UCI QSAR biodegradation directly when ucimlrepo cannot import it.

    UCI currently lists the dataset and its `biodeg.csv` file, but the
    ucimlrepo package can report that dataset id=254 is not available for import.
    The fallback downloads the official UCI ZIP and reads the bundled CSV.
    """
    local_env = os.getenv("QSAR_BIODEG_LOCAL_FILE", "").strip()
    local_candidates = []
    if local_env:
        local_candidates.append(Path(local_env).expanduser())
    local_candidates.extend([
        QSAR_BIODEG_CACHE_DIR / "biodeg.csv",
        Path("biodeg.csv"),
    ])

    csv_path = next((p for p in local_candidates if p.is_file()), None)
    if csv_path is not None:
        data = pd.read_csv(csv_path, sep=";", header=None)
    else:
        zip_path = QSAR_BIODEG_CACHE_DIR / "qsar_biodegradation.zip"
        try:
            if not zip_path.is_file():
                request = urllib.request.Request(
                    QSAR_BIODEG_UCI_ZIP_URL,
                    headers={"User-Agent": "RANQ-VKIL-Paper2-V5.2"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    zip_bytes = response.read()
                zip_path.write_bytes(zip_bytes)

            with zipfile.ZipFile(zip_path, "r") as archive:
                names = [name for name in archive.namelist() if name.lower().endswith("biodeg.csv")]
                if not names:
                    raise FileNotFoundError(
                        "The downloaded UCI QSAR archive does not contain biodeg.csv."
                    )
                with archive.open(names[0]) as fh:
                    raw = fh.read()
                data = pd.read_csv(io.BytesIO(raw), sep=";", header=None)
        except Exception as exc:
            raise RuntimeError(
                "QSAR Biodegradation could not be loaded through ucimlrepo, and the "
                "official UCI direct-download fallback also failed.\n"
                f"UCI URL: {QSAR_BIODEG_UCI_ZIP_URL}\n"
                "You can download the official `biodeg.csv` from UCI and either place "
                "it beside this script or set the environment variable "
                "QSAR_BIODEG_LOCAL_FILE to its full path.\n"
                f"Original fallback error: {exc}"
            ) from exc

    if data.shape[1] < 2:
        raise ValueError(
            f"qsar_biodegradation: expected 41 features plus 1 target; got {data.shape[1]} columns."
        )

    # Official UCI biodeg.csv is semicolon-separated and contains 41 descriptor
    # columns followed by the experimental class (RB/NRB).
    X_df = data.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    y_df = data.iloc[:, -1]

    if X_df.shape[1] != 41:
        raise ValueError(
            f"qsar_biodegradation: expected 41 descriptor columns, got {X_df.shape[1]}."
        )

    y = _normalize_binary_target(y_df, "qsar_biodegradation")
    X = X_df.to_numpy(dtype=float)

    if len(X) != len(y):
        raise ValueError(
            f"qsar_biodegradation: feature/target length mismatch: X={len(X)} y={len(y)}"
        )

    return X, y


def _load_uci_dataset(dataset_id):
    """Load a UCI dataset through ucimlrepo, with a direct UCI fallback for QSAR."""
    if dataset_id not in DATASET_UCI_IDS:
        raise KeyError(dataset_id)

    # Dataset id 254 is listed by UCI, but its current ucimlrepo endpoint can
    # report that it is not available for import. Use the official file fallback.
    if dataset_id == "qsar_biodegradation":
        try:
            from ucimlrepo import fetch_ucirepo
            try:
                repo = fetch_ucirepo(id=DATASET_UCI_IDS[dataset_id])
                X_df = repo.data.features.copy()
                y_df = repo.data.targets.copy()
            except Exception as exc:
                warnings.warn(
                    "ucimlrepo could not import QSAR Biodegradation; using the "
                    f"official UCI file fallback. Details: {exc}"
                )
                return _load_qsar_biodegradation_direct()
        except ImportError:
            warnings.warn(
                "ucimlrepo is not installed; using the official UCI file fallback "
                "for QSAR Biodegradation."
            )
            return _load_qsar_biodegradation_direct()
    else:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise ImportError(
                "The multi-dataset V5.2 code requires ucimlrepo for the UCI datasets. "
                "Install it with: python -m pip install ucimlrepo"
            ) from exc

        repo = fetch_ucirepo(id=DATASET_UCI_IDS[dataset_id])
        X_df = repo.data.features.copy()
        y_df = repo.data.targets.copy()

    # The selected UCI predictors are numeric/real-valued.
    X_num = X_df.apply(pd.to_numeric, errors="coerce")
    if X_num.shape[1] == 0:
        raise ValueError(f"{dataset_id}: no feature columns were returned by UCI.")

    y = _normalize_binary_target(y_df, dataset_id)
    X = X_num.to_numpy(dtype=float)

    if len(X) != len(y):
        raise ValueError(
            f"{dataset_id}: feature/target length mismatch: X={len(X)} y={len(y)}"
        )

    # Keep raw observations. Missing feature values are handled later by the
    # existing leakage-controlled median imputation in the training pipeline.
    valid_y = np.isfinite(y.astype(float))
    if not np.all(valid_y):
        raise ValueError(f"{dataset_id}: non-finite target values were returned.")

    return X, y


def _generate_synthetic_3way_interaction(
    n_samples=600,
    random_seed=20260920,
    variant="noisy",
):
    """Generate a reproducible binary dataset with known 2-way and 3-way interactions.

    Shared latent ground truth:
        s = z1*z2 + 0.8*z3*z4 + 0.6*z1*z2*z3 + epsilon

    Both synthetic variants use the same latent variables and class labels:
      - clean: 4 latent features + 16 low-noise correlated measurements of those latents.
      - noisy: 4 latent + 8 redundant/correlated features + 8 unit-variance nuisance features.

    The clean variant is intended to isolate the higher-order interaction signal
    while preserving a 20-dimensional observed input. The noisy variant preserves
    the original V5.1 synthetic construction for comparability with the prior run.
    """
    if variant not in {"clean", "noisy"}:
        raise ValueError(f"Unknown synthetic variant: {variant}")

    rng = np.random.default_rng(int(random_seed))

    # Ground-truth latent variables and labels are identical across variants.
    z = rng.normal(0.0, 1.0, size=(n_samples, 4))
    target_noise = rng.normal(0.0, 0.35, size=n_samples)
    score = (
        z[:, 0] * z[:, 1]
        + 0.8 * z[:, 2] * z[:, 3]
        + 0.6 * z[:, 0] * z[:, 1] * z[:, 2]
        + target_noise
    )
    threshold = float(np.median(score))
    y = (score > threshold).astype(int)

    if variant == "clean":
        # All 20 observed features are noisy linear measurements of the same
        # four latent variables. This keeps the observed input 20-dimensional
        # while preserving an effectively 4D latent subspace after the pipeline's
        # StandardScaler + PCA preprocessing. There are no independent nuisance
        # dimensions in the clean control.
        redundant_noise = rng.normal(0.0, 0.02, size=(n_samples, 16))
        redundant = np.column_stack([
            z[:, 0] + redundant_noise[:, 0],
            z[:, 1] + redundant_noise[:, 1],
            z[:, 2] + redundant_noise[:, 2],
            z[:, 3] + redundant_noise[:, 3],
            z[:, 0] + 0.2 * z[:, 1] + redundant_noise[:, 4],
            z[:, 1] - 0.2 * z[:, 2] + redundant_noise[:, 5],
            z[:, 2] + 0.2 * z[:, 3] + redundant_noise[:, 6],
            z[:, 3] + 0.2 * z[:, 0] + redundant_noise[:, 7],
            0.7 * z[:, 0] + 0.3 * z[:, 2] + redundant_noise[:, 8],
            0.6 * z[:, 1] - 0.2 * z[:, 3] + redundant_noise[:, 9],
            0.5 * z[:, 0] + 0.5 * z[:, 1] + redundant_noise[:, 10],
            0.5 * z[:, 2] + 0.5 * z[:, 3] + redundant_noise[:, 11],
            z[:, 0] - z[:, 2] + redundant_noise[:, 12],
            z[:, 1] - z[:, 3] + redundant_noise[:, 13],
            z[:, 0] + z[:, 2] + redundant_noise[:, 14],
            z[:, 1] + z[:, 2] + redundant_noise[:, 15],
        ])
        X = np.column_stack([z, redundant]).astype(float)

        redundant_noise_std = 0.02
        nuisance_std = 0.0
        n_redundant = 16
        n_nuisance = 0

    else:
        # This is the original noisy V5.1 construction.
        redundant_noise = rng.normal(0.0, 0.20, size=(n_samples, 8))
        redundant = np.column_stack([
            z[:, 0] + redundant_noise[:, 0],
            z[:, 1] + redundant_noise[:, 1],
            z[:, 2] + redundant_noise[:, 2],
            z[:, 3] + redundant_noise[:, 3],
            0.7 * z[:, 0] + 0.3 * z[:, 2] + redundant_noise[:, 4],
            0.6 * z[:, 1] - 0.2 * z[:, 3] + redundant_noise[:, 5],
            z[:, 0] * z[:, 2] + redundant_noise[:, 6],
            z[:, 1] * z[:, 3] + redundant_noise[:, 7],
        ])
        nuisance = rng.normal(0.0, 1.0, size=(n_samples, 8))
        X = np.column_stack([z, redundant, nuisance]).astype(float)

        redundant_noise_std = 0.20
        nuisance_std = 1.0
        n_redundant = 8
        n_nuisance = 8

    return X, y, {
        "n_samples_full": int(n_samples),
        "n_features_full": int(X.shape[1]),
        "interaction_definition": "z1*z2 + 0.8*z3*z4 + 0.6*z1*z2*z3 + epsilon",
        "expected_interaction_degrees": [2, 3],
        "target_noise_std": 0.35,
        "redundant_feature_noise_std": float(redundant_noise_std),
        "n_latent_features": 4,
        "n_redundant_features": int(n_redundant),
        "n_nuisance_features": int(n_nuisance),
        "nuisance_feature_std": float(nuisance_std),
        "target_threshold": threshold,
        "random_seed": int(random_seed),
        "synthetic_variant": variant,
    }


def _generate_synthetic_pair_for_validation():
    """Deterministically verify the two synthetic controls before the main run."""
    X_clean, y_clean, _ = _generate_synthetic_3way_interaction(
        variant="clean"
    )
    X_noisy, y_noisy, _ = _generate_synthetic_3way_interaction(
        variant="noisy"
    )
    if X_clean.shape != (600, 20) or X_noisy.shape != (600, 20):
        raise RuntimeError("Synthetic validation failed: expected 600x20 for both variants.")
    if not np.array_equal(y_clean, y_noisy):
        raise RuntimeError(
            "Synthetic validation failed: clean and noisy variants must share the same labels."
        )

    # Descriptive audit only: the actual experiment still fits preprocessing on
    # outer-train / inner-train partitions and never uses these full-data statistics.
    clean_std = StandardScaler().fit_transform(X_clean)
    noisy_std = StandardScaler().fit_transform(X_noisy)
    clean_pca4 = float(PCA(n_components=4, random_state=RANDOM_SEED).fit(clean_std).explained_variance_ratio_.sum())
    noisy_pca4 = float(PCA(n_components=4, random_state=RANDOM_SEED).fit(noisy_std).explained_variance_ratio_.sum())

    return {
        "shape_clean": tuple(X_clean.shape),
        "shape_noisy": tuple(X_noisy.shape),
        "class_counts": {
            "0": int(np.sum(y_clean == 0)),
            "1": int(np.sum(y_clean == 1)),
        },
        "full_dataset_standardized_pca4_explained_variance_clean": clean_pca4,
        "full_dataset_standardized_pca4_explained_variance_noisy": noisy_pca4,
        "same_ground_truth_labels": True,
    }


def load_dataset(dataset_id):
    """Return X, y, and provenance metadata for one Paper 2 dataset."""
    if dataset_id in PAPER1_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_id}' is listed as a Paper 1 dataset and is excluded from Paper 2."
        )

    if dataset_id == "synthetic_3way_interaction_clean":
        X, y, extra = _generate_synthetic_3way_interaction(variant="clean")
        meta = {
            "dataset_id": dataset_id,
            "dataset_name": DATASET_DISPLAY_NAMES[dataset_id],
            "source": "synthetic_generated_in_code",
            "uci_id": None,
            "doi": None,
            **extra,
        }
        return X, y, meta

    if dataset_id == "synthetic_3way_interaction":
        X, y, extra = _generate_synthetic_3way_interaction(variant="noisy")
        meta = {
            "dataset_id": dataset_id,
            "dataset_name": DATASET_DISPLAY_NAMES[dataset_id],
            "source": "synthetic_generated_in_code",
            "uci_id": None,
            "doi": None,
            **extra,
        }
        return X, y, meta

    X, y = _load_uci_dataset(dataset_id)
    meta = {
        "dataset_id": dataset_id,
        "dataset_name": DATASET_DISPLAY_NAMES[dataset_id],
        "source": "UCI Machine Learning Repository",
        "uci_id": int(DATASET_UCI_IDS[dataset_id]),
        "doi": DATASET_DOIS[dataset_id],
        "n_samples_full": int(len(X)),
        "n_features_full": int(X.shape[1]),
    }
    return X, y, meta

def load_outer_pilot_split(dataset_id, split_seed=None):
    X, y, meta = load_dataset(dataset_id)
    split_seed = int(ACTIVE_RUN_SEED if split_seed is None else split_seed)

    # Split the untouched outer test first, then cap only the training pool.
    X_pool, X_test, y_pool, y_test = train_test_split(
        X,
        y,
        test_size=PILOT_TEST_SAMPLES,
        stratify=y,
        random_state=split_seed,
    )

    X_train, _, y_train, _ = train_test_split(
        X_pool,
        y_pool,
        train_size=PILOT_TRAIN_SAMPLES,
        stratify=y_pool,
        random_state=split_seed,
    )

    meta = {
        **meta,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "raw_features": int(X.shape[1]),
        "train_positive": int(np.sum(y_train == 1)),
        "train_negative": int(np.sum(y_train == 0)),
        "test_positive": int(np.sum(y_test == 1)),
        "test_negative": int(np.sum(y_test == 0)),
        "test_majority_accuracy_baseline": float(
            max(np.sum(y_test == 0), np.sum(y_test == 1)) / len(y_test)
        ),
        "test_positive_rate": float(np.mean(y_test == 1)),
    }
    return X_train, X_test, y_train, y_test, meta

def fit_quantum_preprocessor(X_train):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca = PCA(n_components=N_QUBITS, random_state=RANDOM_SEED)
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))

    X1 = imputer.fit_transform(X_train)
    X2 = scaler.fit_transform(X1)
    X3 = pca.fit_transform(X2)
    X4 = angle_scaler.fit_transform(X3)
    preproc = (imputer, scaler, pca, angle_scaler)
    return preproc, X4


def transform_quantum_features(X, preproc):
    imputer, scaler, pca, angle_scaler = preproc
    X1 = imputer.transform(X)
    X2 = scaler.transform(X1)
    X3 = pca.transform(X2)
    return angle_scaler.transform(X3)


def classical_preprocess(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(X_train))
    Xte = scaler.transform(imputer.transform(X_test))
    return Xtr, Xte


# ============================================================
# VQC construction
# ============================================================

def entanglement_edges(name: str, q: int, layer: int) -> List[Tuple[int, int]]:
    if name == "linear":
        return [(0, 1), (1, 2), (2, 3)]
    if name == "reverse_linear":
        return [(1, 0), (2, 1), (3, 2)]
    if name == "brickwall":
        if layer % 2 == 0:
            return [(0, 1), (2, 3)]
        return [(1, 2)]
    if name == "staggered":
        if layer % 2 == 0:
            return [(1, 2)]
        return [(0, 1), (2, 3)]
    raise ValueError(f"Unknown entanglement pattern: {name}")


def apply_rotation(qc: QuantumCircuit, gate: str, qubit: int, angle):
    if gate == "rx":
        qc.rx(angle, qubit)
    elif gate == "ry":
        qc.ry(angle, qubit)
    elif gate == "rz":
        qc.rz(angle, qubit)
    else:
        raise ValueError(gate)


def build_vqc(x, theta, spec: KernelSpec):
    x = np.asarray(x)
    theta = np.asarray(theta, dtype=float)
    per_layer = N_QUBITS * len(spec.rotation_pattern)
    expected = spec.depth * per_layer
    if theta.size != expected:
        raise ValueError(
            f"{spec.kernel_id}: expected {expected} parameters, got {theta.size}"
        )

    qc = QuantumCircuit(N_QUBITS)
    cursor = 0

    for layer in range(spec.depth):
        # Data re-uploading layer.
        for q in range(N_QUBITS):
            qc.ry(x[q], q)
            qc.rz(0.5 * x[q], q)

        # Trainable block.
        for q in range(N_QUBITS):
            for gate in spec.rotation_pattern:
                apply_rotation(qc, gate, q, float(theta[cursor]))
                cursor += 1

        # Entanglement.
        for a, b in entanglement_edges(spec.entanglement, N_QUBITS, layer):
            qc.cx(a, b)

    return qc


def build_symbolic_vqc(theta, spec: KernelSpec):
    x = ParameterVector(f"x_{spec.kernel_id}", N_QUBITS)
    theta = np.asarray(theta, dtype=float)

    per_layer = N_QUBITS * len(spec.rotation_pattern)
    expected = spec.depth * per_layer
    if theta.size != expected:
        raise ValueError(f"Parameter size mismatch for {spec.kernel_id}")

    qc = QuantumCircuit(N_QUBITS)
    cursor = 0

    for layer in range(spec.depth):
        for q in range(N_QUBITS):
            qc.ry(x[q], q)
            qc.rz(0.5 * x[q], q)
        for q in range(N_QUBITS):
            for gate in spec.rotation_pattern:
                apply_rotation(qc, gate, q, float(theta[cursor]))
                cursor += 1
        for a, b in entanglement_edges(spec.entanglement, N_QUBITS, layer):
            qc.cx(a, b)

    return qc, x


def parameter_count(spec: KernelSpec) -> int:
    return spec.depth * N_QUBITS * len(spec.rotation_pattern)


# ============================================================
# Ideal VQC training by centered kernel-target alignment
# ============================================================

def label_kernel(y):
    s = np.where(np.asarray(y) == 1, 1.0, -1.0)
    return np.outer(s, s)


def center_kernel(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def centered_alignment(K, Y):
    Kc = center_kernel(K)
    Yc = center_kernel(Y)
    denom = np.linalg.norm(Kc, "fro") * np.linalg.norm(Yc, "fro")
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(Kc * Yc) / denom)


def statevector_kernel_from_features(X, theta, spec):
    states = [Statevector.from_instruction(build_vqc(x, theta, spec)) for x in X]
    n = len(states)
    K = np.empty((n, n), dtype=float)
    arrays = [np.asarray(s.data, dtype=complex) for s in states]
    for i, a in enumerate(arrays):
        for j, b in enumerate(arrays):
            K[i, j] = float(abs(np.vdot(a, b)) ** 2)
    return np.clip(K, 0.0, 1.0)


def train_vqc_theta(X_train, y_train, spec, seed):
    """
    Train VQC parameters with SPSA.

    Powell was excessively expensive in the V2 pilot because each objective
    evaluation constructs a full kernel matrix. SPSA needs only two objective
    evaluations per iteration and is therefore much more suitable for the
    pilot-stage trainable VQC search.
    """
    rng = np.random.default_rng(seed)

    theta = rng.uniform(
        -0.15,
        0.15,
        size=parameter_count(spec),
    )

    Y = label_kernel(y_train)

    evaluations = 0
    best_theta = theta.copy()
    best_value = -np.inf

    def objective(current_theta):
        nonlocal evaluations
        evaluations += 1

        K = statevector_kernel_from_features(
            X_train,
            current_theta,
            spec,
        )

        return centered_alignment(K, Y)

    start = time.time()

    a0 = 0.08
    c0 = 0.12
    A = max(10.0, VQC_MAXITER / 10.0)
    alpha = 0.602
    gamma = 0.101

    for k in range(VQC_MAXITER):
        delta = rng.choice(
            [-1.0, 1.0],
            size=theta.size,
        )

        ak = a0 / ((k + 1 + A) ** alpha)
        ck = c0 / ((k + 1) ** gamma)

        theta_plus = np.clip(
            theta + ck * delta,
            -np.pi,
            np.pi,
        )

        theta_minus = np.clip(
            theta - ck * delta,
            -np.pi,
            np.pi,
        )

        y_plus = objective(theta_plus)
        y_minus = objective(theta_minus)

        ghat = (
            (y_plus - y_minus)
            / (2.0 * ck)
        ) * delta

        theta = np.clip(
            theta + ak * ghat,
            -np.pi,
            np.pi,
        )

        current_value = objective(theta)

        if current_value > best_value:
            best_value = current_value
            best_theta = theta.copy()

    elapsed = time.time() - start

    return (
        np.asarray(best_theta, dtype=float),
        {
            "success": True,
            "completed": True,
            "terminated_by_budget": True,
            "message": "SPSA completed fixed iteration budget",
            "iterations": int(VQC_MAXITER),
            "function_evaluations": int(evaluations),
            "elapsed_seconds": float(elapsed),
            "kta": float(best_value),
            "optimizer": "SPSA",
        },
    )

def connect_ibm():
    try:
        return QiskitRuntimeService(instance=IBM_INSTANCE)
    except Exception:
        try:
            return QiskitRuntimeService()
        except Exception as exc:
            raise RuntimeError(
                "IBM authentication failed. Save credentials locally or provide "
                "a valid IBM account configuration."
            ) from exc


def iso(dt):
    return dt.isoformat() if hasattr(dt, "isoformat") else str(dt)


def get_current_properties(backend):
    try:
        return backend.properties(refresh=True)
    except TypeError:
        return backend.properties()


def fetch_calibration_profiles(backend):
    profiles = []

    current = get_current_properties(backend)
    if current is None:
        raise RuntimeError("Current IBM backend properties are unavailable.")
    current_ts = getattr(current, "last_update_date", None)
    profiles.append(("CURRENT", current))

    for profile_id, timestamp_text in HISTORICAL_REQUESTS:
        target_dt = datetime.fromisoformat(timestamp_text)
        print(f"[IBM] {profile_id} request: {timestamp_text}")
        try:
            props = backend.properties(datetime=target_dt)
        except Exception as exc:
            warnings.warn(f"Historical properties unavailable for {profile_id}: {exc}")
            continue
        if props is None:
            warnings.warn(f"IBM returned None for {profile_id}")
            continue
        returned_ts = getattr(props, "last_update_date", None)
        print(f"[IBM] {profile_id} returned: {iso(returned_ts)}")
        profiles.append((profile_id, props))

    return profiles, current_ts


# ============================================================
# Backend property extraction
# ============================================================

def nduv_value(properties, qubit: int, name: str):
    try:
        records = properties.qubits[qubit]
    except Exception:
        return None

    for nd in records:
        nd_name = str(getattr(nd, "name", ""))
        if nd_name.lower() != name.lower():
            continue
        value = getattr(nd, "value", None)
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        unit = str(getattr(nd, "unit", "") or "").lower()
        if name.lower() in {"t1", "t2"}:
            if unit == "us":
                value *= 1e-6
            elif unit == "ns":
                value *= 1e-9
            elif unit == "ms":
                value *= 1e-3
        return value
    return None


def gate_error_map(properties):
    result = {}
    for gate in getattr(properties, "gates", []):
        name = str(getattr(gate, "gate", ""))
        qubits = tuple(int(q) for q in getattr(gate, "qubits", []))
        err = None
        for param in getattr(gate, "parameters", []):
            p_name = str(getattr(param, "name", ""))
            if p_name == "gate_error" or "error" in p_name.lower():
                try:
                    err = float(getattr(param, "value"))
                except (TypeError, ValueError):
                    pass
                break
        if err is not None:
            result[(name, qubits)] = err
    return result


def profile_statistics(properties, physical_qubits):
    selected = set(physical_qubits)
    one_q = []
    two_q = []
    readout = []
    t1 = []
    t2 = []

    for (name, qubits), err in gate_error_map(properties).items():
        if not all(q in selected for q in qubits):
            continue
        if len(qubits) == 1:
            one_q.append(err)
        elif len(qubits) == 2:
            two_q.append(err)

    for q in physical_qubits:
        for pname, store in [
            ("readout_error", readout),
            ("T1", t1),
            ("T2", t2),
        ]:
            value = nduv_value(properties, q, pname)
            if value is not None:
                store.append(value)

    med = lambda a: float(np.median(a)) if a else None
    return {
        "one_qubit_gate_error_median": med(one_q),
        "two_qubit_gate_error_median": med(two_q),
        "readout_error_median": med(readout),
        "t1_median_s": med(t1),
        "t2_median_s": med(t2),
    }


def validate_fixed_qubits(properties, backend):
    if len(FIXED_PHYSICAL_QUBITS) != N_QUBITS:
        raise RuntimeError("FIXED_PHYSICAL_QUBITS must contain exactly four qubits.")

    if len(set(FIXED_PHYSICAL_QUBITS)) != N_QUBITS:
        raise RuntimeError("FIXED_PHYSICAL_QUBITS contains duplicates.")

    if max(FIXED_PHYSICAL_QUBITS) >= backend.num_qubits:
        raise RuntimeError("A fixed physical qubit is outside backend size.")

    # Validate that the reference calibration contains the selected qubits and
    # at least the chain edges required by the VQC layouts.
    edge_names = set()
    for gate in getattr(properties, "gates", []):
        q = tuple(int(v) for v in getattr(gate, "qubits", []))
        if len(q) == 2:
            edge_names.add(tuple(sorted(q)))

    chain = [tuple(sorted((FIXED_PHYSICAL_QUBITS[i], FIXED_PHYSICAL_QUBITS[i + 1])))
             for i in range(N_QUBITS - 1)]
    missing = [e for e in chain if e not in edge_names]
    if missing:
        raise RuntimeError(
            f"Fixed physical qubits do not form the required chain in the reference "
            f"calibration. Missing edges: {missing}"
        )


def build_noise_model(properties, physical_qubits):
    mapping = {int(p): i for i, p in enumerate(physical_qubits)}
    model = NoiseModel()

    kwargs = dict(
        properties=properties,
        gate_error=True,
        thermal_relaxation=True,
    )
    try:
        gate_errors = basic_device_gate_errors(**kwargs, temperature=0.0)
    except TypeError:
        gate_errors = basic_device_gate_errors(**kwargs)

    added = set()
    for name, qubits, error in gate_errors:
        phys = tuple(int(q) for q in qubits)
        if not all(q in mapping for q in phys):
            continue
        logical = tuple(mapping[q] for q in phys)
        key = (str(name), logical)
        if key in added:
            continue
        model.add_quantum_error(error, str(name), list(logical))
        added.add(key)

    try:
        ro_errors = basic_device_readout_errors(properties=properties)
        for qubits, error in ro_errors:
            phys = tuple(int(q) for q in qubits)
            if len(phys) == 1 and phys[0] in mapping:
                model.add_readout_error(error, [mapping[phys[0]]])
    except Exception as exc:
        warnings.warn(f"Readout-error extraction skipped: {exc}")

    if not getattr(model, "noise_instructions", None):
        raise RuntimeError("Restricted IBM noise model contains no noisy instructions.")

    return model


# ============================================================
# Hardware target compilation
# ============================================================

def compact_transpiled_circuit(transpiled_circuit, physical_qubits):
    mapping = {int(p): i for i, p in enumerate(physical_qubits)}
    compact = QuantumCircuit(N_QUBITS)

    used_physical = set()
    for item in transpiled_circuit.data:
        op = item.operation
        if op.name in {"barrier", "measure", "reset", "delay"}:
            continue
        for qb in item.qubits:
            idx = transpiled_circuit.find_bit(qb).index
            used_physical.add(idx)
            if idx not in mapping:
                raise RuntimeError(
                    "Hardware transpilation used an additional physical qubit "
                    f"{idx}, outside the frozen set {physical_qubits}."
                )

        qargs = [mapping[transpiled_circuit.find_bit(qb).index] for qb in item.qubits]
        compact.append(op.copy() if hasattr(op, "copy") else op, qargs)

    return compact, used_physical


def compile_kernel_template(backend, theta, spec):
    symbolic, x_params = build_symbolic_vqc(theta, spec)

    transpiled = transpile(
        symbolic,
        backend,
        initial_layout=FIXED_PHYSICAL_QUBITS,
        optimization_level=1,
    )

    compact, used_physical = compact_transpiled_circuit(
        transpiled,
        FIXED_PHYSICAL_QUBITS,
    )

    # Check no unexpected multi-qubit routing operations were introduced.
    if len(used_physical) > N_QUBITS:
        raise RuntimeError("Unexpected routing outside frozen physical qubits.")

    return compact, x_params, transpiled


def bind_compact_circuit(template, x_params, x):
    bind = {p: float(v) for p, v in zip(x_params, np.asarray(x, dtype=float))}
    return template.assign_parameters(bind, inplace=False)


# ============================================================
# Kernel generation
# ============================================================

def statevectors_from_templates(X, compiled_templates):
    states = {}
    for kid, (template, x_params) in compiled_templates.items():
        states[kid] = []
        for x in X:
            qc = bind_compact_circuit(template, x_params, x)
            states[kid].append(Statevector.from_instruction(qc))
    return states


def statevector_kernel(states_a, states_b=None):
    if states_b is None:
        states_b = states_a
    A = [np.asarray(s.data, dtype=complex) for s in states_a]
    B = [np.asarray(s.data, dtype=complex) for s in states_b]
    K = np.empty((len(A), len(B)), dtype=float)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            K[i, j] = float(abs(np.vdot(a, b)) ** 2)
    return np.clip(K, 0.0, 1.0)


def density_matrices_for_profile(X, compiled_templates, profile):
    result = {}
    sim = profile.simulator
    for kid, (template, x_params) in compiled_templates.items():
        states = []
        for x in X:
            qc = bind_compact_circuit(template, x_params, x)
            qc.save_density_matrix()
            data = sim.run(qc).result().data(0)["density_matrix"]
            states.append(DensityMatrix(data))
        result[kid] = states
    return result


def density_kernel(states_a, states_b=None):
    if states_b is None:
        states_b = states_a
    K = np.empty((len(states_a), len(states_b)), dtype=float)
    for i, a in enumerate(states_a):
        for j, b in enumerate(states_b):
            K[i, j] = float(state_fidelity(a, b, validate=False))
    return np.clip(K, 0.0, 1.0)


def psd_project_kernel(K, eigen_floor=1e-10):
    """Project a symmetric training Gram matrix onto the PSD cone.

    Mixed-state fidelity similarities can exhibit small negative eigenvalues.
    The projection is applied to the training Gram matrix used by SVC so the
    optimization operates on a valid PSD kernel. A final diagonal rescaling
    restores unit self-similarity while preserving positive semidefiniteness.
    """
    S = 0.5 * (np.asarray(K, dtype=float) + np.asarray(K, dtype=float).T)
    eigvals, eigvecs = np.linalg.eigh(S)
    clipped = np.maximum(eigvals, float(eigen_floor))
    P = (eigvecs * clipped) @ eigvecs.T
    P = 0.5 * (P + P.T)

    diag = np.clip(np.diag(P), float(eigen_floor), None)
    scale = np.sqrt(diag)
    P = P / np.outer(scale, scale)
    P = 0.5 * (P + P.T)
    return P, eigvals, np.linalg.eigvalsh(P), float(np.linalg.norm(P - S, "fro"))


# ============================================================
# Kernel normalization and drift
# ============================================================

def frob_norm(K):
    n = float(np.linalg.norm(K, "fro"))
    if n <= 1e-12:
        return K.copy(), n
    return K / n, n


def normalized_coalition_train(kernel_bank, ids):
    raw = np.zeros_like(next(iter(kernel_bank.values())))
    for kid in ids:
        raw += kernel_bank[kid]
    out, scale = frob_norm(raw)
    return out, scale


def normalized_singleton_train(kernel_bank, kid):
    return frob_norm(kernel_bank[kid])[0]


def relative_kernel_drift(K, K_ref):
    denom = float(np.linalg.norm(K_ref, "fro"))
    if denom <= 1e-12:
        return 0.0
    return float(np.linalg.norm(K - K_ref, "fro") / denom)


def kernel_alignment_to_reference(K, K_ref):
    return centered_alignment(K, K_ref)


# ============================================================
# Fold utilities
# ============================================================

def make_folds(y, n_splits, seed):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros((len(y), 1)), y))


def fold_predictions_and_utility(
    K,
    y,
    train_idx,
    val_idx,
):
    Ktr = K[np.ix_(train_idx, train_idx)]
    Kva = K[np.ix_(val_idx, train_idx)]

    clf = SVC(
        kernel="precomputed",
        C=1.0,
    )

    clf.fit(
        Ktr,
        y[train_idx],
    )

    pred = clf.predict(Kva)

    try:
        decision = clf.decision_function(Kva)

        if len(np.unique(y[val_idx])) == 2:
            auc = float(
                roc_auc_score(
                    y[val_idx],
                    decision,
                )
            )
        else:
            auc = float("nan")

    except Exception:
        auc = float("nan")

    f1 = float(
        f1_score(
            y[val_idx],
            pred,
            zero_division=0,
        )
    )

    return f1, auc


def fold_f1_for_kernel(
    K,
    y,
    train_idx,
    val_idx,
):
    f1, _ = fold_predictions_and_utility(
        K,
        y,
        train_idx,
        val_idx,
    )
    return f1


def fold_utility_for_kernel(
    K,
    y,
    train_idx,
    val_idx,
):
    f1, auc = fold_predictions_and_utility(
        K,
        y,
        train_idx,
        val_idx,
    )

    if np.isfinite(auc):
        return auc

    return f1


def fold_majority_f1(y, train_idx, val_idx):
    majority = int(
        np.bincount(
            y[train_idx]
        ).argmax()
    )

    pred = np.full(
        len(val_idx),
        majority,
        dtype=int,
    )

    return float(
        f1_score(
            y[val_idx],
            pred,
            zero_division=0,
        )
    )


def fold_majority_utility(y, train_idx, val_idx):
    if len(np.unique(y[val_idx])) == 2:
        return 0.5

    return fold_majority_f1(
        y,
        train_idx,
        val_idx,
    )


# ============================================================
# Distributional contribution + pair synergy
# ============================================================

def attribution_for_profile(kernel_bank, y, seed_base):
    """
    Compute three complementary quantities at fold level:

    1) Marginal contribution:
         Delta_i = U(K_all) - U(K_-i)

    2) Canonical pairwise synergy:
         Gamma_ij = U(K_ij) - U(K_i) - U(K_j) + U(empty)

    3) The canonical synergy is retained as a structural diagnostic.
       Interaction selection itself is handled later in the LOCO stage by
       an optimized-eta incremental Hadamard gain measured relative to the
       reliability-selected base fusion.

    Utility is ROC-AUC from the SVM decision_function when available;
    F1 remains recorded and is used only as a fallback if ROC-AUC is undefined.
    """
    ids = list(kernel_bank.keys())
    all_set = frozenset(ids)

    contributions = {kid: [] for kid in ids}

    pair_list = [
        (ids[i], ids[j])
        for i in range(len(ids))
        for j in range(i + 1, len(ids))
    ]

    synergies = {
        f"{a}x{b}": []
        for a, b in pair_list
    }

    diagnostic = []

    for repeat in range(ATTRIBUTION_REPEATS):
        folds = make_folds(
            y,
            ATTRIBUTION_FOLDS,
            seed_base + 101 * repeat,
        )

        for fold_no, (train_idx, val_idx) in enumerate(folds):
            coalitions = {all_set}

            for kid in ids:
                coalitions.add(frozenset([kid]))
                coalitions.add(frozenset(set(ids) - {kid}))

            for a, b in pair_list:
                coalitions.add(frozenset([a, b]))

            score_cache = {}
            f1_cache = {}
            utility_values = []
            prediction_signatures = set()
            rank_signatures = set()
            decision_ranges = []

            for coalition in coalitions:
                coalition_ids = sorted(coalition)

                Kc, _ = normalized_coalition_train(
                    kernel_bank,
                    coalition_ids,
                )

                Ktr = Kc[np.ix_(train_idx, train_idx)]
                Kva = Kc[np.ix_(val_idx, train_idx)]

                clf = SVC(
                    kernel="precomputed",
                    C=1.0,
                )
                clf.fit(
                    Ktr,
                    y[train_idx],
                )

                pred = clf.predict(Kva)
                decision = np.asarray(
                    clf.decision_function(Kva),
                    dtype=float,
                ).reshape(-1)

                f1 = float(
                    f1_score(
                        y[val_idx],
                        pred,
                        zero_division=0,
                    )
                )

                try:
                    if len(np.unique(y[val_idx])) == 2:
                        auc = float(
                            roc_auc_score(
                                y[val_idx],
                                decision,
                            )
                        )
                    else:
                        auc = float("nan")
                except Exception:
                    auc = float("nan")

                utility = (
                    auc
                    if np.isfinite(auc)
                    else f1
                )

                score_cache[coalition] = float(utility)
                f1_cache[coalition] = float(f1)
                utility_values.append(float(utility))
                prediction_signatures.add(tuple(pred.tolist()))

                if decision.size:
                    decision_ranges.append(
                        float(np.max(decision) - np.min(decision))
                    )
                    rank_order = tuple(
                        np.argsort(
                            -decision,
                            kind="stable",
                        ).tolist()
                    )
                    rank_signatures.add(rank_order)

            empty_utility = fold_majority_utility(
                y,
                train_idx,
                val_idx,
            )

            # ------------------------------------------------
            # Marginal contribution
            # ------------------------------------------------
            for kid in ids:
                without = frozenset(
                    set(ids) - {kid}
                )
                contributions[kid].append(
                    float(
                        score_cache[all_set]
                        - score_cache[without]
                    )
                )

            # ------------------------------------------------
            # Canonical pairwise synergy
            # ------------------------------------------------
            for a, b in pair_list:
                key = f"{a}x{b}"
                pair = frozenset([a, b])

                gamma = (
                    score_cache[pair]
                    - score_cache[frozenset([a])]
                    - score_cache[frozenset([b])]
                    + empty_utility
                )

                synergies[key].append(
                    float(gamma)
                )

            # Utility-focused diagnostic for all coalition evaluations.
            utility_array = np.asarray(
                utility_values,
                dtype=float,
            )

            diagnostic.append(
                {
                    "repeat": repeat,
                    "fold": fold_no,
                    "n_coalitions": len(coalitions),
                    "n_unique_prediction_patterns": len(prediction_signatures),
                    "n_unique_utility_values": len(
                        np.unique(
                            np.round(
                                utility_array,
                                12,
                            )
                        )
                    ),
                    "utility_min": float(
                        np.min(utility_array)
                    ),
                    "utility_max": float(
                        np.max(utility_array)
                    ),
                    "utility_std": float(
                        np.std(utility_array)
                    ),
                    "mean_decision_range": float(
                        np.mean(decision_ranges)
                    ) if decision_ranges else 0.0,
                    "n_unique_decision_rank_patterns": len(rank_signatures),
                }
            )

    return (
        contributions,
        synergies,
        diagnostic,
    )


# ============================================================
# Continuous reliability base prior + Hadamard interaction gain
# ============================================================

def reliability_score(summary):
    """Continuous reliability score from lower-tail quantiles.

    R = alpha * q10 + (1-alpha) * q25.
    Higher values indicate a more favorable lower-tail contribution profile.
    """
    q10 = float(summary.get("q10", float("nan")))
    q25 = float(summary.get("q25", float("nan")))
    if not np.isfinite(q10) and not np.isfinite(q25):
        return float("nan")
    if not np.isfinite(q10):
        q10 = q25
    if not np.isfinite(q25):
        q25 = q10
    return float(
        RELIABILITY_SCORE_ALPHA * q10
        + (1.0 - RELIABILITY_SCORE_ALPHA) * q25
    )


def softmax(values, temperature=1.0):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if not np.all(np.isfinite(arr)):
        finite = arr[np.isfinite(arr)]
        fill = float(np.min(finite)) if finite.size else 0.0
        arr = np.where(np.isfinite(arr), arr, fill)
    scale = float(np.std(arr))
    if scale > 1e-12:
        z = (arr - float(np.mean(arr))) / scale
    else:
        z = arr - float(np.mean(arr))
    t = max(float(temperature), 1e-6)
    z = z / t
    z = z - np.max(z)
    expz = np.exp(np.clip(z, -60.0, 60.0))
    total = float(np.sum(expz))
    if total <= 1e-12 or not np.isfinite(total):
        return np.full(arr.size, 1.0 / arr.size)
    return expz / total


def continuous_reliability_weights(summaries, ids):
    """Convert reliability scores to normalized continuous base weights."""
    scores = [reliability_score(summaries[k]) for k in ids]
    weights = softmax(scores, RELIABILITY_SOFTMAX_TEMPERATURE)
    return {k: float(w) for k, w in zip(ids, weights)}


def reliability_prior_for_components(component_names, base_reliability, interaction_reliability):
    """Build a reliability-informed simplex prior used only for initialization/ties.

    Base components receive the full prior mass unless at least one interaction has
    positive reliability score. In that case a small adaptive interaction mass is
    seeded proportional to the positive interaction reliability while never exceeding
    INTERACTION_WEIGHT_CAP. This avoids forcing negative-reliability interactions into
    the model while still giving reliable interactions a data-informed starting point.
    """
    base_names = [k for k in component_names if "x" not in k]
    pair_names = [k for k in component_names if "x" in k]
    base_scores = np.asarray(
        [float(base_reliability.get(k, 0.0)) for k in base_names],
        dtype=float,
    )
    base_prior = softmax(base_scores, RELIABILITY_SOFTMAX_TEMPERATURE)

    prior = np.zeros(len(component_names), dtype=float)
    name_to_idx = {name: i for i, name in enumerate(component_names)}

    positive_inter = np.asarray(
        [max(float(interaction_reliability.get(k, 0.0)), 0.0) for k in pair_names],
        dtype=float,
    )
    if positive_inter.size and float(np.max(positive_inter)) > 0.0:
        # Adaptive seed mass: up to the cap, but small on weak positive signals.
        ref = max(float(np.max(positive_inter)), 1e-12)
        interaction_mass = float(
            min(
                INTERACTION_WEIGHT_CAP,
                0.20 * (float(np.mean(positive_inter)) / ref),
            )
        )
        interaction_mass = max(0.05, interaction_mass)
        interaction_prior = positive_inter / float(np.sum(positive_inter))
    else:
        interaction_mass = 0.0
        interaction_prior = np.zeros(len(pair_names), dtype=float)

    base_mass = 1.0 - interaction_mass
    for k, w in zip(base_names, base_prior):
        prior[name_to_idx[k]] = base_mass * float(w)
    for k, w in zip(pair_names, interaction_prior):
        prior[name_to_idx[k]] = interaction_mass * float(w)

    prior = np.clip(prior, 0.0, None)
    total = float(np.sum(prior))
    if total <= 1e-12:
        prior[:] = 1.0 / len(component_names)
    else:
        prior /= total
    return prior


def make_probe_base_components(kernel_bank, base_ids, base_weights=None):
    """Build a continuous reliability-weighted base fusion."""
    components = {}
    for kid in base_ids:
        K, _ = frob_norm(kernel_bank[kid])
        components[kid] = K

    if base_weights is None:
        weights = np.full(len(base_ids), 1.0 / len(base_ids))
    else:
        weights = np.asarray([float(base_weights[k]) for k in base_ids], dtype=float)
        total = float(np.sum(weights))
        if total <= 1e-12:
            weights = np.full(len(base_ids), 1.0 / len(base_ids))
        else:
            weights /= total

    K_base = weighted_sum(weights, base_ids, components)
    K_base, base_scale = frob_norm(K_base)
    return K_base, base_scale


def make_probe_interaction_components(kernel_bank, pair):
    """Build a normalized singleton Hadamard interaction H_ij."""
    a, b = pair
    Ka, scale_a = frob_norm(kernel_bank[a])
    Kb, scale_b = frob_norm(kernel_bank[b])
    H_raw = Ka * Kb
    H, hscale = frob_norm(H_raw)
    return H, scale_a, scale_b, hscale


def interaction_probe_utility(K_base, H, eta, y, train_idx, val_idx):
    """Utility of a bounded convex probe between base fusion and H_ij."""
    K_mix = (1.0 - float(eta)) * K_base + float(eta) * H
    K_mix, _ = frob_norm(K_mix)
    return fold_utility_for_kernel(K_mix, y, train_idx, val_idx)


def learn_interaction_eta(
    profile_ids,
    kernel_banks,
    y,
    base_ids,
    base_weights,
    pair,
    seed,
):
    """Select probe eta from a small grid using only LOCO training profiles."""
    candidate_scores = []
    for eta in INTERACTION_ETA_GRID:
        scores = []
        for profile_offset, pid in enumerate(profile_ids):
            K_base, _ = make_probe_base_components(
                kernel_banks[pid],
                base_ids,
                base_weights,
            )
            H, _, _, _ = make_probe_interaction_components(
                kernel_banks[pid],
                pair,
            )
            folds = make_folds(
                y,
                INTERACTION_ETA_SELECTION_FOLDS,
                seed + 17 * profile_offset,
            )
            for train_idx, val_idx in folds:
                scores.append(
                    interaction_probe_utility(
                        K_base,
                        H,
                        eta,
                        y,
                        train_idx,
                        val_idx,
                    )
                )

        candidate_scores.append({
            "eta": float(eta),
            "mean_utility": float(np.mean(scores)) if scores else float("nan"),
            "std_utility": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "n": int(len(scores)),
        })

    valid = [row for row in candidate_scores if np.isfinite(row["mean_utility"])]
    if not valid:
        return float(INTERACTION_ETA_GRID[0]), candidate_scores

    best = max(valid, key=lambda row: (row["mean_utility"], -row["eta"]))
    return float(best["eta"]), candidate_scores


def interaction_gain_for_profile(
    kernel_bank,
    y,
    base_ids,
    base_weights,
    pair,
    eta,
    seed_base,
):
    """Measure OOF incremental gain relative to continuous reliability base fusion."""
    K_base, _ = make_probe_base_components(
        kernel_bank,
        base_ids,
        base_weights,
    )
    H, _, _, _ = make_probe_interaction_components(kernel_bank, pair)

    gains = []
    diagnostics = []
    for repeat in range(ATTRIBUTION_REPEATS):
        folds = make_folds(
            y,
            ATTRIBUTION_FOLDS,
            seed_base + 101 * repeat,
        )
        for fold_no, (train_idx, val_idx) in enumerate(folds):
            base_utility = fold_utility_for_kernel(
                K_base,
                y,
                train_idx,
                val_idx,
            )
            augmented_utility = interaction_probe_utility(
                K_base,
                H,
                eta,
                y,
                train_idx,
                val_idx,
            )
            gain = float(augmented_utility - base_utility)
            gains.append(gain)
            diagnostics.append({
                "repeat": int(repeat),
                "fold": int(fold_no),
                "interaction": f"{pair[0]}x{pair[1]}",
                "eta": float(eta),
                "base_utility": float(base_utility),
                "augmented_utility": float(augmented_utility),
                "incremental_gain": gain,
                "base_kernel_count": int(len(base_ids)),
            })
    return gains, diagnostics


# ============================================================
# Nonlinear interaction components
# ============================================================

def make_train_components(kernel_bank, base_ids, pairs):
    components = {}
    scales = {}

    for kid in base_ids:
        K, scale = frob_norm(kernel_bank[kid])
        components[kid] = K
        scales[kid] = {"type": "base", "scale": scale}

    for a, b in pairs:
        Ka, _ = frob_norm(kernel_bank[a])
        Kb, _ = frob_norm(kernel_bank[b])
        Hraw = Ka * Kb
        H, hscale = frob_norm(Hraw)
        name = f"{a}x{b}"
        components[name] = H
        scales[name] = {"type": "hadamard", "scale": hscale}

    return components, scales


def make_cross_components(kernel_bank_train, kernel_bank_test, base_ids, pairs):
    # Training-side scales must be used for corresponding test/train cross kernels.
    test_components = {}

    train_norm = {}
    train_scales = {}
    for kid, K in kernel_bank_train.items():
        Kn, s = frob_norm(K)
        train_norm[kid] = Kn
        train_scales[kid] = s

    for kid in base_ids:
        if train_scales[kid] <= 1e-12:
            test_components[kid] = kernel_bank_test[kid].copy()
        else:
            test_components[kid] = kernel_bank_test[kid] / train_scales[kid]

    for a, b in pairs:
        raw = train_norm[a] * train_norm[b]
        hscale = float(np.linalg.norm(raw, "fro"))
        cross = (
            (kernel_bank_test[a] / train_scales[a])
            * (kernel_bank_test[b] / train_scales[b])
        )
        if hscale > 1e-12:
            cross = cross / hscale
        test_components[f"{a}x{b}"] = cross

    return test_components


def weighted_sum(weights, component_names, components):
    K = np.zeros_like(next(iter(components.values())))
    for w, name in zip(weights, component_names):
        K += float(w) * components[name]
    return K


# ============================================================# Structural diagnostic: pairwise kernel distinctiveness
# ============================================================

def kernel_relationship_matrix(kernel_bank):

    ids = list(kernel_bank.keys())
    n = len(ids)

    alignment = np.eye(n)
    frob_distance = np.zeros(
        (n, n),
        dtype=float,
    )

    centered = {
        kid: center_kernel(
            kernel_bank[kid]
        )
        for kid in ids
    }

    for i in range(n):
        for j in range(i + 1, n):

            ai = centered[ids[i]]
            aj = centered[ids[j]]

            den = (
                np.linalg.norm(ai, "fro")
                *
                np.linalg.norm(aj, "fro")
            )

            if den <= 1e-12:
                a = 0.0
            else:
                a = float(
                    np.sum(ai * aj)
                    / den
                )

            alignment[i, j] = a
            alignment[j, i] = a

            ni = np.linalg.norm(
                kernel_bank[ids[i]],
                "fro",
            )

            nj = np.linalg.norm(
                kernel_bank[ids[j]],
                "fro",
            )

            if (
                ni <= 1e-12
                or nj <= 1e-12
            ):
                d = 0.0
            else:
                d = float(
                    np.linalg.norm(
                        (
                            kernel_bank[ids[i]]
                            / ni
                        )
                        -
                        (
                            kernel_bank[ids[j]]
                            / nj
                        )
                    )
                )

            frob_distance[i, j] = d
            frob_distance[j, i] = d

    return (
        ids,
        alignment,
        frob_distance,
    )


# ============================================================
# Robust fusion optimizer
# ============================================================

def robust_objective(
    weights,
    component_names,
    profile_components,
    y,
    fold_cache,
    interaction_names=None,
):
    """Continuous robust ROC-AUC objective with an interaction-mass constraint.

    Objective = mean OOF ROC-AUC - lambda * std(OOF ROC-AUC).
    The interaction cap is enforced as an SLSQP constraint rather than embedded
    in the objective so that the reported objective remains directly interpretable.
    """
    w = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(w)):
        return 1e6
    if np.min(w) < -1e-8:
        return 1e6
    if abs(float(np.sum(w)) - 1.0) > 1e-6:
        return 1e6

    if interaction_names:
        interaction_mass = float(
            np.sum(
                [w[component_names.index(name)] for name in interaction_names]
            )
        )
        if interaction_mass > INTERACTION_WEIGHT_CAP + 1e-8:
            return 1e6

    scores = []
    for profile_id, components in profile_components.items():
        K = weighted_sum(w, component_names, components)
        for train_idx, val_idx in fold_cache[profile_id]:
            scores.append(fold_utility_for_kernel(K, y, train_idx, val_idx))

    if not scores:
        return 1e6

    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1e6

    utility = float(np.mean(arr))
    if arr.size > 1:
        utility -= ROBUSTNESS_LAMBDA * float(np.std(arr, ddof=1))
    return -utility


def _valid_start(weights, interaction_names, component_names):
    w = np.asarray(weights, dtype=float).copy()
    w = np.clip(w, 0.0, None)
    if w.size != len(component_names):
        return None
    total = float(np.sum(w))
    if total <= 1e-12:
        return None
    w /= total
    if interaction_names:
        idx = [component_names.index(name) for name in interaction_names]
        mass = float(np.sum(w[idx]))
        if mass > INTERACTION_WEIGHT_CAP:
            target = INTERACTION_WEIGHT_CAP
            if mass > 1e-12:
                w[idx] *= target / mass
                base_idx = [i for i in range(len(component_names)) if i not in idx]
                base_mass = float(np.sum(w[base_idx]))
                if base_mass <= 1e-12:
                    w[base_idx] = (1.0 - target) / len(base_idx)
                else:
                    w[base_idx] *= (1.0 - target) / base_mass
    total = float(np.sum(w))
    return w / total


def optimize_weights(
    component_names,
    profile_components,
    y,
    seed_base,
    reliability_prior=None,
    interaction_names=None,
):
    """Multi-start robust weight learning with continuous reliability prior.

    The optimizer does not hard-select kernels. Reliability produces a soft prior
    from lower-tail contribution statistics; this prior is included as a start and
    as a deterministic tie-break among solutions with indistinguishable objective
    values. The primary objective is still robust cross-calibration ROC-AUC.
    """
    component_names = list(component_names)
    interaction_names = list(interaction_names or [])
    m = len(component_names)
    profile_ids = list(profile_components.keys())
    fold_cache = {
        pid: make_folds(y, ROBUSTNESS_FOLDS, seed_base + i * 17)
        for i, pid in enumerate(profile_ids)
    }

    if m == 1:
        w = np.ones(1, dtype=float)
        utility = -robust_objective(
            w,
            component_names,
            profile_components,
            y,
            fold_cache,
            interaction_names,
        )
        return w, {
            "success": True,
            "iterations": 0,
            "message": "single component",
            "elapsed_seconds": 0.0,
            "objective": -utility,
            "optimization_utility": ROBUST_OPTIMIZATION_UTILITY,
            "n_starts": 1,
            "best_start": 0,
            "uniform_robust_utility": utility,
            "best_robust_utility": utility,
            "improvement_over_uniform": 0.0,
            "prior_distance": 0.0,
            "start_logs": [],
        }

    base_names = [n for n in component_names if n not in interaction_names]
    base_indices = [component_names.index(n) for n in base_names]
    inter_indices = [component_names.index(n) for n in interaction_names]

    # Uniform-all-components start respecting the interaction mass cap.
    uniform = np.zeros(m, dtype=float)
    if inter_indices:
        inter_mass = min(INTERACTION_WEIGHT_CAP, 0.20)
        uniform[base_indices] = (1.0 - inter_mass) / len(base_indices)
        uniform[inter_indices] = inter_mass / len(inter_indices)
    else:
        uniform[:] = 1.0 / m

    if reliability_prior is not None:
        prior = _valid_start(
            reliability_prior,
            interaction_names,
            component_names,
        )
        if prior is None:
            prior = uniform.copy()
    else:
        prior = uniform.copy()

    starts = [uniform.copy(), prior.copy()]

    # Base-only reliability start: trust reliability but do not force interactions.
    base_only = np.zeros(m, dtype=float)
    if base_names:
        base_prior = prior[base_indices].copy()
        total_base_prior = float(np.sum(base_prior))
        if total_base_prior > 1e-12:
            base_prior /= total_base_prior
        else:
            base_prior[:] = 1.0 / len(base_indices)
        base_only[base_indices] = base_prior
        starts.append(base_only)

    rng = np.random.default_rng(seed_base)

    # Explore sparse base vertices.
    for idx in base_indices[: min(5, len(base_indices))]:
        v = np.zeros(m, dtype=float)
        v[idx] = 1.0
        starts.append(v)

    # Reliability-guided interaction starts: one interaction gets up to the cap,
    # remaining mass follows the reliability base prior.
    if inter_indices:
        for idx in inter_indices[: min(3, len(inter_indices))]:
            s = base_only.copy()
            inter_mass = min(0.20, INTERACTION_WEIGHT_CAP)
            s *= (1.0 - inter_mass)
            s[idx] = inter_mass
            starts.append(s)

    # Random interior simplex starts. The interaction group is sampled with a
    # bounded total mass to respect the model's complexity constraint.
    remaining = max(0, int(FUSION_MULTI_STARTS) - len(starts))
    for _ in range(remaining):
        if inter_indices:
            inter_mass = float(
                rng.uniform(0.0, INTERACTION_WEIGHT_CAP)
            )
            base_part = rng.dirichlet(
                np.full(len(base_indices), float(FUSION_DIRICHLET_ALPHA))
            )
            inter_part = rng.dirichlet(
                np.full(len(inter_indices), float(FUSION_DIRICHLET_ALPHA))
            ) if len(inter_indices) else np.array([])
            s = np.zeros(m, dtype=float)
            s[base_indices] = (1.0 - inter_mass) * base_part
            if inter_indices:
                s[inter_indices] = inter_mass * inter_part
        else:
            s = rng.dirichlet(
                np.full(m, float(FUSION_DIRICHLET_ALPHA))
            )
        starts.append(s)

    cleaned = []
    for s in starts:
        v = _valid_start(s, interaction_names, component_names)
        if v is None:
            continue
        if not any(np.max(np.abs(v - old)) < 1e-10 for old in cleaned):
            cleaned.append(v)
    starts = cleaned

    bounds = [(0.0, 1.0)] * m
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]
    if inter_indices:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: INTERACTION_WEIGHT_CAP - np.sum(w[inter_indices]),
            }
        )

    uniform_obj = robust_objective(
        uniform,
        component_names,
        profile_components,
        y,
        fold_cache,
        interaction_names,
    )
    uniform_utility = -float(uniform_obj)

    start_time = time.time()
    results = []
    start_logs = []

    for start_idx, x0 in enumerate(starts):
        try:
            result = minimize(
                robust_objective,
                x0,
                args=(component_names, profile_components, y, fold_cache, interaction_names),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": FUSION_MAXITER,
                    "ftol": FUSION_FTOL,
                    "disp": False,
                },
            )
            x = _valid_start(result.x, interaction_names, component_names)
            valid = x is not None
            if valid:
                fun = float(
                    robust_objective(
                        x,
                        component_names,
                        profile_components,
                        y,
                        fold_cache,
                        interaction_names,
                    )
                )
            else:
                fun = float("nan")

            log = {
                "start_index": int(start_idx),
                "success": bool(result.success),
                "iterations": int(getattr(result, "nit", -1)),
                "objective": float(fun),
                "utility": -fun if np.isfinite(fun) else float("nan"),
                "message": str(result.message),
                "valid": bool(valid),
            }
            start_logs.append(log)
            if valid and np.isfinite(fun):
                results.append((x, float(-fun), int(start_idx), result))
        except Exception as exc:
            start_logs.append({
                "start_index": int(start_idx),
                "success": False,
                "iterations": -1,
                "objective": float("nan"),
                "utility": float("nan"),
                "message": f"exception: {exc}",
                "valid": False,
            })

    if not results:
        best_w = uniform.copy()
        best_utility = uniform_utility
        best_start_idx = 0
        best_result = None
    else:
        # First maximize objective. Among numerically indistinguishable solutions,
        # choose the one closest to the reliability prior.
        best_utility = max(u for _, u, _, _ in results)
        tied = [r for r in results if (best_utility - r[1]) <= OPTIMIZER_TIE_TOLERANCE]
        best_w, _, best_start_idx, best_result = min(
            tied,
            key=lambda item: float(np.linalg.norm(item[0] - prior))
        )

    elapsed = time.time() - start_time
    best_objective = -float(best_utility)
    prior_distance = float(np.linalg.norm(best_w - prior))
    improvement = float(best_utility - uniform_utility)

    return best_w, {
        "success": bool(best_result.success) if best_result is not None else False,
        "iterations": int(getattr(best_result, "nit", -1)) if best_result is not None else -1,
        "message": str(getattr(best_result, "message", "fallback to uniform")) if best_result is not None else "all starts failed",
        "elapsed_seconds": float(elapsed),
        "objective": best_objective,
        "optimization_utility": ROBUST_OPTIMIZATION_UTILITY,
        "n_starts": int(len(starts)),
        "best_start": int(best_start_idx),
        "uniform_robust_utility": float(uniform_utility),
        "best_robust_utility": float(best_utility),
        "improvement_over_uniform": improvement,
        "prior_distance": prior_distance,
        "start_logs": start_logs,
    }


# ============================================================
# Evaluation helpers
# ============================================================

def metrics(y_true, pred):
    """Return classification metrics that remain informative under imbalance."""
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }


def eval_precomputed(K_train, K_test_train, y_train, y_test):
    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(K_train, y_train)
    pred = clf.predict(K_test_train)
    result = metrics(y_test, pred)

    try:
        decision = clf.decision_function(K_test_train)
        if len(np.unique(y_test)) == 2:
            result["roc_auc"] = float(roc_auc_score(y_test, decision))
        else:
            result["roc_auc"] = float("nan")
    except Exception:
        result["roc_auc"] = float("nan")

    return result


def eval_uniform(kernel_train, kernel_test, y_train, y_test):
    ids = list(kernel_train.keys())
    train, _ = normalized_coalition_train(kernel_train, ids)
    raw_test = np.zeros_like(next(iter(kernel_test.values())))
    train_raw = np.zeros_like(next(iter(kernel_train.values())))
    for kid in ids:
        train_raw += kernel_train[kid]
        raw_test += kernel_test[kid]
    scale = float(np.linalg.norm(train_raw, "fro"))
    test = raw_test / scale if scale > 1e-12 else raw_test
    return eval_precomputed(train, test, y_train, y_test)


def eval_individuals(kernel_train, kernel_test, y_train, y_test):
    result = {}
    for kid in kernel_train:
        Ktr, scale = frob_norm(kernel_train[kid])
        Kte = kernel_test[kid] / scale if scale > 1e-12 else kernel_test[kid]
        result[kid] = eval_precomputed(Ktr, Kte, y_train, y_test)
    return result


# ============================================================
# Main
# ============================================================
# ============================================================
# V5.2 configuration -- repeated outer validation + control study
# ============================================================

V5_VERSION = "V5.2"
V5_REVISION_TAG = "repeated_outer_10x5_metrics_synthetic_r2"
V5_INNER_FOLDS = 3
V5_INNER_REPEATS = 1
V5_INNER_VQC_MAXITER = 80
V5_FUSION_RANDOM_STARTS = 12
V5_ROBUSTNESS_LAMBDA = ROBUSTNESS_LAMBDA
V5_NORMALIZATION_EPS = 1e-12

# Only pure D1/D2/D3 choices are considered in V5.2, preserving the V5.1 controls.
# This removes the alpha-mixture grid so the control study can distinguish
# degree selection from arbitrary convex mixing of polynomialized kernels.
V51_DEGREES = (1, 2, 3)
V5_MIN_POSITIVE_DEV_PROFILES = 0.75
V5_MIN_POSITIVE_DEV_FOLDS = 0.60
V5_DEGREE_GAIN_THRESHOLD = 0.0
V5_DEGREE_RELIABILITY_THRESHOLD = 0.0

def train_vqc_theta_v5(X_train, y_train, spec, seed, maxiter=None):
    maxiter = int(maxiter or V5_INNER_VQC_MAXITER)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-0.15, 0.15, size=parameter_count(spec))
    Y = label_kernel(y_train)
    evaluations = 0
    best_theta = theta.copy()
    best_value = -np.inf
    
    def objective(current_theta):
        nonlocal evaluations
        evaluations += 1
        K = statevector_kernel_from_features(X_train, current_theta, spec)
        return centered_alignment(K, Y)

    t0 = time.time()
    a0, c0 = 0.08, 0.12
    A = max(10.0, maxiter / 10.0)
    alpha, gamma = 0.602, 0.101
    for k in range(maxiter):
        delta = rng.choice([-1.0, 1.0], size=theta.size)
        ak = a0 / ((k + 1 + A) ** alpha)
        ck = c0 / ((k + 1) ** gamma)
        y_plus = objective(np.clip(theta + ck * delta, -np.pi, np.pi))
        y_minus = objective(np.clip(theta - ck * delta, -np.pi, np.pi))
        ghat = ((y_plus - y_minus) / (2.0 * ck)) * delta
        theta = np.clip(theta + ak * ghat, -np.pi, np.pi)
        cur = objective(theta)
        if cur > best_value:
            best_value = cur
            best_theta = theta.copy()
    return np.asarray(best_theta, dtype=float), {
        "completed": True,
        "iterations": int(maxiter),
        "function_evaluations": int(evaluations),
        "elapsed_seconds": float(time.time() - t0),
        "kta": float(best_value),
        "optimizer": "SPSA",
    }


def _fit_fold_preprocessor_v5(X_fit_raw):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca = PCA(n_components=N_QUBITS, random_state=RANDOM_SEED)
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X1 = imputer.fit_transform(X_fit_raw)
    X2 = scaler.fit_transform(X1)
    X3 = pca.fit_transform(X2)
    X4 = angle_scaler.fit_transform(X3)
    return (imputer, scaler, pca, angle_scaler), X4


def build_inner_fold_v5(X_train_raw, y_train, backend):
    fold_specs, fold_logs = [], []
    fold_counter = 0
    for repeat in range(V5_INNER_REPEATS):
        folds = make_folds(y_train, V5_INNER_FOLDS, ACTIVE_RUN_SEED + 200 * repeat)
        for fold_no, (train_idx, val_idx) in enumerate(folds):
            fold_id = f"R{repeat}_F{fold_no}"
            X_fit_raw = np.asarray(X_train_raw[train_idx])
            X_val_raw = np.asarray(X_train_raw[val_idx])
            preproc, X_fit_q = _fit_fold_preprocessor_v5(X_fit_raw)
            X_val_q = transform_quantum_features(X_val_raw, preproc)
            pca_var = float(np.sum(preproc[2].explained_variance_ratio_))
            print(f"    {fold_id}: train={len(train_idx)} val={len(val_idx)} fold_PCA_variance={pca_var:.6f}")
            thetas, logs, compiled = {}, {}, {}
            for kidx, kid in enumerate(ACTIVE_KERNEL_IDS):
                theta, log = train_vqc_theta_v5(
                    X_fit_q, y_train[train_idx], KERNEL_SPECS[kid],
                    ACTIVE_RUN_SEED + 3000 + 97 * fold_counter + kidx,
                )
                template, x_params, _ = compile_kernel_template(backend, theta, KERNEL_SPECS[kid])
                thetas[kid] = theta
                logs[kid] = log
                compiled[kid] = (template, x_params)
                print(f"      {kid}: KTA={log['kta']:.6f} evals={log['function_evaluations']} time={log['elapsed_seconds']:.2f}s")
            fold_specs.append({
                "fold_id": fold_id, "repeat": int(repeat), "fold": int(fold_no),
                "train_idx": np.asarray(train_idx, int), "val_idx": np.asarray(val_idx, int),
                "preprocessor": preproc, "X_train_q": np.asarray(X_fit_q, float),
                "X_val_q": np.asarray(X_val_q, float), "pca_explained_variance": pca_var,
                "thetas": thetas, "compiled": compiled,
            })
            fold_logs.append({
                "fold_id": fold_id, "repeat": int(repeat), "fold": int(fold_no),
                "train_size": int(len(train_idx)), "val_size": int(len(val_idx)),
                "pca_explained_variance": pca_var, "training_logs": logs,
            })
            fold_counter += 1
    return fold_specs, fold_logs


def build_inner_profile_cache_v5(fold_specs, profile):
    out = []
    for fs in fold_specs:
        train_idx, val_idx = fs["train_idx"], fs["val_idx"]
        compiled = fs["compiled"]
        Xtrq, Xteq = fs["X_train_q"], fs["X_val_q"]
        if profile.profile_id == "IDEAL":
            tr_states = statevectors_from_templates(Xtrq, compiled)
            va_states = statevectors_from_templates(Xteq, compiled)
            train_bank = {kid: statevector_kernel(tr_states[kid]) for kid in ACTIVE_KERNEL_IDS}
            val_bank = {kid: statevector_kernel(va_states[kid], tr_states[kid]) for kid in ACTIVE_KERNEL_IDS}
        else:
            tr_states = density_matrices_for_profile(Xtrq, compiled, profile)
            va_states = density_matrices_for_profile(Xteq, compiled, profile)
            train_bank, val_bank = {}, {}
            for kid in ACTIVE_KERNEL_IDS:
                raw_tr = density_kernel(tr_states[kid])
                raw_va = density_kernel(va_states[kid], tr_states[kid])
                projected, _, _, _ = psd_project_kernel(raw_tr)
                train_bank[kid] = projected
                val_bank[kid] = raw_va
        out.append({
            "fold_id": fs["fold_id"], "repeat": fs["repeat"], "fold": fs["fold"],
            "train_idx": train_idx, "val_idx": val_idx,
            "train_bank": train_bank, "val_bank": val_bank,
            "pca_explained_variance": fs["pca_explained_variance"],
        })
    return out


def evaluate_fold_matrices_v5(Ktr, Kva, y, train_idx, val_idx):
    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(Ktr, y[train_idx])
    pred = clf.predict(Kva)
    f1 = float(f1_score(y[val_idx], pred, zero_division=0))
    try:
        decision = np.asarray(clf.decision_function(Kva), dtype=float).reshape(-1)
        auc = float(roc_auc_score(y[val_idx], decision)) if len(np.unique(y[val_idx])) == 2 else float("nan")
    except Exception:
        auc = float("nan")
    utility = auc if np.isfinite(auc) else f1
    return f1, auc, utility


def normalize_pair_v5(Ktr_raw, Kva_raw):
    Ktr_raw = np.asarray(Ktr_raw, dtype=float)
    Kva_raw = np.asarray(Kva_raw, dtype=float)
    scale = float(np.linalg.norm(Ktr_raw, "fro"))
    if scale <= V5_NORMALIZATION_EPS:
        return Ktr_raw.copy(), Kva_raw.copy(), scale
    return Ktr_raw / scale, Kva_raw / scale, scale


def make_base_components_v5(profile_folds):
    rows = []
    for f in profile_folds:
        train_components, val_components = {}, {}
        scales = {}
        for kid in ACTIVE_KERNEL_IDS:
            Kn, s = frob_norm(f["train_bank"][kid])
            train_components[kid] = Kn
            val_components[kid] = f["val_bank"][kid] / s if s > V5_NORMALIZATION_EPS else f["val_bank"][kid].copy()
            scales[kid] = s
        rows.append({
            "fold_id": f["fold_id"], "repeat": f["repeat"], "fold": f["fold"],
            "train_idx": f["train_idx"], "val_idx": f["val_idx"],
            "train_components": train_components, "val_components": val_components,
            "singleton_scales": scales,
        })
    return rows


def base_fusion_v5(base_weights, train_components, val_components):
    Ktr_raw = weighted_sum(base_weights, ACTIVE_KERNEL_IDS, train_components)
    Kva_raw = weighted_sum(base_weights, ACTIVE_KERNEL_IDS, val_components)
    return normalize_pair_v5(Ktr_raw, Kva_raw)


def polynomial_components_v5(base_weights, train_components, val_components):
    """Construct degree-1, degree-2 and degree-3 Hadamard powers from the same base fusion."""
    D1tr, D1va, s1 = base_fusion_v5(base_weights, train_components, val_components)
    D2tr, D2va, s2 = normalize_pair_v5(D1tr * D1tr, D1va * D1va)
    D3tr, D3va, s3 = normalize_pair_v5(D1tr * D1tr * D1tr, D1va * D1va * D1va)
    return {
        "degree1": (D1tr, D1va),
        "degree2": (D2tr, D2va),
        "degree3": (D3tr, D3va),
        "scales": {"degree1": s1, "degree2": s2, "degree3": s3},
    }


def polynomial_mix_v5(poly, alpha):
    a1, a2, a3 = [float(x) for x in alpha]
    Ktr_raw = a1 * poly["degree1"][0] + a2 * poly["degree2"][0] + a3 * poly["degree3"][0]
    Kva_raw = a1 * poly["degree1"][1] + a2 * poly["degree2"][1] + a3 * poly["degree3"][1]
    return normalize_pair_v5(Ktr_raw, Kva_raw)


def robust_base_objective_v5(weights, profile_components, y):
    w = np.asarray(weights, dtype=float)
    if w.size != len(ACTIVE_KERNEL_IDS) or not np.all(np.isfinite(w)) or np.min(w) < -1e-8:
        return 1e6
    if abs(float(np.sum(w)) - 1.0) > 1e-6:
        return 1e6
    utilities = []
    for rows in profile_components.values():
        for row in rows:
            Ktr, Kva, _ = base_fusion_v5(w, row["train_components"], row["val_components"])
            _, _, u = evaluate_fold_matrices_v5(Ktr, Kva, y, row["train_idx"], row["val_idx"])
            utilities.append(u)
    arr = np.asarray([u for u in utilities if np.isfinite(u)], float)
    if arr.size == 0:
        return 1e6
    score = float(np.mean(arr))
    if arr.size > 1:
        score -= V5_ROBUSTNESS_LAMBDA * float(np.std(arr, ddof=1))
    return -score


def optimize_base_weights_v5(profile_components, y, seed, reliability_prior=None):
    """Multi-start robust base fusion with reliability prior used for initialization/ties."""
    rng = np.random.default_rng(seed)
    uniform = np.full(len(ACTIVE_KERNEL_IDS), 1.0 / len(ACTIVE_KERNEL_IDS), dtype=float)

    if reliability_prior is not None:
        prior = np.asarray(reliability_prior, dtype=float).reshape(-1)
        if prior.size != len(ACTIVE_KERNEL_IDS) or not np.all(np.isfinite(prior)) or np.sum(prior) <= 1e-12:
            prior = uniform.copy()
        else:
            prior = np.clip(prior, 0.0, None)
            prior /= np.sum(prior)
    else:
        prior = uniform.copy()

    starts = [uniform.copy(), prior.copy()]
    for i in range(len(ACTIVE_KERNEL_IDS)):
        v = np.zeros(len(ACTIVE_KERNEL_IDS), dtype=float)
        v[i] = 1.0
        starts.append(v)
    for _ in range(max(0, V5_FUSION_RANDOM_STARTS - len(starts))):
        starts.append(rng.dirichlet(np.full(len(ACTIVE_KERNEL_IDS), 0.70)))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * len(ACTIVE_KERNEL_IDS)
    best = None
    logs = []
    t0 = time.time()
    uniform_u = -robust_base_objective_v5(uniform, profile_components, y)

    for idx, x0 in enumerate(starts):
        try:
            res = minimize(
                robust_base_objective_v5,
                x0,
                args=(profile_components, y),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": FUSION_MAXITER, "ftol": FUSION_FTOL, "disp": False},
            )
            w = np.clip(np.asarray(res.x, dtype=float), 0.0, None)
            if np.sum(w) <= 1e-12:
                continue
            w /= np.sum(w)
            u = -robust_base_objective_v5(w, profile_components, y)
            logs.append({
                "start_index": int(idx),
                "success": bool(res.success),
                "iterations": int(getattr(res, "nit", -1)),
                "utility": float(u),
            })
            cand = (u, w, idx, res)
            if (
                best is None
                or u > best[0] + 1e-10
                or (abs(u - best[0]) <= 1e-10 and np.linalg.norm(w - prior) < np.linalg.norm(best[1] - prior))
            ):
                best = cand
        except Exception as exc:
            logs.append({
                "start_index": int(idx),
                "success": False,
                "iterations": -1,
                "utility": float("nan"),
                "message": str(exc),
            })

    if best is None:
        bw, bu, bidx, res = uniform.copy(), uniform_u, 0, None
    else:
        bu, bw, bidx, res = best

    return bw, {
        "success": bool(res.success) if res is not None else False,
        "iterations": int(getattr(res, "nit", -1)) if res is not None else -1,
        "n_starts": int(len(starts)),
        "best_start": int(bidx),
        "uniform_robust_utility": float(uniform_u),
        "best_robust_utility": float(bu),
        "improvement_over_uniform": float(bu - uniform_u),
        "prior_distance": float(np.linalg.norm(bw - prior)),
        "elapsed_seconds": float(time.time() - t0),
        "start_logs": logs,
    }


def reliability_prior_v5(summaries):
    scores = [reliability_score(summaries[k]) for k in ACTIVE_KERNEL_IDS]
    w = softmax(scores, RELIABILITY_SOFTMAX_TEMPERATURE)
    return {k: float(v) for k, v in zip(ACTIVE_KERNEL_IDS, w)}


def summarize_contributions_v5(inner_profile_cache, noisy_ids, y):
    pooled = {kid: [] for kid in ACTIVE_KERNEL_IDS}
    by_profile = {}
    synergy_by_profile = {}
    pair_names = [f"{a}x{b}" for i,a in enumerate(ACTIVE_KERNEL_IDS) for b in ACTIVE_KERNEL_IDS[i+1:]]
    diagnostics = []
    for pid in noisy_ids:
        c, s, d = attribution_from_profile_folds_v5(inner_profile_cache[pid], y, pid)
        by_profile[pid] = c
        synergy_by_profile[pid] = s
        diagnostics.extend(d)
        for kid in ACTIVE_KERNEL_IDS:
            pooled[kid].extend(c[kid])
    return pooled, by_profile, synergy_by_profile, pair_names, diagnostics


def attribution_from_profile_folds_v5(profile_folds, y, profile_id):
    ids = list(ACTIVE_KERNEL_IDS)
    all_set = frozenset(ids)
    pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i+1, len(ids))]
    contributions = {kid: [] for kid in ids}
    synergies = {f"{a}x{b}": [] for a,b in pairs}
    diagnostics = []
    for item in profile_folds:
        tb, vb = item["train_bank"], item["val_bank"]
        train_idx, val_idx = item["train_idx"], item["val_idx"]
        coalitions = {all_set}
        for kid in ids:
            coalitions.add(frozenset([kid]))
            coalitions.add(frozenset(set(ids)-{kid}))
        for a,b in pairs:
            coalitions.add(frozenset([a,b]))
        cache = {}
        utilities = []
        for coalition in coalitions:
            cids = sorted(coalition)
            Ktr_raw = np.zeros_like(tb[cids[0]], dtype=float)
            Kva_raw = np.zeros_like(vb[cids[0]], dtype=float)
            for kid in cids:
                Ktr_raw += tb[kid]
                Kva_raw += vb[kid]
            Ktr, Kva, _ = normalize_pair_v5(Ktr_raw, Kva_raw)
            f1, auc, u = evaluate_fold_matrices_v5(Ktr, Kva, y, train_idx, val_idx)
            cache[coalition] = float(u)
            utilities.append(float(u))
        empty_u = fold_majority_utility(y, train_idx, val_idx)
        for kid in ids:
            contributions[kid].append(cache[all_set] - cache[frozenset(set(ids)-{kid})])
        for a,b in pairs:
            key=f"{a}x{b}"
            synergies[key].append(cache[frozenset([a,b])] - cache[frozenset([a])] - cache[frozenset([b])] + empty_u)
        diagnostics.append({
            "profile_id": profile_id, "repeat": int(item["repeat"]), "fold": int(item["fold"]),
            "n_unique_utility_values": int(len(set(np.round(utilities,12)))),
        })
    return contributions, synergies, diagnostics


def build_profile_base_components_v5(inner_profile_cache, profile_ids):
    return {pid: make_base_components_v5(inner_profile_cache[pid]) for pid in profile_ids}


def nested_base_models_v5(train_profile_ids, inner_profile_cache, attr_by_profile, y, seed_base):
    models = {}
    for didx, dev_pid in enumerate(train_profile_ids):
        fit_ids = [p for p in train_profile_ids if p != dev_pid]
        if len(fit_ids) < 2:
            continue
        summaries = {kid: summarize([v for p in fit_ids for v in attr_by_profile[p][kid]]) for kid in ACTIVE_KERNEL_IDS}
        prior_dict = reliability_prior_v5(summaries)
        fit_components = build_profile_base_components_v5(inner_profile_cache, fit_ids)
        prior = np.asarray([prior_dict[k] for k in ACTIVE_KERNEL_IDS], float)
        # Optimize with the prior as an additional deterministic start by appending it manually.
        weights, opt = optimize_base_weights_v5(fit_components, y, seed_base + didx, reliability_prior=prior)
        models[dev_pid] = {"fit_profiles": fit_ids, "base_weights": weights, "optimizer": opt}
    return models


def evaluate_degree_on_profile_v51(profile_rows, y, degree):
    """Evaluate one pure D1/D2/D3 choice and return gains versus D1."""
    if degree not in V51_DEGREES:
        raise ValueError(f"Unsupported degree={degree}; expected one of {V51_DEGREES}.")
    gains, utils, f1s = [], [], []
    name = f"degree{degree}"
    for row in profile_rows:
        poly = polynomial_components_v5(
            row["base_weights"],
            row["train_components"],
            row["val_components"],
        )
        D1tr, D1va = poly["degree1"]
        _, _, D1u = evaluate_fold_matrices_v5(D1tr, D1va, y, row["train_idx"], row["val_idx"])
        Ktr, Kva = poly[name]
        f1, auc, u = evaluate_fold_matrices_v5(Ktr, Kva, y, row["train_idx"], row["val_idx"])
        gains.append(float(u - D1u))
        utils.append(float(u))
        f1s.append(float(f1))
    return np.asarray(gains, float), np.asarray(utils, float), np.asarray(f1s, float)


def select_degree_v51(heldout_pid, train_profile_ids, inner_profile_cache, attr_by_profile, y, dev_base_models):
    """Nested cross-calibration selection among pure D1/D2/D3 with D1 fallback."""
    candidate_rows = []
    dev_model_rows = []
    rows_by_profile = {pid: make_base_components_v5(inner_profile_cache[pid]) for pid in train_profile_ids}

    for dev_pid in train_profile_ids:
        model = dev_base_models.get(dev_pid)
        if model is None:
            continue
        fit_ids = model["fit_profiles"]
        rel_summaries = {
            kid: summarize([v for pp in fit_ids for v in attr_by_profile[pp][kid]])
            for kid in ACTIVE_KERNEL_IDS
        }
        dev_model_rows.append({
            "heldout_profile": heldout_pid,
            "development_profile": dev_pid,
            "fit_profiles": ",".join(fit_ids),
            "base_weights": json.dumps({k: float(v) for k, v in zip(ACTIVE_KERNEL_IDS, model["base_weights"])}),
            "base_robust_utility": float(model["optimizer"]["best_robust_utility"]),
            "reliability_prior": json.dumps(reliability_prior_v5(rel_summaries)),
        })

    for degree in V51_DEGREES:
        all_gains = []
        profile_means = []
        fold_count = 0
        positive_folds = 0
        for dev_pid in train_profile_ids:
            model = dev_base_models.get(dev_pid)
            if model is None:
                continue
            rows = [{**row, "base_weights": model["base_weights"]} for row in rows_by_profile[dev_pid]]
            gains, _, _ = evaluate_degree_on_profile_v51(rows, y, degree)
            if gains.size:
                profile_means.append(float(np.mean(gains)))
                all_gains.extend(gains.tolist())
                fold_count += int(gains.size)
                positive_folds += int(np.sum(gains > 0))

        ss = summarize(all_gains)
        robust = float(ss["mean"]) if ss["n"] else float("nan")
        if ss["n"] > 1 and np.isfinite(ss["std"]):
            robust -= V5_ROBUSTNESS_LAMBDA * float(ss["std"])
        pfrac = float(np.mean(np.asarray(profile_means) > 0)) if profile_means else 0.0
        ffrac = float(positive_folds / fold_count) if fold_count else 0.0
        reliability = reliability_score(ss) if ss["n"] else 0.0
        eligible = True if degree == 1 else bool(
            ss["n"]
            and ss["mean"] > V5_DEGREE_GAIN_THRESHOLD
            and reliability > V5_DEGREE_RELIABILITY_THRESHOLD
            and pfrac >= V5_MIN_POSITIVE_DEV_PROFILES
            and ffrac >= V5_MIN_POSITIVE_DEV_FOLDS
        )
        candidate_rows.append({
            "heldout_profile": heldout_pid,
            "candidate_index": int(degree - 1),
            "degree": int(degree),
            "mean_gain_vs_D1": ss["mean"],
            "std_gain_vs_D1": ss["std"],
            "q10_gain_vs_D1": ss["q10"],
            "q25_gain_vs_D1": ss["q25"],
            "q50_gain_vs_D1": ss["q50"],
            "q75_gain_vs_D1": ss["q75"],
            "q90_gain_vs_D1": ss["q90"],
            "robust_gain_vs_D1": robust,
            "positive_development_profile_fraction": pfrac,
            "positive_development_fold_fraction": ffrac,
            "reliability_score": reliability,
            "eligible": int(eligible),
            "n_gain_observations": int(ss["n"]),
        })

    eligible_nonbase = [r for r in candidate_rows if r["degree"] > 1 and r["eligible"]]
    if eligible_nonbase:
        chosen = max(
            eligible_nonbase,
            key=lambda r: (
                r["robust_gain_vs_D1"],
                r["mean_gain_vs_D1"],
                r["positive_development_profile_fraction"],
                r["degree"],
            ),
        )
    else:
        chosen = next(r for r in candidate_rows if r["degree"] == 1)
        chosen = dict(chosen)
        chosen["eligible"] = 1
    return chosen, candidate_rows, dev_model_rows


def make_outer_degree_components_v5(base_weights, train_bank, test_bank):    train_components, test_components = {}, {}
    for kid in ACTIVE_KERNEL_IDS:
        Ktr, s = frob_norm(train_bank[kid])
        train_components[kid] = Ktr
        test_components[kid] = test_bank[kid] / s if s > V5_NORMALIZATION_EPS else test_bank[kid].copy()
    return polynomial_components_v5(base_weights, train_components, test_components)


def evaluate_outer_degrees_v51(poly, y_train, y_test):
    result = {}
    for degree in V51_DEGREES:
        result[f"degree{degree}"] = eval_precomputed(
            poly[f"degree{degree}"][0],
            poly[f"degree{degree}"][1],
            y_train,
            y_test,
        )
    return result


def single_kernel_polynomial_components_v51(train_kernel, test_kernel):
    """Build D1/D2/D3 for one VQK using training-side normalization."""
    D1tr, D1te, s1 = normalize_pair_v5(train_kernel, test_kernel)
    D2tr, D2te, s2 = normalize_pair_v5(D1tr * D1tr, D1te * D1te)
    D3tr, D3te, s3 = normalize_pair_v5(D1tr * D1tr * D1tr, D1te * D1te * D1te)
    return {
        "degree1": (D1tr, D1te),
        "degree2": (D2tr, D2te),
        "degree3": (D3tr, D3te),
        "scales": {"degree1": s1, "degree2": s2, "degree3": s3},
    }


def evaluate_single_kernel_controls_v51(train_bank, test_bank, y_train, y_test):
    rows = []
    for kid in ACTIVE_KERNEL_IDS:
        Ktr_raw = np.asarray(train_bank[kid], dtype=float)
        Kte_raw = np.asarray(test_bank[kid], dtype=float)
        poly = single_kernel_polynomial_components_v51(Ktr_raw, Kte_raw)
        for degree in V51_DEGREES:
            Ktr, Kte = poly[f"degree{degree}"]
            mt = eval_precomputed(Ktr, Kte, y_train, y_test)
            rows.append({
                "kernel_id": kid,
                "degree": degree,
                "accuracy": mt["accuracy"],
                "balanced_accuracy": mt["balanced_accuracy"],
                "precision": mt["precision"],
                "recall": mt["recall"],
                "f1": mt["f1"],
                "mcc": mt["mcc"],
                "roc_auc": mt["roc_auc"],
            })
    return rows


def uniform_fusion_components_v51(train_bank, test_bank):
    train_components, test_components = {}, {}
    for kid in ACTIVE_KERNEL_IDS:
        Ktr, s = frob_norm(train_bank[kid])
        train_components[kid] = Ktr
        test_components[kid] = test_bank[kid] / s if s > V5_NORMALIZATION_EPS else test_bank[kid].copy()
    w = np.full(len(ACTIVE_KERNEL_IDS), 1.0 / len(ACTIVE_KERNEL_IDS), dtype=float)
    return polynomial_components_v5(w, train_components, test_components)


def evaluate_uniform_degree_controls_v51(train_bank, test_bank, y_train, y_test):
    poly = uniform_fusion_components_v51(train_bank, test_bank)
    rows = []
    for degree in V51_DEGREES:
        Ktr, Kte = poly[f"degree{degree}"]
        mt = eval_precomputed(Ktr, Kte, y_train, y_test)
        rows.append({
            "fusion": "uniform",
            "degree": degree,
            "accuracy": mt["accuracy"],
            "balanced_accuracy": mt["balanced_accuracy"],
            "precision": mt["precision"],
            "recall": mt["recall"],
            "f1": mt["f1"],
            "mcc": mt["mcc"],
            "roc_auc": mt["roc_auc"],
        })
    return rows


def evaluate_adaptive_degree_controls_v51(
    base_weights,
    train_bank,
    test_bank,
    y_train,
    y_test,
    selected_degree,
):
    poly = make_outer_degree_components_v5(
        base_weights,
        train_bank,
        test_bank,
    )
    rows = []
    for degree in V51_DEGREES:
        Ktr, Kte = poly[f"degree{degree}"]
        mt = eval_precomputed(Ktr, Kte, y_train, y_test)
        rows.append({
            "fusion": "adaptive",
            "degree": degree,
            "selected_degree": int(selected_degree),
            "accuracy": mt["accuracy"],
            "balanced_accuracy": mt["balanced_accuracy"],
            "precision": mt["precision"],
            "recall": mt["recall"],
            "f1": mt["f1"],
            "mcc": mt["mcc"],
            "roc_auc": mt["roc_auc"],
        })
    return rows, poly


def summarize_loco_v51(results):
    if not results:
        return
    print("\n" + "=" * 90)
    print("CALIBRATION-CONDITION AVERAGES (5 profiles; same outer test set)")
    print("These are descriptive averages across calibration conditions, not independent test replicates.")
    print("=" * 90)
    for key, label in [
        ("adaptive_degree1", "Adaptive base, D1"),
        ("adaptive_degree2", "Adaptive base, D2"),
        ("adaptive_degree3", "Adaptive base, D3"),
        ("adaptive_selected", "Adaptive selected degree"),
        ("uniform_degree1", "Uniform fusion, D1"),
        ("uniform_degree2", "Uniform fusion, D2"),
        ("uniform_degree3", "Uniform fusion, D3"),
    ]:
        auc = np.asarray([r[key + "_roc_auc"] for r in results], float)
        f1 = np.asarray([r[key + "_f1"] for r in results], float)
        bacc = np.asarray([r[key + "_balanced_accuracy"] for r in results], float)
        mcc = np.asarray([r[key + "_mcc"] for r in results], float)
        print(
            f"{label:28s}: "
            f"mean AUC={np.mean(auc):.4f} | "
            f"F1={np.mean(f1):.4f} | "
            f"BalancedAcc={np.mean(bacc):.4f} | "
            f"MCC={np.mean(mcc):.4f}"
        )
    gain = np.asarray([r["adaptive_selected_minus_adaptive_d1_auc"] for r in results], float)
    print(f"Adaptive selected - adaptive D1 AUC gain: {np.mean(gain):+.6f}")


def evaluate_classical_baselines_v51(X_train_raw, X_test_raw, y_train, y_test):
    Xtr, Xte = classical_preprocess(X_train_raw, X_test_raw)
    specs = [
        ("RBF", {"kernel": "rbf", "C": 1.0, "gamma": "scale"}),
        ("Poly2", {"kernel": "poly", "degree": 2, "C": 1.0, "gamma": "scale", "coef0": 1.0}),
        ("Poly3", {"kernel": "poly", "degree": 3, "C": 1.0, "gamma": "scale", "coef0": 1.0}),
    ]
    rows = []
    for name, params in specs:
        clf = SVC(**params)
        clf.fit(Xtr, y_train)
        mt = metrics(y_test, clf.predict(Xte))
        try:
            decision = np.asarray(clf.decision_function(Xte), dtype=float).reshape(-1)
            mt["roc_auc"] = float(roc_auc_score(y_test, decision))
        except Exception:
            mt["roc_auc"] = float("nan")
        rows.append({"model": name, **mt})
    return rows


def run_dataset_v51(
    dataset_id,
    dataset_meta,
    X_train_raw,
    X_test_raw,
    y_train,
    y_test,
    backend,
    calibration_profiles,
    run_seed=None,
    output_dir=None,
):
    global OUTPUT_DIR, ACTIVE_RUN_SEED
    ACTIVE_RUN_SEED = int(ACTIVE_RUN_SEED if run_seed is None else run_seed)
    OUTPUT_DIR = Path(output_dir) if output_dir is not None else (OUTPUT_ROOT_DIR / dataset_id)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(ACTIVE_RUN_SEED)
    print("=" * 90)
    print("RANQ-VKIL PAPER 2 -- V5.2 REPEATED OUTER MULTI-DATASET + METRIC/SYNTHETIC CONTROLS")
    print(f"Dataset: {dataset_id} | {dataset_meta['dataset_name']} | outer_seed={ACTIVE_RUN_SEED}")
    print("Validation + controls: single-VQK D1/D2/D3, uniform fusion D1/D2/D3, adaptive fusion D1/D2/D3, nested degree selection, and classical polynomial SVM")
    print("=" * 90)

    print("\n[DATA]")
    print(f"Full dataset: {dataset_meta.get('n_samples_full')} samples | {dataset_meta.get('n_features_full')} raw features")
    if dataset_meta.get("interaction_definition"):
        print(f"Synthetic ground truth: {dataset_meta['interaction_definition']}")
    print(f"Train: {len(X_train_raw)} | Test: {len(X_test_raw)} | Raw features: {X_train_raw.shape[1]}")
    print(
        f"Train class counts: 0={np.sum(y_train == 0)} 1={np.sum(y_train == 1)} | "
        f"Test class counts: 0={np.sum(y_test == 0)} 1={np.sum(y_test == 1)}"
    )
    test_majority = max(
        int(np.sum(y_test == 0)),
        int(np.sum(y_test == 1)),
    ) / float(len(y_test))
    print(f"Test majority-class accuracy baseline: {test_majority:.4f}")
    if dataset_meta.get("synthetic_variant"):
        print(
            f"Synthetic variant={dataset_meta['synthetic_variant']} | "
            f"expected interaction degrees={dataset_meta.get('expected_interaction_degrees')} | "
            f"latent={dataset_meta.get('n_latent_features')} "
            f"redundant={dataset_meta.get('n_redundant_features')} "
            f"nuisance={dataset_meta.get('n_nuisance_features')}"
        )
    outer_preproc, X_train_q = fit_quantum_preprocessor(X_train_raw)
    X_test_q = transform_quantum_features(X_test_raw, outer_preproc)
    print(f"Quantum dimensions: {N_QUBITS}")
    print(f"Outer-train PCA explained variance: {outer_preproc[2].explained_variance_ratio_.sum():.6f}")

    print("\n" + "=" * 90)
    print("OUTER-TRAIN VQC BANK FOR FINAL HELD-OUT EVALUATION -- V5.2")
    print("=" * 90)
    thetas, training_logs = {}, {}
    for idx, kid in enumerate(ACTIVE_KERNEL_IDS):
        theta, log = train_vqc_theta(X_train_q, y_train, KERNEL_SPECS[kid], ACTIVE_RUN_SEED + idx)
        thetas[kid] = theta
        training_logs[kid] = log
        print(f"[{kid}] KTA={log['kta']:.6f} evals={log['function_evaluations']} time={log['elapsed_seconds']:.2f}s")

    print("\n" + "=" * 90)
    print("SHARED IBM CALIBRATION CONTEXT")
    print("=" * 90)
    print(f"Backend: {backend.name}")
    print(f"Reference calibration: {REFERENCE_CALIBRATION_LABEL}")
    print(f"Frozen physical qubits: {FIXED_PHYSICAL_QUBITS}")

    print("\n" + "=" * 90)
    print("FIXED IBM HARDWARE TARGET + HARDWARE-AWARE TRANSPILATION")
    print("=" * 90)
    compiled, transpilation_logs = {}, {}
    for kid in ACTIVE_KERNEL_IDS:
        template, xp, transpiled = compile_kernel_template(backend, thetas[kid], KERNEL_SPECS[kid])
        compiled[kid] = (template, xp)
        transpilation_logs[kid] = {
            "abstract_depth": int(build_vqc(X_train_q[0], thetas[kid], KERNEL_SPECS[kid]).depth()),
            "transpiled_depth": int(transpiled.depth()),
            "compact_depth": int(template.depth()),
            "ops": dict(transpiled.count_ops()),
        }
        z = transpilation_logs[kid]
        print(f"{kid}: abstract={z['abstract_depth']} transpiled={z['transpiled_depth']} compact={z['compact_depth']} ops={z['ops']}")

    ideal_tr_states = statevectors_from_templates(X_train_q, compiled)
    ideal_te_states = statevectors_from_templates(X_test_q, compiled)
    ideal_train = {kid: statevector_kernel(ideal_tr_states[kid]) for kid in ACTIVE_KERNEL_IDS}
    ideal_test = {kid: statevector_kernel(ideal_te_states[kid], ideal_tr_states[kid]) for kid in ACTIVE_KERNEL_IDS}
    train_by_profile = {"IDEAL": ideal_train}
    test_by_profile = {"IDEAL": ideal_test}
    psd_rows, drift_rows = [], []
    noisy_profiles = [p for p in calibration_profiles if p.profile_id != "IDEAL"]
    for p in noisy_profiles:
        tr = density_matrices_for_profile(X_train_q, compiled, p)
        te = density_matrices_for_profile(X_test_q, compiled, p)
        tb, cb = {}, {}
        for kid in ACTIVE_KERNEL_IDS:
            rawtr = density_kernel(tr[kid])
            rawte = density_kernel(te[kid], tr[kid])
            proj, re, pe, pfro = psd_project_kernel(rawtr)
            tb[kid] = proj
            cb[kid] = rawte
            psd_rows.append({"profile_id": p.profile_id, "kernel_id": kid, "raw_min_eig": float(np.min(re)), "psd_min_eig": float(np.min(pe)), "projection_fro": float(pfro)})
            drift_rows.append({"profile_id": p.profile_id, "kernel_id": kid, "relative_frobenius_drift": relative_kernel_drift(proj, ideal_train[kid]), "centered_alignment_to_ideal": kernel_alignment_to_reference(proj, ideal_train[kid])})
        train_by_profile[p.profile_id] = tb
        test_by_profile[p.profile_id] = cb

    pd.DataFrame(psd_rows).to_csv(OUTPUT_DIR / "psd_projection_diagnostics_v5_2.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(OUTPUT_DIR / "kernel_drift_vs_ideal_v5_2.csv", index=False)
    print("\n" + "=" * 90)
    print("KERNEL DRIFT VS IDEAL")
    print("=" * 90)
    for r in drift_rows:
        print(f"{r['profile_id']:14s} {r['kernel_id']}: drift={r['relative_frobenius_drift']:.6f} alignment={r['centered_alignment_to_ideal']:.6f}")

    print("\n" + "=" * 90)
    print("LEAKAGE-FREE INNER-CV VQC TRAINING -- V5.2")
    print("=" * 90)
    print("Each fold independently fits imputer/scaler/PCA/angle scaling on inner-train only; validation data never affect preprocessing or VQC KTA optimization.")
    fold_specs, inner_logs = build_inner_fold_v5(X_train_raw, y_train, backend)
    inner_cache = {}
    for p in noisy_profiles:
        print(f"  inner cache profile: {p.profile_id}")
        inner_cache[p.profile_id] = build_inner_profile_cache_v5(fold_specs, p)
    inner_ideal_cache = build_inner_profile_cache_v5(fold_specs, calibration_profiles[0])

    noisy_ids = [p.profile_id for p in noisy_profiles]
    print("\n" + "=" * 90)
    print("LEAKAGE-FREE DISTRIBUTIONAL CONTRIBUTION + CANONICAL SYNERGY -- V5.2")
    print("=" * 90)
    print("Canonical pairwise synergy remains a structural diagnostic; degree selection is based on nested cross-calibration predictive gain.")
    pooled_attr = {kid: [] for kid in ACTIVE_KERNEL_IDS}
    attr_by_profile, syn_by_profile, attr_diag = {}, {}, []
    pair_names = [f"{a}x{b}" for i, a in enumerate(ACTIVE_KERNEL_IDS) for b in ACTIVE_KERNEL_IDS[i + 1:]]
    for pid in noisy_ids:
        c, s, d = attribution_from_profile_folds_v5(inner_cache[pid], y_train, pid)
        attr_by_profile[pid] = c
        syn_by_profile[pid] = s
        attr_diag.extend(d)
        for kid in ACTIVE_KERNEL_IDS:
            pooled_attr[kid].extend(c[kid])

    contrib_rows = []
    for kid in ACTIVE_KERNEL_IDS:
        ss = summarize(pooled_attr[kid])
        contrib_rows.append({"kernel_id": kid, **ss})
        print(f"{kid:8s} mean={ss['mean']:+.6f} std={ss['std']:.6f} q10={ss['q10']:+.6f} q25={ss['q25']:+.6f} q50={ss['q50']:+.6f} q75={ss['q75']:+.6f} q90={ss['q90']:+.6f} n={ss['n']}")

    syn_rows = []
    for key in pair_names:
        vals = [v for pid in noisy_ids for v in syn_by_profile[pid][key]]
        ss = summarize(vals)
        syn_rows.append({"interaction": key, **ss})
        print(f"{key:16s} synergy_mean={ss['mean']:+.6f} synergy_q10={ss['q10']:+.6f}")

    pd.DataFrame(contrib_rows).to_csv(OUTPUT_DIR / "contribution_summary_v5_2.csv", index=False)
    pd.DataFrame([
        {**{"profile_id": pid, "kernel_id": kid}, **summarize(attr_by_profile[pid][kid])}
        for pid in noisy_ids for kid in ACTIVE_KERNEL_IDS
    ]).to_csv(OUTPUT_DIR / "contribution_by_calibration_profile_v5_2.csv", index=False)
    pd.DataFrame(syn_rows).to_csv(OUTPUT_DIR / "synergy_summary_v5_2.csv", index=False)
    pd.DataFrame([
        {**{"profile_id": pid, "interaction": key}, **summarize(syn_by_profile[pid][key])}
        for pid in noisy_ids for key in pair_names
    ]).to_csv(OUTPUT_DIR / "synergy_by_calibration_profile_v5_2.csv", index=False)
    pd.DataFrame(attr_diag).to_csv(OUTPUT_DIR / "coalition_utility_diagnostics_v5_2.csv", index=False)

    print("\n" + "=" * 90)
    print("LEAVE-ONE-CALIBRATION-OUT ADAPTIVE BASE + PURE DEGREE SELECTION -- V5.2")
    print("=" * 90)
    print("Controls: single VQK D1/D2/D3, uniform fusion D1/D2/D3, adaptive fusion D1/D2/D3, and nested adaptive degree selection.")

    loco_rows, selection_rows, candidate_rows, dev_rows, weight_rows = [], [], [], [], []
    single_rows, uniform_rows, adaptive_rows = [], [], []

    for hidx, heldout_pid in enumerate(noisy_ids):
        train_ids = [p for p in noisy_ids if p != heldout_pid]
        base_summaries = {kid: summarize([v for pid in train_ids for v in attr_by_profile[pid][kid]]) for kid in ACTIVE_KERNEL_IDS}
        base_prior = reliability_prior_v5(base_summaries)
        base_profile_components = build_profile_base_components_v5(inner_cache, train_ids)
        base_weights, base_opt = optimize_base_weights_v5(
            base_profile_components,
            y_train,
            ACTIVE_RUN_SEED + 7000 + hidx,
            reliability_prior=np.asarray([base_prior[k] for k in ACTIVE_KERNEL_IDS], float),
        )
        dev_models = nested_base_models_v5(
            train_ids,
            inner_cache,
            attr_by_profile,
            y_train,
            ACTIVE_RUN_SEED + 10000 + hidx * 100,
        )
        chosen, candidates, dev_model_rows = select_degree_v51(
            heldout_pid,
            train_ids,
            inner_cache,
            attr_by_profile,
            y_train,
            dev_models,
        )
        selected_degree = int(chosen["degree"])

        adaptive_control_rows, held_poly = evaluate_adaptive_degree_controls_v51(
            base_weights,
            train_by_profile[heldout_pid],
            test_by_profile[heldout_pid],
            y_train,
            y_test,
            selected_degree,
        )
        uniform_control_rows = evaluate_uniform_degree_controls_v51(
            train_by_profile[heldout_pid],
            test_by_profile[heldout_pid],
            y_train,
            y_test,
        )
        single_control_rows = evaluate_single_kernel_controls_v51(
            train_by_profile[heldout_pid],
            test_by_profile[heldout_pid],
            y_train,
            y_test,
        )

        selected_mt = next(r for r in adaptive_control_rows if r["degree"] == selected_degree)
        adaptive_d1 = next(r for r in adaptive_control_rows if r["degree"] == 1)
        adaptive_d2 = next(r for r in adaptive_control_rows if r["degree"] == 2)
        adaptive_d3 = next(r for r in adaptive_control_rows if r["degree"] == 3)
        uniform_d1 = next(r for r in uniform_control_rows if r["degree"] == 1)
        uniform_d2 = next(r for r in uniform_control_rows if r["degree"] == 2)
        uniform_d3 = next(r for r in uniform_control_rows if r["degree"] == 3)

        print("\n" + "-" * 90)
        print(f"HELD-OUT CALIBRATION: {heldout_pid}")
        print("-" * 90)
        print(f"Base reliability prior: { {k: round(v,6) for k,v in base_prior.items()} }")
        print(f"Selected pure degree: D{selected_degree}")
        print(f"Selection evidence: mean_gain_vs_D1={chosen['mean_gain_vs_D1']:+.6f} q10={chosen['q10_gain_vs_D1']:+.6f} robust_gain={chosen['robust_gain_vs_D1']:+.6f} positive_profiles={chosen['positive_development_profile_fraction']:.2%} positive_folds={chosen['positive_development_fold_fraction']:.2%} eligible={bool(chosen['eligible'])}")
        print(
            f"Adaptive AUC: D1={adaptive_d1['roc_auc']:.4f} | "
            f"D2={adaptive_d2['roc_auc']:.4f} | "
            f"D3={adaptive_d3['roc_auc']:.4f} | "
            f"selected D{selected_degree}={selected_mt['roc_auc']:.4f}"
        )
        print(
            f"Selected D{selected_degree}: "
            f"F1={selected_mt['f1']:.4f} | "
            f"BalancedAcc={selected_mt['balanced_accuracy']:.4f} | "
            f"MCC={selected_mt['mcc']:.4f} | "
            f"Accuracy={selected_mt['accuracy']:.4f}"
        )
        print(
            f"Uniform AUC:  D1={uniform_d1['roc_auc']:.4f} | "
            f"D2={uniform_d2['roc_auc']:.4f} | "
            f"D3={uniform_d3['roc_auc']:.4f}"
        )
        print(f"Adaptive selected - adaptive D1 AUC={selected_mt['roc_auc'] - adaptive_d1['roc_auc']:+.6f}")
        print(f"Adaptive D1 - uniform D1 AUC={adaptive_d1['roc_auc'] - uniform_d1['roc_auc']:+.6f}")
        print(f"Base optimizer: starts={base_opt['n_starts']} best_start={base_opt['best_start']} uniform={base_opt['uniform_robust_utility']:.6f} best={base_opt['best_robust_utility']:.6f} improvement={base_opt['improvement_over_uniform']:+.6f}")

        loco_rows.append({
            "heldout_profile": heldout_pid,
            "selected_degree": selected_degree,
            "selection_eligible": int(chosen["eligible"]),
            "adaptive_selected_f1": selected_mt["f1"],
            "adaptive_selected_balanced_accuracy": selected_mt["balanced_accuracy"],
            "adaptive_selected_mcc": selected_mt["mcc"],
            "adaptive_selected_accuracy": selected_mt["accuracy"],
            "adaptive_selected_roc_auc": selected_mt["roc_auc"],
            "adaptive_degree1_f1": adaptive_d1["f1"],
            "adaptive_degree1_balanced_accuracy": adaptive_d1["balanced_accuracy"],
            "adaptive_degree1_mcc": adaptive_d1["mcc"],
            "adaptive_degree1_accuracy": adaptive_d1["accuracy"],
            "adaptive_degree1_roc_auc": adaptive_d1["roc_auc"],
            "adaptive_degree2_f1": adaptive_d2["f1"],
            "adaptive_degree2_balanced_accuracy": adaptive_d2["balanced_accuracy"],
            "adaptive_degree2_mcc": adaptive_d2["mcc"],
            "adaptive_degree2_accuracy": adaptive_d2["accuracy"],
            "adaptive_degree2_roc_auc": adaptive_d2["roc_auc"],
            "adaptive_degree3_f1": adaptive_d3["f1"],
            "adaptive_degree3_balanced_accuracy": adaptive_d3["balanced_accuracy"],
            "adaptive_degree3_mcc": adaptive_d3["mcc"],
            "adaptive_degree3_accuracy": adaptive_d3["accuracy"],
            "adaptive_degree3_roc_auc": adaptive_d3["roc_auc"],
            "uniform_degree1_f1": uniform_d1["f1"],
            "uniform_degree1_balanced_accuracy": uniform_d1["balanced_accuracy"],
            "uniform_degree1_mcc": uniform_d1["mcc"],
            "uniform_degree1_accuracy": uniform_d1["accuracy"],
            "uniform_degree1_roc_auc": uniform_d1["roc_auc"],
            "uniform_degree2_f1": uniform_d2["f1"],
            "uniform_degree2_balanced_accuracy": uniform_d2["balanced_accuracy"],
            "uniform_degree2_mcc": uniform_d2["mcc"],
            "uniform_degree2_accuracy": uniform_d2["accuracy"],
            "uniform_degree2_roc_auc": uniform_d2["roc_auc"],
            "uniform_degree3_f1": uniform_d3["f1"],
            "uniform_degree3_balanced_accuracy": uniform_d3["balanced_accuracy"],
            "uniform_degree3_mcc": uniform_d3["mcc"],
            "uniform_degree3_accuracy": uniform_d3["accuracy"],
            "uniform_degree3_roc_auc": uniform_d3["roc_auc"],
            "adaptive_selected_minus_adaptive_d1_auc": selected_mt["roc_auc"] - adaptive_d1["roc_auc"],
            "adaptive_d1_minus_uniform_d1_auc": adaptive_d1["roc_auc"] - uniform_d1["roc_auc"],
            "adaptive_selected_minus_uniform_d1_auc": selected_mt["roc_auc"] - uniform_d1["roc_auc"],
            "selection_mean_gain_vs_D1": chosen["mean_gain_vs_D1"],
            "selection_std_gain_vs_D1": chosen["std_gain_vs_D1"],
            "selection_q10_gain_vs_D1": chosen["q10_gain_vs_D1"],
            "selection_robust_gain_vs_D1": chosen["robust_gain_vs_D1"],
            "positive_development_profile_fraction": chosen["positive_development_profile_fraction"],
            "positive_development_fold_fraction": chosen["positive_development_fold_fraction"],
            "selection_reliability_score": chosen["reliability_score"],
            "base_optimizer_best_utility": base_opt["best_robust_utility"],
        })
        selection_rows.append({
            "heldout_profile": heldout_pid,
            "selected_degree": selected_degree,
            "selection_eligible": int(chosen["eligible"]),
            **{k: chosen[k] for k in [
                "mean_gain_vs_D1", "std_gain_vs_D1", "q10_gain_vs_D1", "q25_gain_vs_D1",
                "q50_gain_vs_D1", "q75_gain_vs_D1", "q90_gain_vs_D1", "robust_gain_vs_D1",
                "positive_development_profile_fraction", "positive_development_fold_fraction",
                "reliability_score", "n_gain_observations"
            ]},
        })
        candidate_rows.extend(candidates)
        dev_rows.extend(dev_model_rows)
        for kid, w in zip(ACTIVE_KERNEL_IDS, base_weights):
            weight_rows.append({"heldout_profile": heldout_pid, "component": kid, "component_type": "base_kernel", "weight": float(w)})
        for row in adaptive_control_rows:
            adaptive_rows.append({"heldout_profile": heldout_pid, **row})
        for row in uniform_control_rows:
            uniform_rows.append({"heldout_profile": heldout_pid, **row})
        for row in single_control_rows:
            single_rows.append({"heldout_profile": heldout_pid, **row})

    summarize_loco_v51(loco_rows)

    print("\n" + "=" * 90)
    print("CALIBRATION-POOLED IDEAL-KERNEL REFERENCE -- V5.2")
    print("=" * 90)
    pooled_summaries = {kid: summarize(pooled_attr[kid]) for kid in ACTIVE_KERNEL_IDS}
    pooled_prior = reliability_prior_v5(pooled_summaries)
    ideal_components = {"IDEAL": make_base_components_v5(inner_ideal_cache)}
    ideal_weights, ideal_opt = optimize_base_weights_v5(
        ideal_components,
        y_train,
        ACTIVE_RUN_SEED + 21000,
        reliability_prior=np.asarray([pooled_prior[k] for k in ACTIVE_KERNEL_IDS], float),
    )
    ideal_poly = make_outer_degree_components_v5(ideal_weights, ideal_train, ideal_test)
    ideal_mt = evaluate_outer_degrees_v51(ideal_poly, y_train, y_test)
    ideal_uniform_poly = uniform_fusion_components_v51(ideal_train, ideal_test)
    ideal_uniform_rows = {}
    for degree in V51_DEGREES:
        Ktr, Kte = ideal_uniform_poly[f"degree{degree}"]
        ideal_uniform_rows[degree] = eval_precomputed(Ktr, Kte, y_train, y_test)
    print(f"Ideal adaptive base weights={dict(zip(ACTIVE_KERNEL_IDS,np.round(ideal_weights,6)))}")
    for degree in V51_DEGREES:
        print(
            f"Ideal adaptive D{degree}: "
            f"AUC={ideal_mt[f'degree{degree}']['roc_auc']:.4f} | "
            f"F1={ideal_mt[f'degree{degree}']['f1']:.4f} | "
            f"BalancedAcc={ideal_mt[f'degree{degree}']['balanced_accuracy']:.4f} | "
            f"MCC={ideal_mt[f'degree{degree}']['mcc']:.4f}"
        )
        print(
            f"Ideal uniform D{degree}:  "
            f"AUC={ideal_uniform_rows[degree]['roc_auc']:.4f} | "
            f"F1={ideal_uniform_rows[degree]['f1']:.4f} | "
            f"BalancedAcc={ideal_uniform_rows[degree]['balanced_accuracy']:.4f} | "
            f"MCC={ideal_uniform_rows[degree]['mcc']:.4f}"
        )

    print("\n" + "=" * 90)
    print("CLASSICAL BASELINES -- RBF + POLYNOMIAL SVM D2/D3")
    print("=" * 90)
    classical_rows = evaluate_classical_baselines_v51(X_train_raw, X_test_raw, y_train, y_test)
    for row in classical_rows:
        print(f"{row['model']:8s}: AUC={row['roc_auc']:.4f} F1={row['f1']:.4f} BalancedAcc={row['balanced_accuracy']:.4f} MCC={row['mcc']:.4f} Accuracy={row['accuracy']:.4f}")

    single_df = pd.DataFrame(single_rows)
    uniform_df = pd.DataFrame(uniform_rows)
    adaptive_df = pd.DataFrame(adaptive_rows)
    classical_df = pd.DataFrame(classical_rows)
    single_summary_df = (single_df.groupby("degree", as_index=False)[["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc"]].mean().sort_values("degree") if not single_df.empty else pd.DataFrame())
    uniform_summary_df = (uniform_df.groupby("degree", as_index=False)[["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc"]].mean().sort_values("degree") if not uniform_df.empty else pd.DataFrame())
    adaptive_summary_df = (adaptive_df.groupby("degree", as_index=False)[["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc"]].mean().sort_values("degree") if not adaptive_df.empty else pd.DataFrame())

    print("\nCONTROL STUDY MEANS (AUC, F1, Balanced Accuracy, MCC, and supporting metrics)")
    if not single_summary_df.empty:
        print("Single-VQK mean by degree:")
        print(single_summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if not uniform_summary_df.empty:
        print("Uniform fusion mean by degree:")
        print(uniform_summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if not adaptive_summary_df.empty:
        print("Adaptive fusion mean by degree:")
        print(adaptive_summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    pd.DataFrame(loco_rows).to_csv(OUTPUT_DIR / "loco_results_v5_2.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(OUTPUT_DIR / "loco_degree_selection_v5_2.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(OUTPUT_DIR / "degree_candidate_consistency_v5_2.csv", index=False)
    pd.DataFrame(dev_rows).to_csv(OUTPUT_DIR / "degree_development_models_v5_2.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(OUTPUT_DIR / "loco_base_weights_v5_2.csv", index=False)
    single_df.to_csv(OUTPUT_DIR / "single_kernel_degree_controls_v5_2.csv", index=False)
    single_summary_df.to_csv(OUTPUT_DIR / "single_kernel_degree_summary_v5_2.csv", index=False)
    uniform_df.to_csv(OUTPUT_DIR / "uniform_fusion_degree_controls_v5_2.csv", index=False)
    uniform_summary_df.to_csv(OUTPUT_DIR / "uniform_fusion_degree_summary_v5_2.csv", index=False)
    adaptive_df.to_csv(OUTPUT_DIR / "adaptive_fusion_degree_controls_v5_2.csv", index=False)
    adaptive_summary_df.to_csv(OUTPUT_DIR / "adaptive_fusion_degree_summary_v5_2.csv", index=False)
    classical_df.to_csv(OUTPUT_DIR / "classical_baselines_v5_2.csv", index=False)

    config = {
        "version": V5_VERSION,
        "revision_tag": V5_REVISION_TAG,
        "outer_seed": int(ACTIVE_RUN_SEED),
        "study": "repeated_outer_evaluation_multi_dataset",
        "repeated_outer_evaluation": {
            "outer_seed": int(ACTIVE_RUN_SEED),
            "target_outer_seeds": list(V52_OUTER_SEEDS),
            "n_calibration_conditions": int(len([p for p in calibration_profiles if p.profile_id != "IDEAL"])),
            "resume_enabled": bool(V52_RESUME),
        },
        "dataset_id": dataset_id,
        "dataset_name": dataset_meta["dataset_name"],
        "dataset_source": dataset_meta.get("source"),
        "dataset_uci_id": dataset_meta.get("uci_id"),
        "dataset_doi": dataset_meta.get("doi"),
        "dataset_full_samples": dataset_meta.get("n_samples_full"),
        "dataset_full_features": dataset_meta.get("n_features_full"),
        "dataset_train_positive": dataset_meta.get("train_positive"),
        "dataset_train_negative": dataset_meta.get("train_negative"),
        "dataset_test_positive": dataset_meta.get("test_positive"),
        "dataset_test_negative": dataset_meta.get("test_negative"),
        "dataset_test_majority_accuracy_baseline": dataset_meta.get("test_majority_accuracy_baseline"),
        "dataset_test_positive_rate": dataset_meta.get("test_positive_rate"),
        "dataset_generation": dataset_meta.get("interaction_definition"),
        "synthetic_variant": dataset_meta.get("synthetic_variant"),
        "synthetic_expected_interaction_degrees": dataset_meta.get("expected_interaction_degrees"),
        "train_samples": int(len(X_train_raw)),
        "test_samples": int(len(X_test_raw)),
        "n_qubits": N_QUBITS,
        "vqc_depth": VQC_DEPTH,
        "active_kernel_ids": ACTIVE_KERNEL_IDS,
        "inner_folds": V5_INNER_FOLDS,
        "inner_repeats": V5_INNER_REPEATS,
        "inner_vqc_maxiter": V5_INNER_VQC_MAXITER,
        "fusion_random_starts": V5_FUSION_RANDOM_STARTS,
        "robustness_lambda": V5_ROBUSTNESS_LAMBDA,
        "candidate_degrees": list(V51_DEGREES),
        "degree_selection": "nested_cross_calibration_predictive_gain_among_pure_D1_D2_D3_with_D1_fallback",
        "controls": {
            "single_vqk": "each VQK evaluated at D1/D2/D3",
            "uniform_fusion": "uniform VQK fusion evaluated at D1/D2/D3",
            "adaptive_fusion": "robustly optimized base weights evaluated at D1/D2/D3",
            "classical": "RBF, polynomial degree 2, polynomial degree 3",
        },
        "metric_protocol": {
            "primary_continuous_metric": "roc_auc",
            "classification_metrics": [
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "mcc",
            ],
            "imbalance_control_note": "Balanced accuracy and MCC are reported because several pilot datasets and outer splits are class-imbalanced; accuracy alone is therefore not sufficient.",
        },
        "polynomial_form": "D1=K_base; D2=K_base^(Hadamard 2); D3=K_base^(Hadamard 3)",
        "normalization": "each degree component uses training-side Frobenius normalization",
        "single_kernel_normalization": "each VQK uses its training kernel before D1/D2/D3 construction",
        "canonical_synergy_role": "structural diagnostic only",
        "ibm_backend": IBM_BACKEND_NAME,
        "reference_calibration": REFERENCE_CALIBRATION_LABEL,
        "fixed_physical_qubits": FIXED_PHYSICAL_QUBITS,
        "same_outer_test_across_calibrations": True,
        "pilot_note": "IBM calibration data are converted to approximate Aer noise channels; this is not physical QPU execution.",
        "interpretation_note": "D2/D3 are Hadamard powers of the fused kernel and therefore represent higher-order kernel lifting/sharpening; they are not direct evidence of explicit pairwise cross-VQK synergy. The clean synthetic variant is intended to isolate latent interaction signal, while the noisy variant tests robustness to redundant and nuisance features.",
        "multi_dataset_note": "Paper 2 evaluates four non-overlapping UCI datasets plus two reproducible synthetic variants (clean and noisy) sharing the same explicit 2-way and 3-way interaction ground truth; the clean variant uses 16 low-noise correlated measurements of the four latent variables, while the noisy variant adds independent nuisance dimensions; the same outer sample cap and calibration-aware pipeline are used for each dataset.",
    }

    bundle = {
        "paper": "RANQ-VKIL Paper 2 Pilot V5.2 Multi-Dataset",
        "version": V5_VERSION,
        "revision_tag": V5_REVISION_TAG,
        "dataset_metadata": dataset_meta,
        "config": config,
        "training_logs": training_logs,
        "inner_training_logs": inner_logs,
        "transpilation_logs": transpilation_logs,
        "profiles": [{"profile_id": p.profile_id, "timestamp": p.timestamp, "physical_qubits": p.physical_qubits, "is_ideal": p.profile_id == "IDEAL"} for p in calibration_profiles],
        "psd_diagnostics": psd_rows,
        "kernel_drift": drift_rows,
        "contribution_summaries": contrib_rows,
        "synergy_summaries": syn_rows,
        "attribution_diagnostics": attr_diag,
        "loco_results": loco_rows,
        "degree_selection": selection_rows,
        "degree_candidates": candidate_rows,
        "degree_development_models": dev_rows,
        "loco_base_weights": weight_rows,
        "single_kernel_degree_controls": single_rows,
        "uniform_fusion_degree_controls": uniform_rows,
        "adaptive_fusion_degree_controls": adaptive_rows,
        "classical_baselines": classical_rows,
        "ideal_reference": {
            "base_weights": dict(zip(ACTIVE_KERNEL_IDS, ideal_weights.tolist())),
            "optimizer": ideal_opt,
            "adaptive_degrees": ideal_mt,
            "uniform_degrees": ideal_uniform_rows,
        },
    }
    with (OUTPUT_DIR / "ranq_vkil_paper2_v5_2_results.json").open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2, default=json_safe)

    print("\n" + "=" * 90)
    print("PILOT V5.2 FINISHED")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("=" * 90)
    return bundle


def prepare_shared_ibm_context():
    """Connect once and construct the fixed calibration/noise context shared by all datasets."""
    print("\n" + "=" * 90)
    print("IBM CONNECTION + SHARED CALIBRATION SERIES")
    print("=" * 90)
    service = connect_ibm()
    backend = service.backend(IBM_BACKEND_NAME)
    print(f"Instance: {IBM_INSTANCE}")
    print(f"Backend: {backend.name}")
    print(f"Qubits: {backend.num_qubits}")
    try:
        print(f"Processor target qubits: {backend.target.num_qubits}")
    except Exception:
        pass

    profiles_raw, current_ts = fetch_calibration_profiles(backend)
    print(f"Current calibration: {iso(current_ts)}")
    reference_props = next(
        (props for pid, props in profiles_raw if pid == REFERENCE_CALIBRATION_LABEL),
        profiles_raw[-1][1],
    )
    validate_fixed_qubits(reference_props, backend)
    print(f"Reference calibration: {REFERENCE_CALIBRATION_LABEL}")
    print(f"Frozen physical qubits: {FIXED_PHYSICAL_QUBITS}")

    calibration_profiles = [
        CalibrationProfile(
            "IDEAL", "ideal", None, None, None, list(FIXED_PHYSICAL_QUBITS)
        )
    ]
    for pid, props in profiles_raw:
        model = build_noise_model(props, FIXED_PHYSICAL_QUBITS)
        sim = AerSimulator(method="density_matrix", noise_model=model)
        calibration_profiles.append(
            CalibrationProfile(
                pid,
                iso(getattr(props, "last_update_date", None)),
                props,
                model,
                sim,
                list(FIXED_PHYSICAL_QUBITS),
            )
        )
        st = profile_statistics(props, FIXED_PHYSICAL_QUBITS)
        print(
            f"  {pid:14s} 1Qerr={fmt_float(st['one_qubit_gate_error_median'])} "
            f"2Qerr={fmt_float(st['two_qubit_gate_error_median'])} "
            f"RO={fmt_float(st['readout_error_median'])} "
            f"T1={fmt_float(st['t1_median_s'])}s "
            f"T2={fmt_float(st['t2_median_s'])}s"
        )
    return backend, calibration_profiles


def _metric_summary(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "se": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "se": float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def aggregate_repeated_outer_v52(all_runs):
    """Aggregate repeated outer evaluations across seeds and datasets.

    The primary estimate is the mean across independent outer seeds for each
    dataset/method/metric. Calibration conditions are repeated measurements on
    the same seed-specific outer test set and are therefore summarized separately
    rather than treated as independent outer replicates.
    """
    tidy_rows = []
    selection_rows = []
    for run in all_runs:
        dataset_id = run["dataset_id"]
        seed = int(run["outer_seed"])
        bundle = run["bundle"]
        single_df = pd.DataFrame(bundle.get("single_kernel_degree_controls", []))
        uniform_df = pd.DataFrame(bundle.get("uniform_fusion_degree_controls", []))
        adaptive_df = pd.DataFrame(bundle.get("adaptive_fusion_degree_controls", []))
        loco_df = pd.DataFrame(bundle.get("loco_results", []))
        classical_df = pd.DataFrame(bundle.get("classical_baselines", []))
        methods = []
        for family, df, prefix in [
            ("single", single_df, "Single-VQK"),
            ("uniform", uniform_df, "Uniform Fusion"),
            ("adaptive", adaptive_df, "Adaptive Fusion"),
        ]:
            if df.empty:
                continue
            for degree in V51_DEGREES:
                sub = df[df["degree"] == degree]
                if sub.empty:
                    continue
                for metric in ["roc_auc", "f1", "balanced_accuracy", "mcc", "accuracy", "precision", "recall"]:
                    vals = sub[metric].to_numpy(dtype=float)
                    methods.append((f"{prefix} D{degree}", metric, float(np.nanmean(vals))))
        if not loco_df.empty:
            for metric in ["roc_auc", "f1", "balanced_accuracy", "mcc", "accuracy", "precision", "recall"]:
                col = f"adaptive_selected_{metric}"
                if col in loco_df.columns:
                    methods.append(("Adaptive Selected Degree", metric, float(np.nanmean(loco_df[col].to_numpy(dtype=float)))) )
        if not classical_df.empty:
            for model in ["RBF", "Poly2", "Poly3"]:
                sub = classical_df[classical_df["model"] == model]
                if sub.empty:
                    continue
                for metric in ["roc_auc", "f1", "balanced_accuracy", "mcc", "accuracy", "precision", "recall"]:
                    if metric in sub.columns:
                        methods.append((f"Classical {model}", metric, float(np.nanmean(sub[metric].to_numpy(dtype=float)))) )
        # Majority-class control: only deterministic classification metrics apply.
        meta = bundle.get("dataset_metadata", {})
        maj = float(meta.get("test_majority_accuracy_baseline", np.nan))
        if np.isfinite(maj):
            methods.append(("Majority Baseline", "accuracy", maj))
        for method, metric, value in methods:
            tidy_rows.append({
                "dataset_id": dataset_id,
                "dataset_name": bundle.get("dataset_metadata", {}).get("dataset_name"),
                "outer_seed": seed,
                "method": method,
                "metric": metric,
                "value": value,
            })
        # Degree-selection frequency and selected-degree AUC/F1 on each outer seed.
        if not loco_df.empty and "selected_degree" in loco_df.columns:
            selected = loco_df["selected_degree"].astype(int).tolist()
            for d in V51_DEGREES:
                selection_rows.append({
                    "dataset_id": dataset_id,
                    "dataset_name": bundle.get("dataset_metadata", {}).get("dataset_name"),
                    "outer_seed": seed,
                    "degree": int(d),
                    "selection_frequency_within_calibrations": float(np.mean(np.asarray(selected) == d)) if selected else float("nan"),
                    "selected_count": int(np.sum(np.asarray(selected) == d)) if selected else 0,
                    "n_calibration_conditions": int(len(selected)),
                })

    tidy = pd.DataFrame(tidy_rows)
    sel = pd.DataFrame(selection_rows)
    if tidy.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    grouped_rows = []
    for (dataset_id, dataset_name, method, metric), sub in tidy.groupby(["dataset_id", "dataset_name", "method", "metric"], dropna=False):
        stats = _metric_summary(sub["value"].to_numpy(dtype=float))
        grouped_rows.append({
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "method": method,
            "metric": metric,
            "n_outer_seeds": stats["n"],
            "mean": stats["mean"],
            "std": stats["std"],
            "se": stats["se"],
            "min": stats["min"],
            "max": stats["max"],
        })
    by_dataset_method_metric = pd.DataFrame(grouped_rows)

    auc = by_dataset_method_metric[by_dataset_method_metric["metric"] == "roc_auc"].copy()
    wide_auc = auc.pivot_table(index=["dataset_id", "dataset_name"], columns="method", values=["mean", "std", "se"], aggfunc="first")
    wide_auc.columns = [f"{a}__{b}" for a, b in wide_auc.columns]
    wide_auc = wide_auc.reset_index()

    overall_rows = []
    for method, sub in auc.groupby("method"):
        vals = sub["mean"].to_numpy(dtype=float)
        st = _metric_summary(vals)
        overall_rows.append({
            "method": method,
            "n_datasets": st["n"],
            "mean_of_dataset_mean_auc": st["mean"],
            "std_across_dataset_mean_auc": st["std"],
            "min_dataset_mean_auc": st["min"],
            "max_dataset_mean_auc": st["max"],
        })
    overall = pd.DataFrame(overall_rows).sort_values("mean_of_dataset_mean_auc", ascending=False) if overall_rows else pd.DataFrame()

    # Degree selection across outer seeds, treating calibration profiles as repeated
    # conditions within each seed rather than independent outer observations.
    if not sel.empty:
        sel_summary_rows = []
        for (dataset_id, dataset_name, degree), sub in sel.groupby(["dataset_id", "dataset_name", "degree"]):
            vals = sub["selection_frequency_within_calibrations"].to_numpy(dtype=float)
            st = _metric_summary(vals)
            sel_summary_rows.append({
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "degree": int(degree),
                "mean_selection_frequency": st["mean"],
                "std_selection_frequency": st["std"],
                "n_outer_seeds": st["n"],
            })
        sel_summary = pd.DataFrame(sel_summary_rows)
    else:
        sel_summary = pd.DataFrame()

    out_root = OUTPUT_ROOT_DIR
    tidy.to_csv(out_root / "v5_2_outer_seed_method_metrics_tidy.csv", index=False)
    by_dataset_method_metric.to_csv(out_root / "v5_2_repeated_outer_mean_std_by_dataset_method_metric.csv", index=False)
    wide_auc.to_csv(out_root / "v5_2_repeated_outer_auc_mean_std_wide.csv", index=False)

    # Human-readable AUC table for the manuscript workflow: mean ± std across outer seeds.
    auc_pretty_rows = []
    for (dataset_id, dataset_name), sub in auc.groupby(["dataset_id", "dataset_name"]):
        row = {"dataset_id": dataset_id, "dataset_name": dataset_name}
        for method in sorted(sub["method"].unique()):
            one = sub[sub["method"] == method].iloc[0]
            m = float(one["mean"])
            sd = float(one["std"])
            row[method] = f"{m:.4f} ± {sd:.4f}"
        auc_pretty_rows.append(row)
    auc_pretty = pd.DataFrame(auc_pretty_rows)
    auc_pretty.to_csv(out_root / "v5_2_auc_mean_plus_minus_std.csv", index=False)

    overall.to_csv(out_root / "v5_2_cross_dataset_auc_summary.csv", index=False)
    sel.to_csv(out_root / "v5_2_degree_selection_by_outer_seed.csv", index=False)
    sel_summary.to_csv(out_root / "v5_2_degree_selection_summary.csv", index=False)

    report = {
        "version": "V5.2",
        "outer_seeds": list(V52_OUTER_SEEDS),
        "n_outer_seeds_target": len(V52_OUTER_SEEDS),
        "n_calibration_conditions": 5,
        "datasets": list(DATASETS_TO_RUN),
        "note": "Calibration conditions are repeated evaluations on the same outer test set within each outer seed and are not treated as independent outer replicates.",
        "auc_overall": overall.to_dict(orient="records"),
        "auc_mean_plus_minus_std_file": str((out_root / "v5_2_auc_mean_plus_minus_std.csv").resolve()),
    }
    with (out_root / "v5_2_repeated_outer_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=json_safe)
    return tidy, by_dataset_method_metric, wide_auc, overall


def main_v52():
    global ACTIVE_RUN_SEED, OUTPUT_DIR
    if not DATASETS_TO_RUN:
        raise RuntimeError("DATASETS_TO_RUN is empty.")
    unknown = [d for d in DATASETS_TO_RUN if d not in DATASET_DISPLAY_NAMES]
    if unknown:
        raise ValueError(f"Unknown dataset IDs: {unknown}")
    if len(V52_OUTER_SEEDS) < 2:
        raise ValueError("V5.2 repeated outer evaluation requires at least two outer seeds.")

    print("=" * 90)
    print("RANQ-VKIL PAPER 2 -- V5.2 REPEATED OUTER MULTI-DATASET")
    print(f"Datasets ({len(DATASETS_TO_RUN)}): {', '.join(DATASETS_TO_RUN)}")
    print(f"Outer seeds ({len(V52_OUTER_SEEDS)}): {list(V52_OUTER_SEEDS)}")
    print("Calibration conditions: 5 per outer seed")
    print("Primary repeated-evaluation summary: mean ± std AUC across independent outer seeds")
    print("=" * 90)

    synthetic_check = _generate_synthetic_pair_for_validation()
    print(
        "Synthetic validation: "
        f"clean={synthetic_check['shape_clean']} "
        f"noisy={synthetic_check['shape_noisy']} "
        f"shared_classes={synthetic_check['class_counts']} | "
        f"full-data standardized PCA4 variance: "
        f"clean={synthetic_check['full_dataset_standardized_pca4_explained_variance_clean']:.4f} "
        f"noisy={synthetic_check['full_dataset_standardized_pca4_explained_variance_noisy']:.4f}"
    )
    (OUTPUT_ROOT_DIR / "synthetic_validation_v5_2.json").write_text(
        json.dumps(synthetic_check, ensure_ascii=False, indent=2, default=json_safe),
        encoding="utf-8",
    )

    backend, calibration_profiles = prepare_shared_ibm_context()
    all_runs = []

    manifest_path = OUTPUT_ROOT_DIR / "v5_2_run_manifest.csv"
    completed = pd.read_csv(manifest_path).to_dict("records") if (V52_RESUME and manifest_path.is_file()) else []
    completed_keys = {(str(r.get("dataset_id")), int(r.get("outer_seed"))) for r in completed if str(r.get("status")) == "completed"}

    for seed_idx, outer_seed in enumerate(V52_OUTER_SEEDS, start=1):
        ACTIVE_RUN_SEED = int(outer_seed)
        print("\n" + "=" * 90)
        print(f"OUTER REPEAT {seed_idx}/{len(V52_OUTER_SEEDS)} | SEED={outer_seed}")
        print("=" * 90)
        for dataset_no, dataset_id in enumerate(DATASETS_TO_RUN, start=1):
            key = (dataset_id, int(outer_seed))
            run_dir = OUTPUT_ROOT_DIR / dataset_id / f"outer_seed_{outer_seed}"
            result_path = run_dir / "ranq_vkil_paper2_v5_2_results.json"
            if V52_RESUME and result_path.is_file():
                can_resume = key in completed_keys
                bundle = None
                if not can_resume:
                    try:
                        bundle = json.loads(result_path.read_text(encoding="utf-8"))
                        can_resume = (
                            bundle.get("version") == "V5.2"
                            and int(bundle.get("config", {}).get("outer_seed", -1)) == int(outer_seed)
                        )
                    except Exception:
                        can_resume = False
                if can_resume:
                    if bundle is None:
                        bundle = json.loads(result_path.read_text(encoding="utf-8"))
                    print(f"[RESUME] Skipping completed {dataset_id} seed={outer_seed}")
                    all_runs.append({"dataset_id": dataset_id, "outer_seed": int(outer_seed), "bundle": bundle})
                    continue

            print("\n" + "#" * 90)
            print(f"DATASET {dataset_no}/{len(DATASETS_TO_RUN)}: {DATASET_DISPLAY_NAMES[dataset_id]} | OUTER SEED {outer_seed}")            print("#" * 90)
            X_train_raw, X_test_raw, y_train, y_test, dataset_meta = load_outer_pilot_split(dataset_id, split_seed=outer_seed)
            bundle = run_dataset_v51(
                dataset_id,
                dataset_meta,
                X_train_raw,
                X_test_raw,
                y_train,
                y_test,
                backend,
                calibration_profiles,
                run_seed=outer_seed,
                output_dir=run_dir,
            )
            all_runs.append({"dataset_id": dataset_id, "outer_seed": int(outer_seed), "bundle": bundle})
            manifest_row = pd.DataFrame([{
                "dataset_id": dataset_id,
                "outer_seed": int(outer_seed),
                "status": "completed",
                "result_path": str(result_path.resolve()),
            }])
            if manifest_path.is_file():
                old = pd.read_csv(manifest_path)
                new = pd.concat([old, manifest_row], ignore_index=True)
            else:
                new = manifest_row
            new = new.drop_duplicates(subset=["dataset_id", "outer_seed"], keep="last")
            new.to_csv(manifest_path, index=False)

        # Incremental aggregate after every outer seed so interrupted runs retain results.
        tidy, by_dmm, wide_auc, overall = aggregate_repeated_outer_v52(all_runs)
        if not overall.empty:
            print("\n[V5.2 incremental AUC summary: mean of dataset means]")
            print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    tidy, by_dmm, wide_auc, overall = aggregate_repeated_outer_v52(all_runs)
    print("\n" + "=" * 90)
    print("V5.2 REPEATED OUTER EVALUATION FINISHED")
    print("=" * 90)
    print(f"Completed outer-seed/dataset runs: {len(all_runs)} / {len(V52_OUTER_SEEDS) * len(DATASETS_TO_RUN)}")
    if not overall.empty:
        print("\nCross-dataset AUC summary (mean across dataset-specific outer-seed means):")
        print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Root output directory: {OUTPUT_ROOT_DIR.resolve()}")
    print("=" * 90)
    return {
        "runs": all_runs,
        "tidy": tidy,
        "by_dataset_method_metric": by_dmm,
        "wide_auc": wide_auc,
        "overall": overall,
    }

def main_v51():
    """Backward-compatible entry point; V5.2 is now the default experiment."""
    return main_v52()


if __name__ == "__main__":
    main_v52()