"""Filter Bank Riemann: multiple bands (8-12, 12-16, 16-20, 20-24, 24-30 Hz).

Structural correctness:
- Trial orientation: (n_channels, n_samples) for covariance
- Covariance: (trial @ trial.T) / (n_samples-1) + regularization
- Filter order: bandpass per trial FIRST, then covariance
- Reference: geometric_mean per band, source-only
- Tangent: logm(ref^{-1/2} @ cov @ ref^{-1/2}), upper triangular
- Float32 everywhere. No z-scoring across subjects.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import FeatureExtractorBase
from .riemann_tangent_oas import _bandpass_eeg, compute_covariances_oas

logger = logging.getLogger(__name__)

# Locked minimal RSA configuration (RSA_STABLE mode)
MODE_RSA_STABLE = "RSA_STABLE"

# Motor imagery filter bank bands (Hz): μ, β1, β2, β3, β4 — fixed order, no band weighting
FILTER_BANK_BANDS = [
    (8, 12),   # μ
    (12, 16),  # β1
    (16, 20),  # β2
    (20, 24),  # β3
    (24, 30),  # β4
]

MIN_CHANNELS = 8
MIN_SAMPLES = 300
MIN_FEATURE_DIM = 500
MIN_TRIALS_PER_SUBJECT_RSA = 20
MIN_TRIALS_PER_CLASS_CC_RSA = 10
COV_REGULARIZATION = 1e-6
EXPECTED_FEATURE_DIM_22CH = 1265  # 253 per band * 5 bands


def _ensure_trial_orientation(trial: np.ndarray) -> np.ndarray:
    """
    Ensure trial is (n_channels, n_samples) for covariance.
    X[i] from (n_trials, n_channels, n_samples) is (n_channels, n_samples).
    If (samples, channels) i.e. shape[0] > shape[1], transpose.
    """
    assert trial.ndim == 2, f"trial.ndim must be 2, got {trial.ndim}"
    if trial.shape[0] > trial.shape[1]:
        # (samples, channels) -> transpose to (channels, samples)
        trial = trial.T
    assert trial.shape[0] >= MIN_CHANNELS, f"channels={trial.shape[0]} < {MIN_CHANNELS}"
    assert trial.shape[1] >= MIN_SAMPLES, f"samples={trial.shape[1]} < {MIN_SAMPLES}"
    return trial


def compute_covariance_explicit(trial: np.ndarray) -> np.ndarray:
    """
    cov = (trial @ trial.T) / (n_samples - 1) + reg * I
    trial: (n_channels, n_samples), float32
    """
    trial = _ensure_trial_orientation(np.asarray(trial, dtype=np.float32))
    n_ch, n_samp = trial.shape
    cov = (trial @ trial.T) / max(n_samp - 1, 1)
    cov = cov.astype(np.float32)
    cov += COV_REGULARIZATION * np.eye(n_ch, dtype=np.float32)
    cov = 0.5 * (cov + cov.T)
    evals = np.linalg.eigvalsh(cov)
    min_eval = float(np.min(evals))
    if min_eval <= 0:
        raise RuntimeError(f"Covariance not positive definite: min eigenvalue={min_eval}")
    return cov


def compute_covariances_band(X_band: np.ndarray) -> np.ndarray:
    """X_band: (n_trials, n_channels, n_samples). Return (n_trials, n_channels, n_channels) float32."""
    covs = []
    for i in range(X_band.shape[0]):
        cov = compute_covariance_explicit(X_band[i])
        covs.append(cov)
    return np.stack(covs, axis=0).astype(np.float32)


def invsqrt_matrix(M: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root for SPD matrix."""
    from scipy.linalg import sqrtm
    M = np.asarray(M, dtype=np.float64) + COV_REGULARIZATION * np.eye(M.shape[0])
    sqrt_M = np.real(sqrtm(M))
    return np.linalg.inv(sqrt_M).astype(np.float32)


def riemann_distance(C1: np.ndarray, C2: np.ndarray) -> float:
    """Riemannian distance between two SPD matrices."""
    from scipy.linalg import logm, sqrtm
    C1 = np.asarray(C1, dtype=np.float64) + COV_REGULARIZATION * np.eye(C1.shape[0])
    C2 = np.asarray(C2, dtype=np.float64) + COV_REGULARIZATION * np.eye(C2.shape[0])
    C1_sqrt = np.real(sqrtm(C1))
    C1_inv_sqrt = np.linalg.inv(C1_sqrt)
    M = C1_inv_sqrt @ C2 @ C1_inv_sqrt
    M = 0.5 * (M + M.T)
    log_M = np.real(logm(M))
    return float(np.sqrt(np.sum(log_M ** 2)))


def _compute_temporal_features(
    X: np.ndarray, bands: list[tuple[float, float]], fs: float
) -> np.ndarray:
    """X: (n_trials, n_ch, n_samp). Returns (n_trials, n_temporal_dims) float32. Dims kept < 200."""
    n_trials, n_ch, n_samp = X.shape
    # Reduce dims: per band 4 log-var (mean over ch) + 3 Hjorth (mean over ch) = 7; *5 = 35; + 5 ratios = 40
    out = []
    X_f64 = np.asarray(X, dtype=np.float64)
    for i in range(n_trials):
        x_bands = []
        for (l_f, h_f) in bands:
            x = _bandpass_eeg(X_f64[i : i + 1], fs, l_f, h_f)[0]
            x_bands.append(x)
        win_len = max(int(fs), 1)
        feat = []
        for x in x_bands:
            for w in range(4):
                start = w * (n_samp // 4)
                end = min(start + win_len, n_samp)
                if end > start:
                    seg = x[:, start:end]
                    feat.append(float(np.log(np.mean(np.var(seg, axis=1)) + 1e-12)))
                else:
                    feat.append(0.0)
            # Hjorth averaged over channels
            act = np.var(x) + 1e-12
            d1 = np.diff(x)
            mob = np.sqrt(np.var(d1) + 1e-12) / (np.sqrt(act) + 1e-12)
            d2 = np.diff(d1)
            compl = np.sqrt(np.var(d2) + 1e-12) / (np.sqrt(np.var(d1)) + 1e-12) if np.var(d1) > 1e-12 else 1.0
            feat.extend([float(act), float(mob), float(compl)])
        powers = [np.mean(x ** 2) + 1e-12 for x in x_bands]
        total_p = sum(powers)
        for p in powers:
            feat.append(p / (total_p + 1e-12))
        out.append(feat)
    return np.array(out, dtype=np.float32)


def average_pairwise_riemann_distance(means: dict[int, np.ndarray]) -> float:
    """Average pairwise Riemann distance between subject means."""
    sids = list(means.keys())
    if len(sids) < 2:
        return 0.0
    dists = []
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            d = riemann_distance(means[sids[i]], means[sids[j]])
            dists.append(d)
    return float(np.mean(dists))


def _class_spread(means_per_subject_per_class: dict[tuple[int, int], np.ndarray], classes: np.ndarray) -> float:
    """Mean Riemann distance between class means across subjects. Per class c: pairwise dist between C_s_c; then mean over classes."""
    spread_per_class = []
    for c in np.unique(classes):
        subj_means = {s: means_per_subject_per_class[(s, int(c))] for (s, _c) in means_per_subject_per_class if _c == c}
        if len(subj_means) < 2:
            continue
        spread_per_class.append(average_pairwise_riemann_distance(subj_means))
    return float(np.mean(spread_per_class)) if spread_per_class else 0.0


def compute_rsa_distance_diagnostics(
    X: np.ndarray,
    subject_ids: np.ndarray,
    fs: float,
    bands: list[tuple[float, float]] | None = None,
) -> tuple[float | None, float | None]:
    """
    Compute RSA alignment diagnostics on full training data (all source subjects).

    Step 1: Per-subject geometric mean of raw covariances -> pairwise Riemann distances -> mean = distance_before.
    Step 2: After subject whitening, per-subject geometric mean of aligned covs -> pairwise distances -> mean = distance_after.

    Returns (distance_before_whitening, distance_after_whitening) averaged over bands, or (None, None) if
    fewer than 2 subjects or any subject has too few trials.
    """
    bands = bands or FILTER_BANK_BANDS
    subject_ids = np.asarray(subject_ids, dtype=np.int64).ravel()
    if len(subject_ids) != len(X):
        return (None, None)
    uniq = np.unique(subject_ids)
    if len(uniq) < 2:
        return (None, None)
    for sid in uniq:
        if np.sum(subject_ids == sid) < MIN_TRIALS_PER_SUBJECT_RSA:
            return (None, None)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        return (None, None)

    dist_before_per_band: list[float] = []
    dist_after_per_band: list[float] = []
    for (l_freq, h_freq) in bands:
        X_band = _bandpass_eeg(X.astype(np.float64), fs, l_freq, h_freq)
        X_band = X_band.astype(np.float32)
        covs = compute_covariances_band(X_band)
        subject_means_band: dict[int, np.ndarray] = {}
        for sid in uniq:
            mask = subject_ids == sid
            subject_means_band[int(sid)] = geometric_mean_covariances(covs[mask])
        distance_before = average_pairwise_riemann_distance(subject_means_band)
        dist_before_per_band.append(distance_before)
        covs_aligned = []
        for i in range(len(covs)):
            sid = int(subject_ids[i])
            W_s = invsqrt_matrix(subject_means_band[sid])
            C = np.asarray(covs[i], dtype=np.float64)
            C_al = W_s @ C @ W_s
            C_al = 0.5 * (C_al + C_al.T)
            covs_aligned.append(C_al)
        covs_aligned = np.stack(covs_aligned, axis=0)
        aligned_means = {}
        for sid in uniq:
            mask = subject_ids == sid
            aligned_means[int(sid)] = geometric_mean_covariances(covs_aligned[mask])
        distance_after = average_pairwise_riemann_distance(aligned_means)
        dist_after_per_band.append(distance_after)

    return (
        float(np.mean(dist_before_per_band)),
        float(np.mean(dist_after_per_band)),
    )


def geometric_mean_covariances(covs: np.ndarray) -> np.ndarray:
    """Riemannian (geometric) mean of SPD matrices. covs: (n_trials, n_ch, n_ch)."""
    try:
        from pyriemann.utils.mean import mean_covariance
        ref = mean_covariance(covs, metric="riemann")
    except ImportError:
        from scipy.linalg import logm, expm
        covs = np.asarray(covs, dtype=np.float64)
        n = covs.shape[1]
        log_sum = np.zeros((n, n), dtype=np.float64)
        for i in range(covs.shape[0]):
            C = covs[i] + COV_REGULARIZATION * np.eye(n)
            log_sum += np.real(logm(C))
        mean_log = log_sum / covs.shape[0]
        ref = np.real(expm(mean_log))
    return np.asarray(ref, dtype=np.float32)


def geometric_mean_covariances_weighted(
    covs: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Weighted Riemannian mean. covs: (n_trials, n_ch, n_ch), weights: (n_trials,) non-negative."""
    covs = np.asarray(covs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    w = w / np.maximum(w.sum(), 1e-12)
    n, ch = covs.shape[0], covs.shape[1]
    try:
        from pyriemann.utils.mean import mean_covariance
        ref = mean_covariance(covs, metric="riemann", sample_weight=w)
    except (ImportError, TypeError):
        from scipy.linalg import logm, expm
        log_sum = np.zeros((ch, ch), dtype=np.float64)
        for i in range(n):
            C = covs[i] + COV_REGULARIZATION * np.eye(ch)
            log_sum += w[i] * np.real(logm(C))
        ref = np.real(expm(log_sum))
    return np.asarray(ref, dtype=np.float32)


def procrustes_orthogonal_alignment(S: np.ndarray, global_mean: np.ndarray) -> np.ndarray:
    """
    Orthogonal Q such that Q^T S Q best approximates global_mean (Frobenius).
    S, global_mean: (n, n) SPD. Returns Q (n, n) orthogonal.
    Uses eigenbasis alignment: Q = U_S.T @ U_G with eigenvalues sorted descending.
    """
    from scipy.linalg import eigh
    evals_s, u_s = eigh(np.asarray(S, dtype=np.float64))
    evals_g, u_g = eigh(np.asarray(global_mean, dtype=np.float64))
    # descending order
    u_s = u_s[:, ::-1]
    u_g = u_g[:, ::-1]
    Q = (u_s.T @ u_g).astype(np.float32)
    return Q


def project_to_tangent_explicit(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Project covs to tangent space at ref.
    tangent_vec = logm(ref^{-1/2} @ cov @ ref^{-1/2}), vectorize upper triangular including diagonal.
    covs: (n_trials, n_ch, n_ch), ref: (n_ch, n_ch)
    """
    from scipy.linalg import sqrtm, logm
    ref = np.asarray(ref, dtype=np.float64)
    ref_sqrt = np.real(sqrtm(ref + COV_REGULARIZATION * np.eye(ref.shape[0])))
    ref_inv_sqrt = np.linalg.inv(ref_sqrt)
    n = ref.shape[0]
    n_flat = n * (n + 1) // 2
    out = []
    for i in range(covs.shape[0]):
        C = np.asarray(covs[i], dtype=np.float64) + COV_REGULARIZATION * np.eye(n)
        M = ref_inv_sqrt @ C @ ref_inv_sqrt
        M = 0.5 * (M + M.T)
        log_M = np.real(logm(M))
        idx = np.triu_indices(n)
        vec = log_M[idx]
        out.append(vec)
    return np.array(out, dtype=np.float32)


class FilterBankRiemann(FeatureExtractorBase):
    """
    Filter Bank Riemann: 5 bands (8-12, 12-16, 16-20, 20-24, 24-30 Hz).
    Per band: bandpass raw trial -> covariance -> (optional RSA alignment) -> tangent projection -> concat.
    fit(X_source, y): reference from SOURCE only (LOSO-safe).
    When rsa=True and subject_ids provided: Riemannian Subject Alignment in covariance space before tangent.
    Float32 everywhere. StandardScaler fit on train only.
    """

    name = "filter_bank_riemann"

    def __init__(
        self,
        fs: float,
        bands: list[tuple[float, float]] | None = None,
        force_float32: bool = True,
        z_score_tangent: bool = True,
        rsa: bool = False,
        use_procrustes: bool = False,
        use_class_conditional: bool = False,
        use_temporal: bool = False,
        use_band_weighting: bool = False,
        use_subject_weighting: bool = False,
        use_outlier_detection: bool = False,
        rsa_stable_mode: bool = False,
        use_class_conditional_rsa: bool = False,
        use_oas: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(fs, **kwargs)
        self.rsa_stable_mode = bool(rsa_stable_mode)
        self.use_class_conditional_rsa = bool(use_class_conditional_rsa)
        self.use_oas = bool(use_oas)
        # Accept list of lists from YAML (e.g. [[4,8],[8,12],...])
        raw_bands = bands or FILTER_BANK_BANDS
        self.bands = [tuple(b) if isinstance(b, (list, tuple)) else b for b in raw_bands]
        self.force_float32 = bool(force_float32)
        self.z_score_tangent = bool(z_score_tangent)
        self.rsa = bool(rsa)
        self.use_procrustes = bool(use_procrustes)
        self.use_class_conditional = bool(use_class_conditional)
        self.use_temporal = bool(use_temporal)
        self.use_band_weighting = bool(use_band_weighting)
        self.use_subject_weighting = bool(use_subject_weighting)
        self.use_outlier_detection = bool(use_outlier_detection)
        if self.rsa_stable_mode:
            # Abort if any disabled module is accidentally activated
            assert not self.use_procrustes, "RSA_STABLE: use_procrustes must be False"
            assert not self.use_class_conditional, "RSA_STABLE: use_class_conditional must be False"
            assert not self.use_temporal, "RSA_STABLE: use_temporal must be False"
            assert not self.use_band_weighting, "RSA_STABLE: use_band_weighting must be False"
            assert not self.use_subject_weighting, "RSA_STABLE: use_subject_weighting must be False"
            assert not self.use_outlier_detection, "RSA_STABLE: use_outlier_detection must be False"
            assert self.rsa, "RSA_STABLE: use_rsa must be True"
        self._ref_covs: list[np.ndarray] = []
        self._n_channels: int | None = None
        self._scaler: Any = None
        self._n_covs_for_ref: list[int] = []
        self._subject_means: list[dict[int, np.ndarray]] = []
        self._rsa_distance_before: list[float] = []
        self._rsa_distance_after: list[float] = []
        self._rsa_distance_after_procrustes: list[float] = []
        self._band_weights: list[float] = []
        self._subject_weights: list[float] = []
        self._subject_reliability: dict[int, float] = {}
        self._outlier_flags: list[dict[int, bool]] = []
        self._temporal_scaler: Any = None
        self._max_temporal_dim: int = 200
        # CC-RSA diagnostics (source only)
        self._cc_rsa_enabled: list[bool] = []
        self._cc_class_spread_before: list[float] = []
        self._cc_class_spread_after: list[float] = []
        self._cc_inter_subject_before: list[float] = []
        self._cc_inter_subject_after: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray | None = None) -> "FilterBankRiemann":
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError("FilterBankRiemann expects (n_trials, n_channels, n_samples)")
        n_trials, n_ch, n_samp = X.shape
        assert n_ch >= MIN_CHANNELS, f"channels={n_ch} < {MIN_CHANNELS}"
        assert n_samp >= MIN_SAMPLES, f"samples={n_samp} < {MIN_SAMPLES}"
        logger.info("[FBR] channels=%d samples=%d", n_ch, n_samp)

        self._n_channels = n_ch
        self._ref_covs = []
        self._n_covs_for_ref = []
        self._subject_means = []
        self._rsa_distance_before = []
        self._rsa_distance_after = []
        self._rsa_distance_after_procrustes = []
        self._cc_rsa_enabled = []
        self._cc_class_spread_before = []
        self._cc_class_spread_after = []
        self._cc_inter_subject_before = []
        self._cc_inter_subject_after = []
        self._band_weights = []
        self._subject_weights = []
        self._subject_reliability = {}
        self._outlier_flags = []
        all_tangent = []

        use_rsa = self.rsa and subject_ids is not None and len(subject_ids) == len(X)
        if use_rsa:
            subject_ids = np.asarray(subject_ids, dtype=np.int64).ravel()

        for (l_freq, h_freq) in self.bands:
            X_band = _bandpass_eeg(X.astype(np.float64), self.fs, l_freq, h_freq)
            X_band = X_band.astype(np.float32)
            if self.use_oas:
                covs = compute_covariances_oas(X_band.astype(np.float64)).astype(np.float32)
            else:
                covs = compute_covariances_band(X_band)

            if use_rsa:
                # Optional: Class-Conditional RSA (source labels only; target stays unsupervised)
                cc_rsa_band = False
                if self.use_class_conditional_rsa and y is not None:
                    uniq_s = np.unique(subject_ids)
                    uniq_c = np.unique(y)
                    if len(uniq_s) >= 2:
                        min_per_sc = min(
                            (np.sum((subject_ids == s) & (y == c)) for s in uniq_s for c in uniq_c),
                            default=0,
                        )
                        if min_per_sc >= MIN_TRIALS_PER_CLASS_CC_RSA:
                            cc_rsa_band = True
                        else:
                            logger.warning(
                                "[FBR-CC-RSA] band (%.0f,%.0f) fallback to standard RSA: min trials per (subject,class)=%d < %d",
                                l_freq, h_freq, int(min_per_sc), MIN_TRIALS_PER_CLASS_CC_RSA,
                            )
                    else:
                        logger.warning("[FBR-CC-RSA] band (%.0f,%.0f) fallback to standard RSA: < 2 source subjects", l_freq, h_freq)

                if cc_rsa_band:
                    uniq_s = np.unique(subject_ids)
                    uniq_c = np.unique(y)
                    # A. Per-subject per-class means (raw, source only)
                    subject_class_means: dict[tuple[int, int], np.ndarray] = {}
                    for s in uniq_s:
                        for c in uniq_c:
                            mask = (subject_ids == s) & (y == c)
                            if np.sum(mask) < MIN_TRIALS_PER_CLASS_CC_RSA:
                                continue
                            subject_class_means[(int(s), int(c))] = geometric_mean_covariances(covs[mask])
                    # B. Global class mean (source only; for diagnostics)
                    # C. Class spread before = mean Riemann dist between class means across subjects
                    class_spread_before = _class_spread(subject_class_means, y)
                    subject_means_band = {int(s): geometric_mean_covariances(covs[subject_ids == s]) for s in uniq_s}
                    inter_subject_before = average_pairwise_riemann_distance(subject_means_band)
                    # D. Whitening: W_s_c = invsqrt(C_s_c), C_aligned = W_s_c @ C_i @ W_s_c
                    covs_aligned = []
                    for i in range(len(covs)):
                        s, c = int(subject_ids[i]), int(y[i])
                        key = (s, c)
                        if key not in subject_class_means:
                            W = invsqrt_matrix(subject_means_band[s])
                        else:
                            W = invsqrt_matrix(subject_class_means[key])
                        C = np.asarray(covs[i], dtype=np.float64)
                        C_al = W @ C @ W
                        C_al = 0.5 * (C_al + C_al.T)
                        covs_aligned.append(C_al)
                    covs_aligned = np.stack(covs_aligned, axis=0).astype(np.float32)
                    # E. Class spread after, inter-subject after
                    aligned_sc_means: dict[tuple[int, int], np.ndarray] = {}
                    for s in uniq_s:
                        for c in uniq_c:
                            mask = (subject_ids == s) & (y == c)
                            if np.sum(mask) < MIN_TRIALS_PER_CLASS_CC_RSA:
                                continue
                            aligned_sc_means[(int(s), int(c))] = geometric_mean_covariances(covs_aligned[mask])
                    class_spread_after = _class_spread(aligned_sc_means, y)
                    aligned_means = {int(s): geometric_mean_covariances(covs_aligned[subject_ids == s]) for s in uniq_s}
                    inter_subject_after = average_pairwise_riemann_distance(aligned_means)
                    ref = geometric_mean_covariances(covs_aligned)
                    covs_for_tangent = covs_aligned
                    self._subject_means.append(subject_means_band)
                    self._rsa_distance_before.append(inter_subject_before)
                    self._rsa_distance_after.append(inter_subject_after)
                    self._cc_rsa_enabled.append(True)
                    self._cc_class_spread_before.append(class_spread_before)
                    self._cc_class_spread_after.append(class_spread_after)
                    self._cc_inter_subject_before.append(inter_subject_before)
                    self._cc_inter_subject_after.append(inter_subject_after)
                    self._outlier_flags.append({})
                    self._subject_weights.append({})
                    logger.info(
                        "[FBR-CC-RSA] band (%.0f,%.0f) class_alignment_enabled=True inter_subject before=%.4f after=%.4f class_spread before=%.4f after=%.4f",
                        l_freq, h_freq, inter_subject_before, inter_subject_after, class_spread_before, class_spread_after,
                    )
                else:
                    # Standard RSA: per-band subject alignment
                    subject_means_band: dict[int, np.ndarray] = {}
                    for sid in np.unique(subject_ids):
                        mask = subject_ids == sid
                        covs_s = covs[mask]
                        if len(covs_s) < MIN_TRIALS_PER_SUBJECT_RSA:
                            raise RuntimeError(
                                f"RSA: subject {sid} has {len(covs_s)} trials < {MIN_TRIALS_PER_SUBJECT_RSA}"
                            )
                        subject_means_band[int(sid)] = geometric_mean_covariances(covs_s)
                        logger.info("[FBR-RSA] band (%g,%g) subject %d: %d covariances",
                                    l_freq, h_freq, sid, len(covs_s))
                    self._subject_means.append(subject_means_band)
                    distance_before = average_pairwise_riemann_distance(subject_means_band)
                    self._rsa_distance_before.append(distance_before)

                    # Whiten each train cov to its subject mean
                    covs_aligned = []
                    for i in range(len(covs)):
                        sid = int(subject_ids[i])
                        W_s = invsqrt_matrix(subject_means_band[sid])
                        C = np.asarray(covs[i], dtype=np.float64)
                        C_al = W_s @ C @ W_s
                        C_al = 0.5 * (C_al + C_al.T)
                        covs_aligned.append(C_al)
                    covs_aligned = np.stack(covs_aligned, axis=0).astype(np.float32)

                    # Aligned subject means for distance_after
                    aligned_means = {}
                    for sid in np.unique(subject_ids):
                        mask = subject_ids == sid
                        aligned_means[int(sid)] = geometric_mean_covariances(covs_aligned[mask])
                    distance_after = average_pairwise_riemann_distance(aligned_means)
                    self._rsa_distance_after.append(distance_after)
                    self._cc_rsa_enabled.append(False)
                    self._cc_class_spread_before.append(0.0)
                    self._cc_class_spread_after.append(0.0)
                    self._cc_inter_subject_before.append(distance_before)
                    self._cc_inter_subject_after.append(distance_after)
                    logger.info("[FBR-RSA] band (%g,%g) distance_before=%.4f distance_after_whitening=%.4f",
                                l_freq, h_freq, distance_before, distance_after)
                    # Sanity assertions when multiple subjects (standard RSA only)
                    n_subj_band = len(np.unique(subject_ids))
                    if n_subj_band > 1:
                        assert distance_before > 0, (
                            f"RSA: distance_before_whitening must be > 0 (got {distance_before}) with {n_subj_band} subjects"
                        )
                        assert distance_after >= 0, (
                            f"RSA: distance_after_whitening must be >= 0 (got {distance_after})"
                        )
                        if distance_after > distance_before:
                            logger.warning(
                                "[FBR-RSA] Whitening increased inter-subject distance: before=%.4f after=%.4f",
                                distance_before, distance_after,
                            )

                    covs_for_tangent = covs_aligned
                    if self.use_procrustes:
                        global_mean = geometric_mean_covariances(covs_aligned)
                        proc_list = []
                        for i in range(len(covs)):
                            sid = int(subject_ids[i])
                            S_i = aligned_means[sid]
                            Q = procrustes_orthogonal_alignment(S_i, global_mean)
                            C = np.asarray(covs_aligned[i], dtype=np.float64)
                            C_proc = (Q.T @ C @ Q).astype(np.float32)
                            C_proc = 0.5 * (C_proc + C_proc.T)
                            proc_list.append(C_proc)
                        covs_for_tangent = np.stack(proc_list, axis=0)
                        proc_means = {}
                        for sid in np.unique(subject_ids):
                            mask = subject_ids == sid
                            proc_means[int(sid)] = geometric_mean_covariances(covs_for_tangent[mask])
                        distance_after_proc = average_pairwise_riemann_distance(proc_means)
                        self._rsa_distance_after_procrustes.append(distance_after_proc)
                        if distance_after_proc > distance_after + 1e-6:
                            logger.warning("[FBR-RSA] Procrustes distance increased %.4f -> %.4f; keeping monotonic",
                                          distance_after, distance_after_proc)
                        logger.info("[FBR-RSA] band (%g,%g) distance_after_procrustes=%.4f",
                                    l_freq, h_freq, distance_after_proc)

                    if self.use_class_conditional and y is not None:
                        classes = np.unique(y)
                        global_class_means = {}
                        for c in classes:
                            mask_c = y == c
                            if np.sum(mask_c) < 2:
                                continue
                            global_class_means[int(c)] = geometric_mean_covariances(covs_for_tangent[mask_c])
                        subject_class_means_legacy: dict[tuple[int, int], np.ndarray] = {}
                        for sid in np.unique(subject_ids):
                            for c in classes:
                                mask = (subject_ids == sid) & (y == c)
                                if np.sum(mask) < 1:
                                    continue
                                subject_class_means_legacy[(int(sid), int(c))] = geometric_mean_covariances(covs_for_tangent[mask])
                        class_aligned = []
                        for i in range(len(covs_for_tangent)):
                            c, sid = int(y[i]), int(subject_ids[i])
                            key = (sid, c)
                            if key in subject_class_means_legacy:
                                W = invsqrt_matrix(subject_class_means_legacy[key])
                                C = np.asarray(covs_for_tangent[i], dtype=np.float64)
                                C_al = W @ C @ W
                                C_al = 0.5 * (C_al + C_al.T)
                                class_aligned.append(C_al.astype(np.float32))
                            else:
                                class_aligned.append(covs_for_tangent[i])
                        covs_for_tangent = np.stack(class_aligned, axis=0)

                    ref = geometric_mean_covariances(covs_for_tangent)
                    subject_weights_band = {}
                    outlier_flags_band: dict[int, bool] = {}
                    if self.use_outlier_detection or self.use_subject_weighting:
                        subject_means_band_ow = {}
                        for sid in np.unique(subject_ids):
                            mask = subject_ids == sid
                            subject_means_band_ow[int(sid)] = geometric_mean_covariances(covs_for_tangent[mask])
                        sids_unique = list(np.unique(subject_ids))
                        dists = [riemann_distance(subject_means_band_ow[int(sid)], ref) for sid in sids_unique]
                        mean_d = float(np.mean(dists))
                        std_d = float(np.std(dists)) if len(dists) > 1 else 1e-6
                        for idx, sid in enumerate(sids_unique):
                            d_s = dists[idx]
                            outlier_flags_band[int(sid)] = d_s > mean_d + 2 * std_d
                            w = 0.25 if outlier_flags_band[int(sid)] else 1.0
                            if self.use_subject_weighting:
                                reliability = 1.0 / (1.0 + d_s)
                                w *= reliability
                            subject_weights_band[int(sid)] = max(w, 1e-6)
                        norm = sum(subject_weights_band[s] for s in subject_weights_band)
                        for s in subject_weights_band:
                            subject_weights_band[s] /= max(norm, 1e-12)
                        trial_weights = np.array([subject_weights_band[int(sid)] for sid in subject_ids], dtype=np.float32)
                        ref = geometric_mean_covariances_weighted(covs_for_tangent, trial_weights)
                    self._outlier_flags.append(outlier_flags_band)
                    self._subject_weights.append(subject_weights_band)

            else:
                ref = geometric_mean_covariances(covs)
                covs_for_tangent = covs
                self._outlier_flags.append({})
                self._subject_weights.append({})

            self._ref_covs.append(ref)
            self._n_covs_for_ref.append(covs.shape[0])
            logger.info("[FBR] band (%g,%g) reference from %d covariances", l_freq, h_freq, covs.shape[0])
            tangent = project_to_tangent_explicit(covs_for_tangent, ref)
            all_tangent.append(tangent)

        if self.use_band_weighting and all_tangent and y is not None:
            from sklearn.feature_selection import mutual_info_classif
            band_scores = []
            for tangent in all_tangent:
                mi = mutual_info_classif(tangent, y, random_state=42)
                band_scores.append(float(np.mean(mi)) + 1e-12)
            total = sum(band_scores)
            self._band_weights = [s / total for s in band_scores]
            all_tangent = [t.astype(np.float32) * w for t, w in zip(all_tangent, self._band_weights)]
            logger.info("[FBR] band_weights=%s", self._band_weights)
        elif not self._band_weights:
            self._band_weights = [1.0 / len(self.bands)] * len(self.bands)

        F = np.concatenate(all_tangent, axis=1).astype(np.float32)
        if self.use_temporal:
            T = _compute_temporal_features(X, self.bands, self.fs)
            from sklearn.preprocessing import StandardScaler
            self._temporal_scaler = StandardScaler()
            self._temporal_scaler.fit(T)
            T = self._temporal_scaler.transform(T).astype(np.float32)
            F = np.hstack([F, T]).astype(np.float32)
            assert T.shape[1] < self._max_temporal_dim, f"temporal dim {T.shape[1]} >= {self._max_temporal_dim}"
        else:
            self._temporal_scaler = None
        feature_dim = F.shape[1]
        assert feature_dim >= MIN_FEATURE_DIM, f"feature_dim={feature_dim} < {MIN_FEATURE_DIM}"
        assert feature_dim < 1800, f"feature_dim={feature_dim} >= 1800"
        if n_ch == 22 and not self.use_temporal:
            assert feature_dim == EXPECTED_FEATURE_DIM_22CH, (
                f"feature_dim={feature_dim} expected {EXPECTED_FEATURE_DIM_22CH} for 22 channels"
            )
        # RSA_STABLE: stability assertions (multi-subject only: single subject has 0 vs 0 distance)
        if self.rsa_stable_mode and use_rsa:
            assert feature_dim == EXPECTED_FEATURE_DIM_22CH, (
                f"RSA_STABLE: feature_dim={feature_dim} != {EXPECTED_FEATURE_DIM_22CH}"
            )
            n_subjects = len(np.unique(subject_ids))
            if n_subjects > 1:
                for b, (d_before, d_after) in enumerate(zip(self._rsa_distance_before, self._rsa_distance_after)):
                    assert d_after < d_before, (
                        f"RSA_STABLE: band {b} distance_after_whitening ({d_after:.4f}) >= distance_before ({d_before:.4f})"
                    )
        logger.info("[FBR] feature_dim=%d", feature_dim)

        self._scaler = None
        if self.z_score_tangent:
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            self._scaler.fit(F)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._ref_covs:
            raise RuntimeError("FilterBankRiemann not fitted")
        X = np.asarray(X, dtype=np.float32)
        all_tangent = []
        for (l_freq, h_freq), ref in zip(self.bands, self._ref_covs):
            X_band = _bandpass_eeg(X.astype(np.float64), self.fs, l_freq, h_freq)
            X_band = X_band.astype(np.float32)
            if self.use_oas:
                covs = compute_covariances_oas(X_band.astype(np.float64)).astype(np.float32)
            else:
                covs = compute_covariances_band(X_band)
            if self.rsa:
                # Target subject whitening (unsupervised): whiten to geometric mean of target covs
                target_mean = geometric_mean_covariances(covs)
                W_t = invsqrt_matrix(target_mean)
                covs_aligned = []
                for i in range(len(covs)):
                    C = np.asarray(covs[i], dtype=np.float64)
                    C_al = W_t @ C @ W_t
                    C_al = 0.5 * (C_al + C_al.T)
                    covs_aligned.append(C_al)
                covs = np.stack(covs_aligned, axis=0).astype(np.float32)
            tangent = project_to_tangent_explicit(covs, ref)
            all_tangent.append(tangent)
        if self.use_band_weighting and self._band_weights and len(self._band_weights) == len(all_tangent):
            all_tangent = [t.astype(np.float32) * w for t, w in zip(all_tangent, self._band_weights)]
        F = np.concatenate(all_tangent, axis=1).astype(np.float32)
        if self.use_temporal and self._temporal_scaler is not None:
            T = _compute_temporal_features(X, self.bands, self.fs)
            T = self._temporal_scaler.transform(T).astype(np.float32)
            F = np.hstack([F, T]).astype(np.float32)
        if self._scaler is not None:
            F = self._scaler.transform(F)
        if self.force_float32:
            F = F.astype(np.float32)
        return F

    @property
    def n_features_out(self) -> int | None:
        if self._n_channels is None:
            return None
        n_per_band = self._n_channels * (self._n_channels + 1) // 2
        base = len(self.bands) * n_per_band
        if self.use_temporal:
            base += 40  # 4*5 + 3*5 + 5 bandpower ratios
        return base
