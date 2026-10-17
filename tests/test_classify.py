import numpy as np
import pytest
import scipy.sparse as sp
import simulate as s

import scpreflight


def state(X, **kw):
    return scpreflight.check(s.make(X, **kw)).matrix("X")


def test_raw_counts(counts):
    st = state(counts)
    assert st.kind == "raw_counts"
    assert st.confidence == "high"


@pytest.mark.parametrize(
    "target,base,expected_base",
    [(1e4, None, "e"), (1e6, 2, "2"), (1e4, 10, "10"), (1e6, None, "e")],
)
def test_log1p_normalized_base_and_target(counts, target, base, expected_base):
    st = state(s.log1p(s.normalize_total(counts, target), base=base))
    assert st.kind == "log1p_normalized"
    assert st.details["log_base"] == expected_base
    assert st.details["target_sum"] == pytest.approx(target, rel=1e-3)


def test_recovers_original_depth(counts):
    true_median = np.median(np.asarray(counts.sum(axis=1)).ravel())
    st = state(s.log1p(s.normalize_total(counts)))
    assert st.details["estimated_median_counts_per_cell"] == pytest.approx(true_median, rel=0.02)


def test_median_target(counts):
    st = state(s.log1p(s.normalize_total(counts, None)))
    assert "median library size" in st.details["target_sum_label"]


def test_normalized_not_logged(counts):
    st = state(s.normalize_total(counts, 1e6))
    assert st.kind == "normalized_linear"
    assert "CPM" in st.label


def test_proportions(counts):
    st = state(s.normalize_total(counts, 1.0))
    assert st.kind == "normalized_linear"
    assert "proportions" in st.label


def test_double_log(counts):
    st = state(s.log1p(s.log1p(s.normalize_total(counts))))
    assert st.kind == "log1p_twice"
    assert any(f.level == "error" for f in st.findings)


def test_log_of_unnormalized_counts(counts):
    st = state(s.log1p(counts))
    assert st.kind == "log1p_counts"


def test_hvg_subset_still_detected(counts):
    st = state(s.log1p(s.normalize_total(counts))[:, :300], var_names=s.gene_names(300))
    assert st.kind == "log1p_normalized"
    assert "target_sum" not in st.details  # sums vary after subsetting


def test_scaled_clipped(counts):
    st = state(s.scale(s.log1p(s.normalize_total(counts)), max_value=10))
    assert st.kind == "scaled"
    assert st.details["clipped_at"] == 10.0


def test_scaled_unclipped(counts):
    st = state(s.scale(s.log1p(s.normalize_total(counts)), max_value=None))
    assert st.kind == "scaled"
    assert st.details["clipped_at"] is None


def test_residuals_are_signed_other(counts):
    D = counts.toarray()
    R = (D - D.mean(0)) / np.sqrt(D.mean(0) + 1) * 0.5 + 0.1
    assert state(R).kind == "signed_other"


def test_tpm_like(counts):
    rng = np.random.default_rng(0)
    T = counts.toarray() / rng.integers(500, 5000, counts.shape[1])
    T = T / T.sum(1, keepdims=True) * 1e6
    assert state(sp.csr_matrix(T)).kind == "tpm_like"
    assert state(s.log1p(sp.csr_matrix(T))).kind == "log_tpm_like"


def test_fractional_counts(counts):
    F = counts.copy()
    F.data = F.data * np.random.default_rng(0).uniform(0.7, 1.0, F.data.size)
    assert state(F).kind == "fractional_counts"


def test_binary(counts):
    B = counts.copy()
    B.data[:] = 1
    assert state(B).kind == "binary"


def test_random_continuous_is_not_called_lognormalized():
    rng = np.random.default_rng(3)
    X = rng.lognormal(0, 1, (500, 800)) * (rng.random((500, 800)) < 0.1)
    st = state(sp.csr_matrix(np.log1p(X)))
    assert st.kind not in ("log1p_normalized", "log1p_counts", "normalized_linear")


def test_float64_and_dense_input(counts):
    L = s.log1p(s.normalize_total(counts)).toarray().astype(np.float64)
    assert state(L).kind == "log1p_normalized"


def test_all_zero():
    st = state(sp.csr_matrix((50, 40)), var_names=s.gene_names(40))
    assert st.kind == "all_zero"
