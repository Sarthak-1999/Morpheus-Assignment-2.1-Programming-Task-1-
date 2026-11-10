"""Streaming data-handling tests for the independent Assignment 2.2 package."""

import numpy as np
import pytest

from numcompute_stream.io import iter_supervised_csv
from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
from numcompute_stream.stats import StreamingStats


@pytest.fixture
def measurements():
    return np.array([[1., 2.], [2., 4.], [3., 6.], [4., 8.], [5., 10.]])


@pytest.mark.parametrize("cut", [1, 2, 3, 4])
def test_scaler_streaming_equals_batch(measurements, cut):
    a = StandardScaler().fit(measurements)
    b = StandardScaler().partial_fit(measurements[:cut]).partial_fit(measurements[cut:])
    np.testing.assert_allclose(b.mean_, a.mean_)
    np.testing.assert_allclose(b.var_, a.var_)
    np.testing.assert_allclose(b.transform(measurements), a.transform(measurements))


def test_scaler_ignores_missing_per_column():
    X = np.array([[2., np.nan], [4., 8.], [np.nan, 12.]])
    sc = StandardScaler().partial_fit(X[:1]).partial_fit(X[1:])
    np.testing.assert_array_equal(sc.n_samples_seen_, [2, 2])
    np.testing.assert_allclose(sc.mean_, [3, 10])
    np.testing.assert_allclose(sc.var_, [1, 4])
    assert np.isnan(sc.transform(X)[0, 1])


def test_scaler_all_nan_column_and_constant_feature():
    sc = StandardScaler().fit([[3., np.nan], [3., np.nan]])
    np.testing.assert_allclose(sc.scale_, [1., 1.])
    np.testing.assert_allclose(sc.transform([[3., 20.]])[0, 0], 0.)
    assert sc.n_samples_seen_[1] == 0


def test_scaler_huge_offset_stable():
    X = 1e9 + np.arange(20, dtype=float).reshape(-1, 1)
    sc = StandardScaler().partial_fit(X[:10]).partial_fit(X[10:])
    np.testing.assert_allclose(sc.mean_, X.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(sc.var_, X.var(axis=0), atol=1e-4)


def test_scaler_1d_is_one_feature_matrix():
    sc = StandardScaler().fit([1, 2, 3])
    assert sc.transform([4, 5]).shape == (2, 1)


def test_scaler_fit_resets_history():
    sc = StandardScaler().fit([[0.], [10.]])
    sc.fit([[20.], [20.]])
    np.testing.assert_array_equal(sc.mean_, [20.])


def test_scaler_transform_before_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        StandardScaler().transform([[1.]])


def test_scaler_feature_mismatch():
    with pytest.raises(ValueError, match="Expected 2 features"):
        StandardScaler().fit([[1., 2.]]).transform([[1.]])


@pytest.mark.parametrize("bad", [np.ones((2, 2, 2)), np.empty((0, 2)), np.array([[np.inf]])])
def test_scaler_rejects_bad_training_data(bad):
    with pytest.raises((ValueError, TypeError)):
        StandardScaler().partial_fit(bad)


def test_minmax_streaming_equals_batch(measurements):
    a = MinMaxScaler().fit(measurements)
    b = MinMaxScaler().partial_fit(measurements[:2]).partial_fit(measurements[2:])
    np.testing.assert_allclose(a.transform(measurements), b.transform(measurements))


def test_minmax_range_and_constant():
    sc = MinMaxScaler(feature_range=(-1, 1)).fit([[3., 2.], [3., 4.]])
    np.testing.assert_allclose(sc.transform([[3., 3.]]), [[-1., 0.]])


def test_minmax_nan_only_feature():
    sc = MinMaxScaler().fit([[np.nan, 1.], [np.nan, 2.]])
    assert np.isnan(sc.transform([[np.nan, 1.5]])[0, 0])
    np.testing.assert_allclose(sc.transform([[np.nan, 1.5]])[0, 1], 0.5)


def test_minmax_bad_range():
    with pytest.raises(ValueError, match="feature_range"):
        MinMaxScaler(feature_range=(1, 1))


def test_imputer_running_mean():
    im = Imputer(strategy="mean").partial_fit([[1., np.nan], [3., 9.]])
    im.partial_fit([[5., 15.], [np.nan, 21.]])
    np.testing.assert_allclose(im.statistics_, [3., 15.])
    np.testing.assert_allclose(im.transform([[np.nan, np.nan]]), [[3., 15.]])


def test_imputer_median_uses_previous_chunks():
    im = Imputer(strategy="median").partial_fit([[1.], [100.]])
    im.partial_fit([[2.]])
    np.testing.assert_allclose(im.statistics_, [2.])


def test_imputer_all_missing_defaults_to_fill():
    im = Imputer(strategy="mean", fill_value=-5).fit([[np.nan], [np.nan]])
    np.testing.assert_allclose(im.transform([[np.nan]]), [[-5.]])


def test_imputer_constant_without_valid_data():
    im = Imputer(strategy="constant", fill_value=17).partial_fit([[np.nan, 1.]])
    np.testing.assert_allclose(im.transform([[np.nan, np.nan]]), [[17., 17.]])


def test_imputer_fit_resets_statistics():
    im = Imputer().fit([[10.], [30.]])
    im.fit([[4.]])
    np.testing.assert_allclose(im.statistics_, [4.])


def test_imputer_invalid_strategy():
    with pytest.raises(ValueError, match="strategy"):
        Imputer(strategy="mode")


def test_imputer_mismatch_after_fit():
    with pytest.raises(ValueError, match="Expected"):
        Imputer().fit([[2., 3.]]).transform([[2.]])


def test_encoder_categories_expand_across_chunks():
    enc = OneHotEncoder().partial_fit(["blue", "red"])
    before = enc.transform(["blue", "red"])
    enc.partial_fit(["green", "blue"])
    after = enc.transform(["blue", "red", "green"])
    assert before.shape == (2, 2)
    assert after.shape == (3, 3)
    np.testing.assert_array_equal(after[:2, :2], before)
    assert np.all(after.sum(axis=1) == 1)


def test_encoder_multiple_columns():
    enc = OneHotEncoder().fit([["M", "S"], ["F", "S"], ["M", "C"]])
    out = enc.transform([["F", "C"], ["M", "S"]])
    assert out.shape == (2, 4)
    assert np.all(out.sum(axis=1) == 2)


def test_encoder_ignores_unknown_categories():
    enc = OneHotEncoder().fit(["a", "b"])
    np.testing.assert_array_equal(enc.transform(["z"]), np.zeros((1, 2)))


def test_encoder_rejects_unknown_categories_if_configured():
    enc = OneHotEncoder(handle_unknown="error").fit(["a"])
    with pytest.raises(ValueError, match="Unknown category"):
        enc.transform(["b"])


def test_encoder_reset_on_fit():
    enc = OneHotEncoder().partial_fit(["a", "b"])
    enc.fit(["c"])
    assert enc.transform(["c"]).shape == (1, 1)


def test_encoder_missing_category():
    enc = OneHotEncoder().fit([None, "a", np.nan])
    out = enc.transform([None, np.nan, "a"])
    np.testing.assert_array_equal(out[0], out[1])
    assert np.all(out.sum(axis=1) == 1)


def test_encoder_reserved_missing_token_is_invalid():
    with pytest.raises(ValueError, match="reserved"):
        OneHotEncoder().fit(["<MISSING>"])


def test_encoder_feature_count_mismatch():
    with pytest.raises(ValueError, match="Expected"):
        OneHotEncoder().fit([["a", "x"]]).transform([["b"]])


def test_encoder_empty_prediction_shape():
    enc = OneHotEncoder().fit(["a", "b"])
    assert enc.transform(np.empty(0, dtype=str)).shape == (0, 2)


def test_stats_incremental_equals_numpy(measurements):
    state = StreamingStats()
    state.update_stats(measurements[:2]).update_stats(measurements[2:])
    np.testing.assert_allclose(state.mean(), measurements.mean(axis=0))
    np.testing.assert_allclose(state.variance(), measurements.var(axis=0))
    np.testing.assert_allclose(state.std(), measurements.std(axis=0))
    assert state.n_chunks_seen_ == 2


def test_stats_nan_per_column():
    state = StreamingStats().update_stats([[1., np.nan], [3., 10.], [np.nan, 20.]])
    np.testing.assert_allclose(state.mean(), [2., 15.])
    np.testing.assert_allclose(state.variance(), [1., 25.])
    np.testing.assert_array_equal(state.count_, [2, 2])


def test_stats_unobserved_column_mean_variance_nan():
    state = StreamingStats().update_stats([[np.nan, 2.], [np.nan, 4.]])
    assert np.isnan(state.mean()[0])
    assert np.isnan(state.variance()[0])


def test_stats_variance_sample_ddof():
    st = StreamingStats().fit([[1.], [3.]])
    np.testing.assert_allclose(st.variance(ddof=1), [2.])


def test_stats_bad_ddof():
    with pytest.raises(ValueError, match="ddof"):
        StreamingStats().fit([1, 3]).variance(ddof=-1)


def test_stats_exact_quantiles_across_chunks():
    st = StreamingStats().partial_fit([1, 2]).partial_fit([3, 4])
    np.testing.assert_allclose(st.quantile([0.25, 0.5, 0.75]).ravel(), [1.75, 2.5, 3.25])


def test_stats_rolling_quantiles():
    st = StreamingStats(window_size=3).update_stats([1, 2]).update_stats([3, 4])
    np.testing.assert_allclose(st.quantile(0.5), [3.])
    np.testing.assert_allclose(st.mean(), [2.5])  # still cumulative


def test_stats_window_chunk_larger_than_window():
    st = StreamingStats(window_size=2).update_stats([1, 2, 3, 4, 5])
    np.testing.assert_allclose(st.quantile(0.5), [4.5])


def test_stats_quantile_invalid():
    st = StreamingStats().fit([1, 3])
    with pytest.raises(ValueError, match="quantiles"):
        st.quantile(2)


def test_stats_histogram_fixed_edges():
    st = StreamingStats().update_stats([0, 1, 2]).update_stats([3, 4])
    counts, edges = st.histogram(bins=5, value_range=(0, 5))
    np.testing.assert_array_equal(counts, [[1, 1, 1, 1, 1]])
    np.testing.assert_allclose(edges[0], [0, 1, 2, 3, 4, 5])


def test_stats_window_histogram():
    st = StreamingStats(window_size=2).update_stats([0, 1]).update_stats([2, 3])
    counts, _ = st.histogram(bins=4, value_range=(0, 4))
    np.testing.assert_array_equal(counts, [[0, 0, 1, 1]])


def test_stats_result_copies_arrays():
    st = StreamingStats().fit([[1.], [2.]])
    out = st.result()
    out["mean"][0] = 100
    np.testing.assert_allclose(st.mean(), [1.5])


def test_stats_invalid_window_size():
    with pytest.raises(ValueError, match="window_size"):
        StreamingStats(window_size=0)


def test_stats_transform_before_update():
    with pytest.raises(RuntimeError, match="not fitted"):
        StreamingStats().mean()


def test_io_read_titanic_csv_chunks():
    iterator = iter_supervised_csv(
        "demo/data/titanic.csv", target="survived", features=["pclass", "age", "fare"], chunksize=110
    )
    X, y = next(iterator)
    assert X.shape == (110, 3)
    assert y.shape == (110,)
    assert (y == "0").any()


def test_io_invalid_feature_name():
    with pytest.raises(ValueError, match="missing required"):
        next(iter_supervised_csv("demo/data/titanic.csv", target="survived", features=["not_a_col"]))


def test_io_rejects_target_as_feature():
    with pytest.raises(ValueError, match="exclude the target"):
        next(iter_supervised_csv("demo/data/titanic.csv", target="survived", features=["survived"]))
