"""Regression checks for missing categories in incremental one-hot encoding."""

import numpy as np
import pytest

from numcompute_stream.preprocessing import OneHotEncoder


def test_missing_marker_not_truncated_by_short_categories():
    encoder = OneHotEncoder().partial_fit(np.array([[None], ["x"]], dtype=object))
    assert "<MISSING>" in encoder.categories_[0]
    assert "<MIS" not in encoder.categories_[0]
    np.testing.assert_array_equal(
        encoder.transform(np.array([[None]], dtype=object)),
        encoder.transform(np.array([[np.nan]], dtype=object)),
    )


def test_missing_category_is_not_duplicated_across_streaming_chunks():
    encoder = OneHotEncoder()
    encoder.partial_fit(np.array([[None], ["x"]], dtype=object))
    width_after_first = encoder.n_features_out_
    encoder.partial_fit(np.array([[np.nan], ["abcdefg"]], dtype=object))
    assert encoder.n_features_out_ == width_after_first + 1
    assert np.count_nonzero(encoder.categories_[0] == "<MISSING>") == 1
    np.testing.assert_array_equal(
        encoder.transform(np.array([[None]], dtype=object)),
        encoder.transform(np.array([[np.nan]], dtype=object)),
    )


def test_missing_tokens_from_different_columns_remain_consistent():
    encoder = OneHotEncoder()
    encoder.partial_fit(np.array([[None, "a"], ["b", np.nan]], dtype=object))
    encoder.partial_fit(np.array([[np.nan, "long-category"], ["b", None]], dtype=object))
    for categories in encoder.categories_:
        assert np.count_nonzero(categories == "<MISSING>") == 1


def test_literal_missing_sentinel_is_still_rejected():
    with pytest.raises(ValueError, match="reserved"):
        OneHotEncoder().partial_fit(np.array(["<MISSING>", "x"], dtype=object))
