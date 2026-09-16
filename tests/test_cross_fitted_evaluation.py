import numpy as np
import pytest

from counterfactuallab.data.synthetic import SyntheticConfig, generate_randomized_data
from counterfactuallab.evaluation.cross_fitting import cross_fitted_ranking_scores


def test_cross_fitted_ranking_scores_cover_every_row():
    data = generate_randomized_data(SyntheticConfig(n_samples=2_000, seed=101))
    response, cate = cross_fitted_ranking_scores(data, n_splits=5, seed=101)

    for scores in (response, cate):
        assert scores.shape == (len(data),)
        assert np.isfinite(scores).all()

    assert ((response >= 0.0) & (response <= 1.0)).all()
    assert np.std(cate) > 0.0


def test_cross_fitted_ranking_scores_are_reproducible():
    data = generate_randomized_data(SyntheticConfig(n_samples=2_000, seed=102))
    first = cross_fitted_ranking_scores(data, n_splits=4, seed=17)
    second = cross_fitted_ranking_scores(data, n_splits=4, seed=17)

    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])


@pytest.mark.parametrize("n_splits", [1, 101])
def test_cross_fitted_ranking_scores_reject_invalid_fold_count(n_splits):
    data = generate_randomized_data(SyntheticConfig(n_samples=100, seed=103))
    with pytest.raises(ValueError, match="n_splits"):
        cross_fitted_ranking_scores(data, n_splits=n_splits)
