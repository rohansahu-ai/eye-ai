"""Tests for modules/similarity_index.py"""
import numpy as np
import pytest
from modules.similarity_index import SimilarityIndex


def test_feature_matrix_shape(products_df):
    idx = SimilarityIndex()
    matrix = idx.build_features(products_df)
    assert matrix.shape[0] == len(products_df)
    assert matrix.shape[1] > 0
    # All values should be in [0, 1] after MinMaxScaler
    assert matrix.min() >= -1e-6
    assert matrix.max() <= 1.0 + 1e-6


def test_similarity_matrix_diagonal_is_one(products_df):
    idx = SimilarityIndex().fit(products_df)
    sim = idx.similarity_matrix
    diag = np.diag(sim.values)
    np.testing.assert_allclose(diag, 1.0, atol=1e-6)


def test_similar_skus_excludes_self(products_df):
    idx = SimilarityIndex().fit(products_df)
    sku = products_df.iloc[0]["sku_id"]
    result = idx.get_similar_skus(sku, top_n=5)
    assert sku not in result["similar_sku_id"].values
    assert len(result) <= 5
    assert len(result) > 0


def test_similarity_scores_between_0_and_1(products_df):
    idx = SimilarityIndex().fit(products_df)
    sku = products_df.iloc[0]["sku_id"]
    result = idx.get_similar_skus(sku, top_n=5)
    assert (result["similarity_score"] >= 0).all()
    assert (result["similarity_score"] <= 1.0 + 1e-6).all()


def test_cluster_count_respected(products_df):
    idx = SimilarityIndex().fit(products_df)
    df = idx.products_df
    assert "cluster_id" in df.columns
    assert "cluster_label" in df.columns
    n_clusters = df["cluster_id"].nunique()
    assert 1 <= n_clusters <= 15  # Allow KMeans to vary slightly


def test_spillover_excludes_low_stock(products_df, inventory_df):
    idx = SimilarityIndex().fit(products_df)
    sku = products_df.iloc[0]["sku_id"]
    candidates = idx.demand_spillover_candidates(sku, inventory_df, min_similarity=0.0)
    # All candidates must have days_of_supply > 30
    if len(candidates) > 0:
        assert (candidates["days_of_supply"] > 30).all()


def test_fit_returns_self(products_df):
    idx = SimilarityIndex()
    result = idx.fit(products_df)
    assert result is idx
    assert idx.similarity_matrix is not None
    assert idx.feature_matrix is not None


def test_save_and_load(products_df, tmp_path):
    idx = SimilarityIndex().fit(products_df)
    path = str(tmp_path / "test_similarity.pkl")
    idx.save(path)
    loaded = SimilarityIndex.load(path)
    assert loaded.similarity_matrix is not None
    assert len(loaded.sku_ids) == len(products_df)
