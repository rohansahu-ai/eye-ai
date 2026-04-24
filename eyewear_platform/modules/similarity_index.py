"""
Product similarity index using multi-modal feature fusion.
Combines categorical attributes, price positioning, and visual metadata.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from config.settings import Settings, get_settings
from utils.logger import get_logger


class SimilarityIndex:
    """
    Computes product similarity using multi-modal feature fusion.
    Combines categorical attributes, price positioning, and visual metadata.
    """

    PRICE_POINT_ORD = {"budget": 1, "mid": 2, "premium": 3, "luxury": 4}

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or get_settings()
        self.feature_matrix: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[pd.DataFrame] = None
        self.kmeans_model: Optional[KMeans] = None
        self.sku_ids: Optional[list] = None
        self.products_df: Optional[pd.DataFrame] = None
        self.logger = get_logger(__name__)
        self._label_encoders: dict = {}
        self._scaler = MinMaxScaler()

    def build_features(self, products_df: pd.DataFrame) -> np.ndarray:
        """
        One-hot encode categoricals, ordinal-encode price_point,
        normalize all features. Returns NxF feature matrix.
        """
        df = products_df.copy()

        # Ordinal: price_point
        df["price_point_enc"] = (
            df["price_point"].map(self.PRICE_POINT_ORD).fillna(2).astype(float)
        )

        # One-hot categoricals
        cat_cols = ["frame_shape", "material", "color", "gender", "category"]
        encoded_parts = [df[["price_point_enc"]]]
        for col in cat_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
                encoded_parts.append(dummies)

        feature_df = pd.concat(encoded_parts, axis=1)
        matrix = self._scaler.fit_transform(feature_df.values)
        return matrix

    def compute_similarity(
        self, feature_matrix: np.ndarray, sku_ids: list
    ) -> pd.DataFrame:
        """
        Cosine similarity matrix. Returns NxN DataFrame indexed by sku_id.
        Self-similarity on diagonal = 1.0.
        """
        sim = cosine_similarity(feature_matrix)
        np.fill_diagonal(sim, 1.0)
        return pd.DataFrame(sim, index=sku_ids, columns=sku_ids)

    def get_similar_skus(
        self, sku_id: str, top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Returns top_n most similar SKUs with columns:
        [sku_id, similar_sku_id, similarity_score, frame_shape, material, color, price_point]
        Excludes the query SKU itself.
        """
        if self.similarity_matrix is None:
            raise RuntimeError("Call fit() first.")
        if sku_id not in self.similarity_matrix.index:
            return pd.DataFrame()

        n = top_n or self.config.SIMILARITY_TOP_N
        row = self.similarity_matrix.loc[sku_id].drop(index=sku_id, errors="ignore")
        top = row.nlargest(n).reset_index()
        top.columns = ["similar_sku_id", "similarity_score"]
        top["sku_id"] = sku_id

        if self.products_df is not None:
            meta = self.products_df[["sku_id", "frame_shape", "material", "color", "price_point"]]
            top = top.merge(meta, left_on="similar_sku_id", right_on="sku_id", how="left", suffixes=("", "_y"))
            top = top.drop(columns=["sku_id_y"], errors="ignore")

        cols = ["sku_id", "similar_sku_id", "similarity_score", "frame_shape", "material", "color", "price_point"]
        return top[[c for c in cols if c in top.columns]].reset_index(drop=True)

    def cluster_products(self, n_clusters: int = 10) -> pd.DataFrame:
        """
        KMeans clustering. Returns products_df with cluster_id and cluster_label columns.
        """
        if self.feature_matrix is None or self.products_df is None:
            raise RuntimeError("Call fit() first.")

        # Elbow method validation: try k=5..15 and find the best within 1.2x the requested n
        best_k = n_clusters
        self.kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = self.kmeans_model.fit_predict(self.feature_matrix)

        df = self.products_df.copy()
        df["cluster_id"] = labels

        # Human-readable cluster labels
        def _make_label(cluster_df: pd.DataFrame) -> str:
            pp = cluster_df["price_point"].mode().iloc[0] if len(cluster_df) > 0 else "mid"
            mat = cluster_df["material"].mode().iloc[0] if "material" in cluster_df else "acetate"
            fs = cluster_df["frame_shape"].mode().iloc[0] if "frame_shape" in cluster_df else "round"
            return f"{pp.title()} {mat.title()} {fs.replace('_', ' ').title()}"

        label_map = {
            cid: _make_label(df[df["cluster_id"] == cid])
            for cid in range(best_k)
        }
        df["cluster_label"] = df["cluster_id"].map(label_map)
        self.products_df = df
        return df

    def demand_spillover_candidates(
        self,
        sku_id: str,
        inventory_df: pd.DataFrame,
        min_similarity: float = 0.70,
    ) -> pd.DataFrame:
        """
        For a given (potentially stocked-out) SKU, find similar SKUs
        that have sufficient stock (days_of_supply > 30).
        """
        if self.similarity_matrix is None:
            raise RuntimeError("Call fit() first.")
        if sku_id not in self.similarity_matrix.index:
            return pd.DataFrame()

        row = self.similarity_matrix.loc[sku_id].drop(index=sku_id, errors="ignore")
        candidates = row[row >= min_similarity].sort_values(ascending=False).reset_index()
        candidates.columns = ["similar_sku_id", "similarity_score"]

        stocked = inventory_df[inventory_df["days_of_supply"] > 30][
            ["sku_id", "store_id", "days_of_supply", "quantity_on_hand"]
        ]
        result = candidates.merge(
            stocked, left_on="similar_sku_id", right_on="sku_id", how="inner"
        )
        result["query_sku_id"] = sku_id
        return result.sort_values("similarity_score", ascending=False).reset_index(drop=True)

    def fit(self, products_df: pd.DataFrame) -> "SimilarityIndex":
        """Chain: build_features → compute_similarity → cluster_products."""
        self.products_df = products_df.copy()
        self.sku_ids = products_df["sku_id"].tolist()
        self.feature_matrix = self.build_features(products_df)
        self.similarity_matrix = self.compute_similarity(self.feature_matrix, self.sku_ids)
        self.cluster_products()
        self.logger.info("SimilarityIndex fitted", n_skus=len(self.sku_ids))
        return self

    def save(self, path: str) -> None:
        """Pickle the fitted index to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self.logger.info("SimilarityIndex saved", path=path)

    @classmethod
    def load(cls, path: str) -> "SimilarityIndex":
        """Load from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.generate_synthetic_data import generate_products
    from config.settings import get_settings

    cfg = get_settings()
    products = generate_products(100)
    idx = SimilarityIndex(cfg).fit(products)

    similar = idx.get_similar_skus("SKU0001", top_n=5)
    print("Top 5 similar to SKU0001:")
    print(similar.to_string())

    clusters = idx.products_df[["sku_id", "cluster_id", "cluster_label"]].head(10)
    print("\nCluster sample:")
    print(clusters.to_string())
