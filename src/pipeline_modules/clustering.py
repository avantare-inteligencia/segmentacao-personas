from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..config import DEFAULT_CONFIG, PersonaConfig


def avaliar_k(X: pd.DataFrame, config: PersonaConfig = DEFAULT_CONFIG) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    metricas_k = []
    for k in config.k_range:
        model = KMeans(n_clusters=k, random_state=config.random_state, n_init=10)
        labels = model.fit_predict(X_scaled)
        metricas_k.append(
            {
                "k": k,
                "inercia": model.inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
            }
        )
    return pd.DataFrame(metricas_k), X_scaled, scaler


def treinar_clusterizacao(
    user_full: pd.DataFrame,
    X_scaled: np.ndarray,
    n_clusters: int,
    config: PersonaConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, float, KMeans]:
    user_full = user_full.copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.random_state, n_init=10)
    user_full["cluster"] = kmeans.fit_predict(X_scaled)
    score_final = silhouette_score(X_scaled, user_full["cluster"])
    return user_full, score_final, kmeans


def fundir_cluster_tecnico_notset(
    user_full: pd.DataFrame,
    features_modelo: list[str],
    cols_categoria: list[str],
    config: PersonaConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    user_full = user_full.copy()
    info_merge = {
        "merge_aplicado": False,
        "cluster_notset": None,
        "media_notset": None,
        "cluster_destino": None,
        "mapa_reindex": None,
        "distancias": None,
    }
    cols_presentes = [c for c in cols_categoria if c in user_full.columns]

    if "NotSet" not in cols_presentes:
        user_full["cluster_final"] = user_full["cluster"]
        return user_full, info_merge

    profile_tmp = user_full.groupby("cluster")[cols_presentes].mean().reset_index()
    cluster_notset = int(profile_tmp.loc[profile_tmp["NotSet"].idxmax(), "cluster"])
    media_notset = float(profile_tmp.loc[profile_tmp["cluster"] == cluster_notset, "NotSet"].iloc[0])
    info_merge["cluster_notset"] = cluster_notset
    info_merge["media_notset"] = media_notset

    if media_notset < config.notset_merge_threshold:
        user_full["cluster_final"] = user_full["cluster"]
        return user_full, info_merge

    scaler_merge = StandardScaler()
    X_scaled_merge = scaler_merge.fit_transform(user_full[features_modelo])
    X_scaled_merge_df = pd.DataFrame(X_scaled_merge, columns=features_modelo, index=user_full.index)
    X_scaled_merge_df["cluster"] = user_full["cluster"].values
    centroides = X_scaled_merge_df.groupby("cluster")[features_modelo].mean()
    centroide_notset = centroides.loc[cluster_notset]
    clusters_negocio = [c for c in centroides.index if c != cluster_notset]

    distancias = {int(c): float(np.linalg.norm(centroide_notset.values - centroides.loc[c].values)) for c in clusters_negocio}
    cluster_destino = min(distancias, key=distancias.get)

    user_full["cluster_final"] = user_full["cluster"]
    user_full.loc[user_full["cluster"] == cluster_notset, "cluster_final"] = cluster_destino
    clusters_ordenados = sorted(user_full["cluster_final"].unique())
    mapa_reindex = {int(old): new for new, old in enumerate(clusters_ordenados)}
    user_full["cluster_final"] = user_full["cluster_final"].map(mapa_reindex)

    info_merge.update(
        {
            "merge_aplicado": True,
            "cluster_destino": int(cluster_destino),
            "mapa_reindex": mapa_reindex,
            "distancias": distancias,
        }
    )
    return user_full, info_merge
