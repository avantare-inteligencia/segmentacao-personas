from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..config import DEFAULT_CONFIG, PersonaConfig
from .clustering import fundir_cluster_tecnico_notset


def rodar_estabilidade(
    user_full: pd.DataFrame,
    cols_categoria: list[str],
    features_modelo: list[str],
    config: PersonaConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = user_full[features_modelo].copy()
    scaler_estab = StandardScaler()
    X_scaled_estab = scaler_estab.fit_transform(X)
    resultados_estabilidade = []
    resumos_clusters = []
    cols_presentes = [c for c in cols_categoria if c in user_full.columns]

    for seed in config.stability_seeds:
        km = KMeans(n_clusters=config.n_clusters, random_state=seed, n_init=20)
        labels = km.fit_predict(X_scaled_estab)
        temp = user_full.copy()
        temp["cluster"] = labels
        sil = silhouette_score(X_scaled_estab, labels)
        temp, _ = fundir_cluster_tecnico_notset(temp, features_modelo, cols_categoria, config)
        cluster_final_col = "cluster_final" if "cluster_final" in temp.columns else "cluster"

        metric_cols = list(dict.fromkeys(features_modelo + cols_presentes))
        persona_tmp = (
            temp.groupby(cluster_final_col)[metric_cols]
            .mean()
            .reset_index()
            .rename(columns={cluster_final_col: "cluster"})
        )
        cluster_pct_tmp = (
            temp[cluster_final_col].value_counts(normalize=True).sort_index().mul(100).round(1).rename("% da base").reset_index()
        )
        cluster_pct_tmp.columns = ["cluster", "% da base"]
        persona_tmp = persona_tmp.merge(cluster_pct_tmp, on="cluster", how="left")
        persona_tmp["categoria_dominante"] = persona_tmp[cols_presentes].idxmax(axis=1)
        persona_tmp["share_dominante"] = persona_tmp[cols_presentes].max(axis=1)

        cols_sem_login = [c for c in cols_presentes if c != "Login"]
        if cols_sem_login:
            persona_tmp["categoria_dominante_sem_login"] = persona_tmp[cols_sem_login].idxmax(axis=1)
            persona_tmp["share_dominante_sem_login"] = persona_tmp[cols_sem_login].max(axis=1)
        else:
            persona_tmp["categoria_dominante_sem_login"] = "Sem categoria"
            persona_tmp["share_dominante_sem_login"] = 0.0

        resultados_estabilidade.append(
            {
                "seed": seed,
                "silhouette": round(float(sil), 4),
                "n_clusters_finais": int(temp[cluster_final_col].nunique()),
                "maior_cluster_%": round(float(cluster_pct_tmp["% da base"].max()), 1),
                "menor_cluster_%": round(float(cluster_pct_tmp["% da base"].min()), 1),
            }
        )

        resumo_cols = [
            "seed",
            "cluster",
            "% da base",
            "categoria_dominante",
            "share_dominante",
            "categoria_dominante_sem_login",
            "share_dominante_sem_login",
        ]
        for c in ["Login", "Apolices/Seguros", "Pagamento/Financeiro", "Home/Navegacao"]:
            if c in persona_tmp.columns:
                resumo_cols.append(c)
        persona_tmp["seed"] = seed
        resumos_clusters.append(persona_tmp[resumo_cols])

    return pd.DataFrame(resultados_estabilidade), pd.concat(resumos_clusters, ignore_index=True)
