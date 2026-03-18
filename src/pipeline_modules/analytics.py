from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA

from ..config import DEFAULT_CONFIG, PersonaConfig


def calcular_pca(X_scaled, user_full: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    pca = PCA(n_components=2, random_state=DEFAULT_CONFIG.random_state)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"pca_1": X_pca[:, 0], "pca_2": X_pca[:, 1], "cluster": user_full["cluster"].values})
    if "cluster_final" in user_full.columns:
        pca_df["cluster_final"] = user_full["cluster_final"].values
    if "persona" in user_full.columns:
        pca_df["persona"] = user_full["persona"].values
    return pca_df, list(pca.explained_variance_ratio_)


def calcular_principal_funcionalidade(user_full: pd.DataFrame, cols_categoria: list[str]) -> pd.DataFrame:
    perfil_persona = user_full.groupby("persona")[cols_categoria].mean()
    principal_func = perfil_persona.idxmax(axis=1)
    principal_valor = perfil_persona.max(axis=1)
    return (
        pd.DataFrame(
            {
                "persona": principal_func.index,
                "principal_funcionalidade": principal_func.values,
                "proporcao_media": principal_valor.values,
            }
        )
        .sort_values("proporcao_media", ascending=False)
        .reset_index(drop=True)
    )


def calcular_top_telas_persona(df: pd.DataFrame, user_full: pd.DataFrame, config: PersonaConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    telas_persona = df.merge(user_full[[config.col_user, "persona"]], on=config.col_user, how="inner")
    total_por_persona = telas_persona.groupby("persona")[config.col_events].sum()
    telas_cluster = (
        telas_persona.groupby(["persona", config.col_screen])[config.col_events].sum().reset_index(name="eventos")
    )
    telas_cluster["% dentro da persona"] = telas_cluster.apply(
        lambda row: round(100 * row["eventos"] / total_por_persona[row["persona"]], 2), axis=1
    )
    return (
        telas_cluster.sort_values(["persona", "% dentro da persona"], ascending=[True, False])
        .groupby("persona")
        .head(10)
        .reset_index(drop=True)
    )

def construir_fluxos_sankey(persona_df: pd.DataFrame, cols_categoria: list[str], total_usuarios: int) -> pd.DataFrame:
    cols_categoria = [c for c in cols_categoria if c.lower() != "login"]
    persona_base = persona_df[["cluster", "persona", "% da base"]].copy()
    persona_base["qtd_usuarios"] = (persona_base["% da base"] / 100 * total_usuarios).round().astype(int)

    fluxos = []
    for _, row in persona_base.iterrows():
        cluster_id = row["cluster"]
        persona_nome = row["persona"]
        qtd_usuarios = row["qtd_usuarios"]
        linha_cluster = persona_df.loc[persona_df["cluster"] == cluster_id].iloc[0]

        for cat in cols_categoria:
            share_cat = float(linha_cluster.get(cat, 0.0))
            valor_fluxo = qtd_usuarios * share_cat
            if valor_fluxo > 0:
                fluxos.append({
                    "source": persona_nome,
                    "target": cat,
                    "value": valor_fluxo
                })

    return pd.DataFrame(fluxos)