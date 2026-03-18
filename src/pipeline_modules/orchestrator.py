from __future__ import annotations

from pathlib import Path

from ..config import DEFAULT_CONFIG, PersonaConfig
from .analytics import (
    calcular_pca,
    calcular_principal_funcionalidade,
    calcular_top_telas_persona,
    construir_fluxos_sankey,
)
from .categorization import aplicar_categorizacao
from .clustering import avaliar_k, fundir_cluster_tecnico_notset, treinar_clusterizacao
from .features import construir_matriz_categorias, construir_user_base, get_feature_lists, montar_base_modelagem
from .io_utils import load_raw_data
from .personas import atribuir_personas, construir_persona_df
from .preprocessing import limpar_dataset
from .stability import rodar_estabilidade
from .types import PipelineArtifacts


def run_pipeline(data_path: str | Path, config: PersonaConfig = DEFAULT_CONFIG) -> PipelineArtifacts:
    df_raw = load_raw_data(data_path, config)
    df_limpo = limpar_dataset(df_raw, config)
    df_eventos = aplicar_categorizacao(df_limpo, config)

    user_base = construir_user_base(df_eventos, config)
    matriz_categorias, cols_categoria = construir_matriz_categorias(df_eventos, config)
    user_full = montar_base_modelagem(user_base, matriz_categorias, cols_categoria, config)

    _, features_modelo = get_feature_lists(cols_categoria)
    X = user_full[features_modelo].copy()
    metricas_k, X_scaled, _ = avaliar_k(X, config)
    user_full, score_final, _ = treinar_clusterizacao(user_full, X_scaled, config.n_clusters, config)
    user_full, info_merge_notset = fundir_cluster_tecnico_notset(user_full, features_modelo, cols_categoria, config)

    persona_df = construir_persona_df(user_full, features_modelo, cols_categoria)
    persona_df, user_full = atribuir_personas(persona_df, user_full)

    pca_df, variancia_pca = calcular_pca(X_scaled, user_full)
    principal_funcionalidade_df = calcular_principal_funcionalidade(user_full, cols_categoria)
    top_telas_persona = calcular_top_telas_persona(df_eventos, user_full, config)
    sankey_df = construir_fluxos_sankey(persona_df, cols_categoria, total_usuarios=len(user_full))
    estabilidade_df, estabilidade_clusters_df = rodar_estabilidade(user_full, cols_categoria, features_modelo, config)

    return PipelineArtifacts(
        df_eventos=df_eventos,
        user_base=user_base,
        matriz_categorias=matriz_categorias,
        user_full=user_full,
        persona_df=persona_df,
        metricas_k=metricas_k,
        pca_df=pca_df,
        estabilidade_df=estabilidade_df,
        estabilidade_clusters_df=estabilidade_clusters_df,
        top_telas_persona=top_telas_persona,
        principal_funcionalidade_df=principal_funcionalidade_df,
        sankey_df=sankey_df,
        features_modelo=features_modelo,
        cols_categoria=cols_categoria,
        score_final=score_final,
        variancia_pca=variancia_pca,
        info_merge_notset=info_merge_notset,
    )
