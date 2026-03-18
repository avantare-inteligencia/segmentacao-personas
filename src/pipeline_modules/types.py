from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class PipelineArtifacts:
    df_eventos: pd.DataFrame
    user_base: pd.DataFrame
    matriz_categorias: pd.DataFrame
    user_full: pd.DataFrame
    persona_df: pd.DataFrame
    metricas_k: pd.DataFrame
    pca_df: pd.DataFrame
    estabilidade_df: pd.DataFrame
    estabilidade_clusters_df: pd.DataFrame
    top_telas_persona: pd.DataFrame
    principal_funcionalidade_df: pd.DataFrame
    sankey_df: pd.DataFrame
    features_modelo: list[str]
    cols_categoria: list[str]
    score_final: float
    variancia_pca: list[float]
    info_merge_notset: dict[str, Any]
