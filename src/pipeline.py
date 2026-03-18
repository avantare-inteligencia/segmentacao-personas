from __future__ import annotations

from .pipeline_modules.analytics import (
    calcular_pca,
    calcular_principal_funcionalidade,
    calcular_top_telas_persona,
    construir_fluxos_sankey,
)
from .pipeline_modules.categorization import aplicar_categorizacao, categorizar_tela
from .pipeline_modules.clustering import avaliar_k, fundir_cluster_tecnico_notset, treinar_clusterizacao
from .pipeline_modules.features import construir_matriz_categorias, construir_user_base, get_feature_lists, montar_base_modelagem
from .pipeline_modules.io_utils import export_outputs, load_raw_data, validar_colunas
from .pipeline_modules.orchestrator import run_pipeline
from .pipeline_modules.personas import atribuir_personas, construir_persona_df, descrever_persona
from .pipeline_modules.preprocessing import limpar_dataset
from .pipeline_modules.stability import rodar_estabilidade
from .pipeline_modules.types import PipelineArtifacts

__all__ = [
    "PipelineArtifacts",
    "validar_colunas",
    "load_raw_data",
    "limpar_dataset",
    "categorizar_tela",
    "aplicar_categorizacao",
    "construir_user_base",
    "construir_matriz_categorias",
    "montar_base_modelagem",
    "get_feature_lists",
    "avaliar_k",
    "treinar_clusterizacao",
    "fundir_cluster_tecnico_notset",
    "construir_persona_df",
    "descrever_persona",
    "atribuir_personas",
    "calcular_pca",
    "calcular_principal_funcionalidade",
    "calcular_top_telas_persona",
    "construir_fluxos_sankey",
    "rodar_estabilidade",
    "run_pipeline",
    "export_outputs",
]
