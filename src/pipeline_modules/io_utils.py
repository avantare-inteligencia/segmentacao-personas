from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import DEFAULT_CONFIG, PersonaConfig, ensure_output_dir
from .types import PipelineArtifacts


def validar_colunas(df: pd.DataFrame, colunas_esperadas: list[str]) -> None:
    faltantes = [c for c in colunas_esperadas if c not in df.columns]
    if faltantes:
        raise ValueError(f"As seguintes colunas estão ausentes no dataset: {faltantes}")


def load_raw_data(path: str | Path, config: PersonaConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    df_raw = pd.read_csv(path, sep=";")
    df_raw.columns = df_raw.columns.str.strip()
    validar_colunas(
        df_raw,
        [config.col_user, config.col_datetime, config.col_events, config.col_screen, config.col_session],
    )
    return df_raw


def export_outputs(artifacts: PipelineArtifacts, base_dir: str | Path, config: PersonaConfig = DEFAULT_CONFIG) -> Path:
    output_dir = ensure_output_dir(base_dir, config)
    artifacts.user_full.to_csv(output_dir / "usuarios_personas.csv", index=False, sep=";", encoding="utf-8-sig")
    artifacts.persona_df.to_csv(output_dir / "personas_resumo.csv", index=False, sep=";", encoding="utf-8-sig")
    artifacts.metricas_k.to_csv(output_dir / "metricas_k.csv", index=False, sep=";", encoding="utf-8-sig")
    artifacts.estabilidade_df.to_csv(output_dir / "estabilidade.csv", index=False, sep=";", encoding="utf-8-sig")
    artifacts.top_telas_persona.to_csv(output_dir / "top_telas_persona.csv", index=False, sep=";", encoding="utf-8-sig")
    return output_dir
