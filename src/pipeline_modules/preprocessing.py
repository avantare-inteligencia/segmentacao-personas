from __future__ import annotations

import pandas as pd

from ..config import DEFAULT_CONFIG, PersonaConfig


def limpar_dataset(df: pd.DataFrame, config: PersonaConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=[config.col_user, config.col_datetime]).copy()

    df[config.col_user] = df[config.col_user].astype("string").str.strip()
    df[config.col_datetime] = pd.to_datetime(
        df[config.col_datetime].astype(str).str.strip(),
        format="%Y%m%d%H",
        errors="coerce",
    )
    df[config.col_events] = pd.to_numeric(df[config.col_events], errors="coerce").fillna(0)

    df = df.dropna(subset=[config.col_user, config.col_datetime]).copy()
    df = df[df[config.col_user].notna()].copy()
    df = df[df[config.col_user].astype(str).str.strip().ne("")]
    df = df[df[config.col_user].astype(str).str.lower().ne("nan")]

    df["data_dia"] = df[config.col_datetime].dt.normalize()
    df[config.col_screen] = df[config.col_screen].fillna("Tela ausente").astype(str).str.strip()
    return df.reset_index(drop=True)
