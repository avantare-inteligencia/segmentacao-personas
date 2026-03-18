from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DEFAULT_CONFIG, PersonaConfig


def construir_user_base(df: pd.DataFrame, config: PersonaConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    hoje = df[config.col_datetime].max()
    user_base = (
        df.groupby(config.col_user)
        .agg(
            num_sessoes=(config.col_session, "nunique"),
            eventos_total=(config.col_events, "sum"),
            primeira_sessao=(config.col_datetime, "min"),
            ultima_sessao=(config.col_datetime, "max"),
            dias_ativos=("data_dia", "nunique"),
        )
        .reset_index()
    )
    user_base["janela_atividade"] = (user_base["ultima_sessao"] - user_base["primeira_sessao"]).dt.days + 1
    user_base["recencia"] = (hoje - user_base["ultima_sessao"]).dt.days
    return user_base


def construir_matriz_categorias(df: pd.DataFrame, config: PersonaConfig = DEFAULT_CONFIG) -> tuple[pd.DataFrame, list[str]]:
    matriz = (
        df.groupby([config.col_user, "categoria_tela"])[config.col_events]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    cols_categoria = [c for c in matriz.columns if c != config.col_user]
    soma_linha = matriz[cols_categoria].sum(axis=1).replace(0, 1)
    matriz[cols_categoria] = matriz[cols_categoria].div(soma_linha, axis=0)
    return matriz, cols_categoria


def montar_base_modelagem(
    user_base: pd.DataFrame,
    matriz_telas: pd.DataFrame,
    cols_categoria: list[str],
    config: PersonaConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    user_full = user_base.merge(matriz_telas, on=config.col_user, how="left").fillna(0).copy()

    user_full["num_sessoes_log"] = np.log1p(user_full["num_sessoes"])
    user_full["eventos_total_log"] = np.log1p(user_full["eventos_total"])
    user_full["dias_ativos_log"] = np.log1p(user_full["dias_ativos"])
    user_full["share_categoria_max"] = user_full[cols_categoria].max(axis=1)

    user_full["share_login"] = user_full["Login"] if "Login" in user_full.columns else 0.0
    user_full["share_outros"] = user_full["Outros"] if "Outros" in user_full.columns else 0.0
    user_full["share_notset"] = user_full["NotSet"] if "NotSet" in user_full.columns else 0.0
    user_full["share_nao_login"] = 1 - user_full["share_login"]

    ordered_shares = np.sort(user_full[cols_categoria].to_numpy(dtype=float), axis=1)
    if ordered_shares.shape[1] >= 2:
        user_full["share_top2"] = ordered_shares[:, -2:].sum(axis=1)
    else:
        user_full["share_top2"] = ordered_shares[:, -1]

    user_full["n_categorias_ativas"] = (user_full[cols_categoria] >= config.min_share_categoria_ativa).sum(axis=1)

    probs = user_full[cols_categoria].to_numpy(dtype=float)
    probs_safe = np.where(probs > 0, probs, 1e-12)
    entropia = -(probs * np.log(probs_safe)).sum(axis=1)
    if len(cols_categoria) > 1:
        entropia = entropia / np.log(len(cols_categoria))
    user_full["entropia_categorias"] = entropia

    cols_sem_login = [c for c in cols_categoria if c != "Login"]
    cols_sem_login_sem_outros = [c for c in cols_categoria if c not in ["Login", "Outros"]]

    if cols_sem_login:
        user_full["share_dominante_sem_login"] = user_full[cols_sem_login].max(axis=1)
        user_full["categoria_dominante_sem_login"] = user_full[cols_sem_login].idxmax(axis=1)
    else:
        user_full["share_dominante_sem_login"] = 0.0
        user_full["categoria_dominante_sem_login"] = "Sem categoria"

    if cols_sem_login_sem_outros:
        user_full["share_dominante_sem_login_sem_outros"] = user_full[cols_sem_login_sem_outros].max(axis=1)
        user_full["categoria_dominante_sem_login_sem_outros"] = user_full[cols_sem_login_sem_outros].idxmax(axis=1)
    else:
        user_full["share_dominante_sem_login_sem_outros"] = 0.0
        user_full["categoria_dominante_sem_login_sem_outros"] = "Sem categoria"

    categorias_transacionais = [c for c in config.categories_transacionais if c in user_full.columns]
    categorias_navegacionais = [c for c in config.categories_navegacionais if c in user_full.columns]

    user_full["soma_transacional"] = user_full[categorias_transacionais].sum(axis=1) if categorias_transacionais else 0.0
    user_full["soma_navegacional"] = user_full[categorias_navegacionais].sum(axis=1) if categorias_navegacionais else 0.0
    user_full["login_puro_score"] = user_full["share_login"] * (1 - user_full["share_dominante_sem_login"])
    user_full["login_com_profundidade_score"] = user_full["share_login"] * user_full["share_nao_login"]
    return user_full


def get_feature_lists(cols_categoria: list[str]) -> tuple[list[str], list[str]]:
    features_comportamentais = [
        "num_sessoes_log",
        "eventos_total_log",
        "dias_ativos_log",
        "recencia",
        "share_categoria_max",
        "share_top2",
        "n_categorias_ativas",
        "entropia_categorias",
        "share_dominante_sem_login",
        "share_dominante_sem_login_sem_outros",
        "soma_transacional",
        "soma_navegacional",
        "login_puro_score",
        "login_com_profundidade_score",
    ]
    cols_categoria_modelo = [c for c in cols_categoria if c != "NotSet"]
    return features_comportamentais, features_comportamentais + cols_categoria_modelo
