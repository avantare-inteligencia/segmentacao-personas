from __future__ import annotations

import pandas as pd


def construir_persona_df(
    user_full: pd.DataFrame,
    features_modelo: list[str],
    cols_categoria: list[str],
) -> pd.DataFrame:
    col_cluster_analise = "cluster_final" if "cluster_final" in user_full.columns else "cluster"
    cols_presentes = [c for c in cols_categoria if c in user_full.columns]
    colunas_aux = [c for c in ["share_login", "share_nao_login", "share_outros"] if c in user_full.columns]
    colunas_analise = list(dict.fromkeys(features_modelo + cols_presentes + colunas_aux))

    persona_df = (
        user_full.groupby(col_cluster_analise)[colunas_analise].mean().reset_index().rename(columns={col_cluster_analise: "cluster"})
    )
    cluster_pct = (
        user_full[col_cluster_analise].value_counts(normalize=True).sort_index().mul(100).round(1).rename("% da base").reset_index()
    )
    cluster_pct.columns = ["cluster", "% da base"]
    persona_df = persona_df.merge(cluster_pct, on="cluster", how="left")

    persona_df["categoria_dominante"] = persona_df[cols_presentes].idxmax(axis=1)
    persona_df["share_dominante"] = persona_df[cols_presentes].max(axis=1).round(3)

    cols_sem_login = [c for c in cols_presentes if c != "Login"]
    if cols_sem_login:
        persona_df["categoria_dominante_sem_login"] = persona_df[cols_sem_login].idxmax(axis=1)
        persona_df["share_dominante_sem_login"] = persona_df[cols_sem_login].max(axis=1).round(3)
    else:
        persona_df["categoria_dominante_sem_login"] = "Sem categoria"
        persona_df["share_dominante_sem_login"] = 0.0

    cols_sem_login_sem_outros = [c for c in cols_presentes if c not in ["Login", "Outros", "NotSet"]]
    if cols_sem_login_sem_outros:
        persona_df["categoria_dominante_sem_login_sem_outros"] = persona_df[cols_sem_login_sem_outros].idxmax(axis=1)
        persona_df["share_dominante_sem_login_sem_outros"] = persona_df[cols_sem_login_sem_outros].max(axis=1).round(3)
    else:
        persona_df["categoria_dominante_sem_login_sem_outros"] = "Sem categoria"
        persona_df["share_dominante_sem_login_sem_outros"] = 0.0

    persona_df["score_engajamento"] = (
        0.30 * persona_df["num_sessoes_log"]
        + 0.30 * persona_df["eventos_total_log"]
        + 0.15 * persona_df["dias_ativos_log"]
        - 0.10 * persona_df["recencia"]
        + 0.10 * persona_df["entropia_categorias"]
        + 0.05 * persona_df["soma_transacional"]
    )
    return persona_df


def descrever_persona(row: pd.Series) -> tuple[str, str]:
    categoria = row["categoria_dominante"]
    categoria_sem_login = row["categoria_dominante_sem_login"]
    categoria_sem_login_sem_outros = row["categoria_dominante_sem_login_sem_outros"]
    share_login = row["share_login"]
    share_sem_login_sem_outros = row["share_dominante_sem_login_sem_outros"]
    pct = row["% da base"]
    entropia = row["entropia_categorias"]
    n_cat = row["n_categorias_ativas"]
    soma_transacional = row["soma_transacional"]
    score = row["score_engajamento"]

    if share_login >= 0.60 and entropia <= 0.35 and n_cat <= 2.5:
        return "Acesso Concentrado", "Usuários com forte concentração em login, com pouca profundidade nas demais funcionalidades."
    if share_login >= 0.20 and soma_transacional >= 0.30:
        if categoria_sem_login_sem_outros == "Pagamento/Financeiro":
            return "Login + Financeiro", "Usuários que passam por login, mas avançam principalmente para pagamentos, cobrança e fluxos financeiros."
        if categoria_sem_login_sem_outros == "Apolices/Seguros":
            return "Login + Seguros", "Usuários que passam por login, mas aprofundam o uso em apólices, seguros e serviços associados."
        if categoria_sem_login_sem_outros == "Extrato/Consulta":
            return "Login + Consulta", "Usuários que passam por login e avançam para consulta de extrato e acompanhamento de informações."
        return "Login + Autosserviço", "Usuários que passam por login e avançam para funcionalidades transacionais do app."
    if share_login >= 0.20 and entropia >= 0.50 and n_cat >= 4:
        if score >= 0:
            return "Multifuncional Engajado", "Usuários com login relevante e navegação distribuída entre múltiplas funcionalidades, com maior engajamento."
        return "Base Ampla de Acesso", "Maior grupo da base, com login relevante e navegação mais ampla entre diferentes áreas do app."
    if share_sem_login_sem_outros >= 0.22:
        if categoria_sem_login_sem_outros == "Pagamento/Financeiro":
            return "Engajado Financeiro", "Usuários com navegação mais distribuída, mas com principal aprofundamento em jornadas financeiras."
        if categoria_sem_login_sem_outros == "Apolices/Seguros":
            return "Engajado em Seguros", "Usuários com navegação mais distribuída, mas com principal aprofundamento em apólices e seguros."
        if categoria_sem_login_sem_outros == "Home/Navegacao":
            return "Explorador", "Usuários que usam login e circulam pelo app com foco maior em navegação e descoberta."
        return f"Multiuso ({categoria_sem_login_sem_outros})", "Usuários com navegação relativamente distribuída, com uma funcionalidade secundária clara após o login."
    if pct >= 40:
        return "Base Ampla de Baixa Intensidade", "Maior grupo da base, com comportamento amplo, menor intensidade média e sem especialização funcional muito forte."
    if categoria == "Login":
        return "Recorrente de Acesso", "Usuários relativamente frequentes, com ênfase principal em acesso, mas sem aprofundamento funcional muito forte."
    return f"Multiuso ({categoria_sem_login})", "Usuários com navegação distribuída e alguma funcionalidade complementar relevante após o login."


def atribuir_personas(persona_df: pd.DataFrame, user_full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    persona_df = persona_df.copy()
    user_full = user_full.copy()
    persona_df[["persona", "descricao"]] = persona_df.apply(lambda row: pd.Series(descrever_persona(row)), axis=1)

    cluster_col = "cluster_final" if "cluster_final" in user_full.columns else "cluster"
    mapa_personas = dict(zip(persona_df["cluster"], persona_df["persona"]))
    user_full["persona"] = user_full[cluster_col].map(mapa_personas)
    return persona_df, user_full
