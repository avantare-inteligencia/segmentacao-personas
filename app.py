from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.charts import (
    fig_distribuicao_personas,
    fig_estabilidade,
    fig_heatmap_personas,
    fig_metricas_k,
    fig_pca,
    fig_principal_funcionalidade,
    fig_sankey,
)
from src.config import DEFAULT_CONFIG
from src.pipeline import export_outputs, run_pipeline

st.set_page_config(page_title="Segmentação de Personas", page_icon="🧭", layout="wide")


@st.cache_data(show_spinner=False)
def cached_run_pipeline(data_path: str):
    return run_pipeline(data_path)


@st.cache_data(show_spinner=False)
def make_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(99,102,241,.10), transparent 22%),
                radial-gradient(circle at top left, rgba(14,165,233,.08), transparent 18%),
                # linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1300px;  
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617 0%, #0f172a 52%, #111827 100%);
            border-right: 1px solid rgba(255,255,255,0.07);
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0;
        }
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #172554 45%, #312e81 100%);
            border: 1px solid rgba(255,255,255,.08);
            padding: 30px 32px;
            border-radius: 26px;
            color: white;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.16);
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
        }
        .hero:after {
            content: "";
            position: absolute;
            width: 240px;
            height: 240px;
            right: -40px;
            top: -60px;
            background: radial-gradient(circle, rgba(255,255,255,.16), transparent 65%);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.15rem;
            line-height: 1.05;
            letter-spacing: -0.02em;
        }
        .hero p {
            margin: 12px 0 0 0;
            max-width: 860px;
            font-size: 1rem;
            color: rgba(255,255,255,0.88);
        }
        .hero-kicker {
            display: inline-block;
            margin-bottom: 12px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,.10);
            color: rgba(255,255,255,.92);
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .04em;
            text-transform: uppercase;
        }
        .surface {
            # background: rgba(255,255,255,.86);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(148,163,184,.18);
            border-radius: 24px;
            padding: 18px 20px;
            box-shadow: 0 12px 30px rgba(15,23,42,.06);
            margin-bottom: 16px;
            margin-top: 16px;
        }
        .surface h3, .surface h4 {
            color: rgba(255,255,255,0.88);
        }
        .section-title {
            color: rgba(255,255,255,0.88);
            font-size: 1.18rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            margin-bottom: 4px;
        }
        .section-subtitle {
            color: rgba(255,255,255,0.88);
            font-size: .95rem;
            margin-bottom: 10px;
        }
        .metric-card {
            # background: linear-gradient(180deg, rgba(255,255,255,.95), rgba(248,250,252,.95));
            border: 1px solid rgba(148,163,184,.16);
            border-radius: 22px;
            padding: 18px 18px;
            box-shadow: 0 10px 22px rgba(15,23,42,.05);
            min-height: 122px;
        }
        .metric-label {
            color: rgba(255,255,255,0.88);
            font-size: .86rem;
            margin-bottom: 8px;
        }
        .metric-value {
            color: rgba(255,255,255,0.88);
            font-size: 1.72rem;
            font-weight: 800;
            line-height: 1.05;
            letter-spacing: -0.02em;
        }
        .metric-sub {
            color: rgba(255,255,255,0.88);
            font-size: .84rem;
            margin-top: 8px;
        }
        .persona-card {
            # background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(248,250,252,.95));
            border: 1px solid rgba(148,163,184,.15);
            border-radius: 24px;
            padding: 18px;
            box-shadow: 0 12px 24px rgba(15,23,42,.05);
            min-height: 215px;
        }
        .persona-chip {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            color: white;
            font-size: .78rem;
            font-weight: 700;
            margin-bottom: 12px;
        }
        .persona-title {
            color: rgba(255,255,255,0.88);
            font-size: 1.1rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 8px;
        }
        .persona-desc {
            color: rgba(255,255,255,0.88);
            font-size: .92rem;
            line-height: 1.5;
            margin-bottom: 14px;
        }
        .persona-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: auto;
        }
        .meta-pill {
            padding: 7px 10px;
            border-radius: 999px;
            # background: #f8fafc;
            border: 1px solid #e2e8f0;
            color: rgba(255,255,255,0.88);
            font-size: .78rem;
            font-weight: 600;
        }
        .note-card {
            # background: linear-gradient(135deg, rgba(255,255,255,.95), rgba(238,242,255,.95));
            border-radius: 24px;
            padding: 18px 20px;
            border: 1px solid rgba(99,102,241,.14);
            box-shadow: 0 12px 28px rgba(79,70,229,.06);
            margin-top: 16px;
        }
        .note-card ul {
            margin: .4rem 0 0 1rem;
        }
        .persona-detail {
            # background: rgba(255,255,255,.92);
            border: 1px solid rgba(148,163,184,.16);
            border-radius: 24px;
            padding: 18px;
            box-shadow: 0 10px 20px rgba(15,23,42,.05);
        }
        .detail-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0,1fr));
            gap: 12px;
        }
        .detail-item {
            # background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 12px 14px;
        }
        .detail-k {
            color: rgba(255,255,255,0.88);
            font-size: .8rem;
            margin-bottom: 4px;
        }
        .detail-v {
            color: rgba(255,255,255,0.88);
            font-size: .98rem;
            font-weight: 700;
            line-height: 1.25;
        }
        .small-muted {
            color: rgba(255,255,255,0.88);
            font-size: .86rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            margin-bottom: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            # background: rgba(255,255,255,.82);
            border: 1px solid rgba(148,163,184,.16);
            border-radius: 999px;
            padding: 0 16px;
        }
        div[data-testid="stMetric"] {
            # background: rgba(255,255,255,.92);
            border: 1px solid rgba(148,163,184,.14);
            padding: 12px 14px;
            border-radius: 18px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }
        .stDownloadButton button, .stButton button {
            border-radius: 14px !important;
            min-height: 44px;
            font-weight: 700;
        }

        .surface-card {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: var(--secondary-background-color);
            border: 1px solid rgba(128,128,128,0.18);
            margin-bottom: 1rem;
        }

        .small-muted {
            font-size: 0.82rem;
            opacity: 0.72;
        }

        .persona-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: rgba(127,127,127,0.12);
            border: 1px solid rgba(127,127,127,0.18);
            font-weight: 600;
            font-size: 0.9rem;
        }

        .mini-step {
            padding: 0.85rem 0.7rem;
            border-radius: 14px;
            text-align: center;
            background: var(--background-color);
            border: 1px solid rgba(128,128,128,0.16);
            font-weight: 600;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, kicker: str = "Projeto analítico") -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">{escape(kicker)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def surface_start(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<div class="section-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="surface">
            <div class="section-title">{escape(title)}</div>
            {subtitle_html}
        """,
        unsafe_allow_html=True,
    )


def surface_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)

def mask_user_id(value) -> str:
    if pd.isna(value):
        return "Usuário não identificado"
    s = str(value)
    if len(s) <= 4:
        return f"Usuário #{s}"
    return f"Usuário #{s[:3]}***{s[-2:]}"


def pick_user_id_column(df: pd.DataFrame) -> str | None:
    priority_candidates = [
        "CPF usuário",
        "CPF",
        "cpf",
        "Cpf",
        "cpf_cnpj",
        "CPF_CNPJ",
        "documento",
        "Documento",
        "user_id",
        "usuario_id",
        "id_usuario",
        "idCliente",
        "cliente_id",
        "id",
    ]

    for col in priority_candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        col_norm = str(col).strip().lower()
        if any(token in col_norm for token in ["cpf", "document", "usuario", "user", "cliente"]):
            return col

    return None


def pick_existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def format_metric_value(value, kind: str = "int") -> str:
    if pd.isna(value):
        return "-"
    if kind == "int":
        return f"{int(round(float(value))):,}".replace(",", ".")
    if kind == "float1":
        return f"{float(value):.1f}"
    if kind == "pct":
        return f"{float(value) * 100:.1f}%"
    return str(value)


def build_user_feature_table(user_row: pd.Series) -> pd.DataFrame:
    rows = [
        ("Sessões", user_row.get("num_sessoes"), "Volume total de sessões associadas ao usuário."),
        ("Eventos", user_row.get("eventos_total"), "Quantidade total de interações registradas."),
        ("Dias ativos", user_row.get("dias_ativos"), "Número de dias com atividade observada."),
        ("Janela de atividade", user_row.get("janela_atividade"), "Distância entre a primeira e a última atividade."),
        ("Recência", user_row.get("recencia"), "Tempo desde a última atividade observada."),
        ("Categorias ativas", user_row.get("n_categorias_ativas"), "Quantidade de categorias funcionais utilizadas."),
        ("Concentração nas 2 principais", user_row.get("share_top2"), "Quanto o uso está concentrado nas duas categorias mais relevantes."),
        ("Categoria dominante sem login", user_row.get("categoria_dominante_sem_login"), "Principal funcionalidade utilizada desconsiderando login."),
        ("Share dominante sem login", user_row.get("share_dominante_sem_login"), "Peso da principal categoria desconsiderando login."),
        ("Diversidade de navegação", user_row.get("entropia_categorias"), "Medida de dispersão do uso entre categorias."),
        ("Soma transacional", user_row.get("soma_transacional"), "Indicador agregado de uso orientado a transação."),
        ("Soma navegacional", user_row.get("soma_navegacional"), "Indicador agregado de uso mais exploratório."),
        ("Login puro", user_row.get("login_puro_score"), "Intensidade de uso concentrado em login."),
        ("Login com profundidade", user_row.get("login_com_profundidade_score"), "Uso com login, mas acompanhado de navegação posterior."),
    ]

    df_out = pd.DataFrame(rows, columns=["Indicador", "Valor", "Leitura"])
    return df_out


def build_user_category_chart_df(user_row: pd.Series, category_cols: list[str]) -> pd.DataFrame:
    values = []
    for col in category_cols:
        val = float(user_row.get(col, 0.0) or 0.0)
        if val > 0:
            values.append({"Categoria": col, "Share": val})
    out = pd.DataFrame(values)
    if out.empty:
        return pd.DataFrame({"Categoria": [], "Share": []})
    out = out.sort_values("Share", ascending=True)
    return out


def render_user_aggregation_story():
    st.markdown(
        """
        <div class="surface-card">
            <h3>Como a agregação funciona</h3>
            <p>O comportamento bruto do app é consolidado em uma visão única por usuário, permitindo comparar intensidade, foco e diversidade de uso.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="surface-card">
                <h4>Antes: navegação bruta</h4>
                <ul>
                    <li>Múltiplas sessões ao longo do tempo</li>
                    <li>Eventos distribuídos em várias telas</li>
                    <li>Interações fragmentadas por funcionalidade</li>
                    <li>Leitura difícil para negócio</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="surface-card">
                <h4>Depois: ficha comportamental</h4>
                <ul>
                    <li>1 linha por usuário</li>
                    <li>Métricas de intensidade e recorrência</li>
                    <li>Composição funcional por categoria</li>
                    <li>Base pronta para segmentação em personas</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### O que a ficha do usuário contém")
    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        st.markdown(
            """
            <div class="surface-card">
                <h4>Intensidade</h4>
                <p>Sessões, eventos e dias ativos.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="surface-card">
                <h4>Recorrência</h4>
                <p>Janela de atividade e recência de uso.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="surface-card">
                <h4>Composição funcional</h4>
                <p>Shares por categoria como Login, Seguros, Pagamentos e Home.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="surface-card">
                <h4>Estrutura do comportamento</h4>
                <p>Concentração, diversidade e profundidade pós-login.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

def stat_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-sub">{escape(sub)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def format_pct(v) -> str:
    try:
        return f"{float(v):.1f}%"
    except Exception:
        return str(v)


def persona_card_html(row, color: str) -> str:
    desc = row.get("descricao", "")
    meta = [
        f"Base: {format_pct(row.get('% da base', 0))}",
        f"Dominante: {row.get('categoria_dominante', '-')}",
        f"Ativas: {int(row.get('n_categorias_ativas', 0)) if row.get('n_categorias_ativas') == row.get('n_categorias_ativas') else '-'}",
    ]
    pills = "".join(f'<div class="meta-pill">{escape(str(item))}</div>' for item in meta)
    return f"""
        <div class="persona-card">
            <div class="persona-chip" style="background:{color};">Persona</div>
            <div class="persona-title">{escape(str(row.get('persona', '')))}</div>
            <div class="persona-desc">{escape(str(desc))}</div>
            <div class="persona-meta">{pills}</div>
        </div>
    """


def persona_detail_card(row, color: str) -> str:
    items = [
        ("Participação na base", format_pct(row.get("% da base", 0))),
        ("Categoria dominante", row.get("categoria_dominante", "-")),
        ("Share dominante", format_pct(row.get("share_dominante", 0))),
        ("Categoria dominante sem login", row.get("categoria_dominante_sem_login", "-")),
        ("Share login", f"{float(row.get('share_login', 0)):.2f}"),
        ("Entropia", f"{float(row.get('entropia_categorias', 0)):.2f}"),
        ("Categorias ativas", str(int(row.get("n_categorias_ativas", 0)))),
        ("Score de engajamento", f"{float(row.get('score_engajamento', 0)):.2f}"),
    ]
    html_items = "".join(
        f'<div class="detail-item"><div class="detail-k">{escape(k)}</div><div class="detail-v">{escape(v)}</div></div>'
        for k, v in items
    )
    return f"""
        <div class="persona-detail">
            <div class="persona-chip" style="background:{color};">Perfil selecionado</div>
            <div class="persona-title">{escape(str(row.get('persona', '')))}</div>
            <div class="persona-desc">{escape(str(row.get('descricao', '')))}</div>
            <div class="detail-grid">{html_items}</div>
        </div>
    """


def render_sidebar() -> tuple[str, str]:
    st.sidebar.markdown("## Segmentação de Personas")
    st.sidebar.caption("Aplicação analítica para explorar metodologia, personas e robustez da solução.")
    st.sidebar.markdown("---")
    data_path = st.sidebar.text_input("CSV bruto", value=DEFAULT_CONFIG.data_path)
    page = st.sidebar.radio(
        "Navegação",
        ["Visão Geral", "Metodologia", "Usuário", "Personas", "Estabilidade", "Fluxo", "Exportar"],
    )
    st.sidebar.markdown("---")
    return data_path, page


def render_persona_cards(persona_df) -> None:
    rows = persona_df.sort_values("% da base", ascending=False).to_dict("records")
    cols = st.columns(2)
    for idx, row in enumerate(rows):
        color = DEFAULT_CONFIG.persona_color_map.get(row["persona"], "#334155")
        with cols[idx % 2]:
            st.markdown(persona_card_html(row, color), unsafe_allow_html=True)


def render_home(artifacts):
    hero(
        "Segmentação de personas por comportamento no app",
        "Uma experiência analítica para explicar o racional do projeto, a modelagem comportamental e os perfis finais de usuários.",
        kicker="Visão executiva",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Usuários modelados", format_int(len(artifacts.user_full)), "Base final usada na leitura comportamental")
    with c2:
        stat_card(
            "Eventos",
            format_int(int(artifacts.df_eventos[DEFAULT_CONFIG.col_events].sum())),
            "Volume total agregado da navegação",
        )
    with c3:
        stat_card("Clusters finais", str(int(artifacts.persona_df["cluster"].nunique())), "Após consolidação do cluster técnico")
    with c4:
        stat_card("Silhouette final", f"{artifacts.score_final:.4f}", "Qualidade relativa da separação dos grupos")

    left, right = st.columns([1.35, 0.95])
    with left:
        surface_start("Distribuição das personas", "Leitura de participação relativa de cada perfil dentro da base consolidada.")
        st.plotly_chart(
            fig_distribuicao_personas(artifacts.persona_df, DEFAULT_CONFIG.persona_color_map),
            use_container_width=True,
        )
        surface_end()
    with right:
        st.markdown(
            """
            <div class="note-card">
                <div class="section-title">Leitura executiva</div>
                <div class="section-subtitle">A aplicação foi redesenhada para comunicar o projeto em camadas e deixar a leitura menos técnica quando necessário.</div>
                <ul>
                    <li>Explica a metodologia sem depender do notebook.</li>
                    <li>Mostra as features que destravaram a separação dos grupos.</li>
                    <li>Apresenta as personas em formato narrativo, não só tabular.</li>
                    <li>Traz evidências de robustez por seeds.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    surface_start("Leitura das personas", "Cards com nome, descrição e assinatura resumida de cada perfil.")
    render_persona_cards(artifacts.persona_df)
    surface_end()


def render_metodologia(artifacts):
    hero(
        "Metodologia da solução",
        "A estrutura abaixo explica como os eventos brutos viram uma representação comportamental, depois clusters e finalmente personas interpretáveis.",
        kicker="Funcionamento técnico",
    )

    tab1, tab2, tab3 = st.tabs(["Pipeline", "Features", "Modelagem"])

    with tab1:
        left, right = st.columns([1.05, 0.95])
        with left:
            surface_start(
                "Etapas da pipeline",
                "Leitura, limpeza, categorização, agregação por usuário, engenharia de features, clusterização, fusão do cluster técnico, nomeação das personas e exportação.",
            )
            st.code("\n".join(artifacts.features_modelo), language="text")
            surface_end()
        with right:
            surface_start("Tratamento do cluster técnico", "Resumo do merge aplicado ao cluster dominado por NotSet, quando identificado.")
            st.json(artifacts.info_merge_notset)
            surface_end()

    with tab2:
        surface_start("Espaço das features", "Visualização bidimensional do espaço modelado após padronização e projeção por PCA.")
        st.caption(f"Variância explicada: PC1 = {artifacts.variancia_pca[0]:.2%} | PC2 = {artifacts.variancia_pca[1]:.2%}")
        color_col = "persona" if "persona" in artifacts.pca_df.columns else "cluster"
        st.plotly_chart(
            fig_pca(artifacts.pca_df, color_col=color_col, color_map=DEFAULT_CONFIG.persona_color_map),
            use_container_width=True,
        )
        surface_end()

    with tab3:
        col1, col2 = st.columns([1.1, 0.9])
        with col1:
            surface_start("Escolha de k", "A solução considera silhouette, inércia, interpretabilidade e consolidação final.")
            st.plotly_chart(fig_metricas_k(artifacts.metricas_k), use_container_width=True)
            surface_end()
        with col2:
            st.markdown(
                """
                <div class="note-card">
                    <div class="section-title">Critérios considerados</div>
                    <div class="section-subtitle">A leitura final não é definida só por métrica matemática.</div>
                    <ul>
                        <li>Silhouette e inércia.</li>
                        <li>Coerência funcional dos grupos.</li>
                        <li>Tratamento do cluster técnico dominado por NotSet.</li>
                        <li>Consolidação em personas inteligíveis para negócio.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

def render_usuario(artifacts):
    hero(
        "Como o usuário é representado",
        "Nesta etapa, os eventos brutos do app são consolidados em uma visão única por usuário, transformando navegação em métricas comportamentais comparáveis.",
        kicker="Visão do usuário",
    )

    user_full = artifacts.user_full.copy()

    if user_full is None or user_full.empty:
        st.warning("A base agregada de usuários não está disponível.")
        return

    render_user_aggregation_story()

    st.markdown("## Explorar um usuário-exemplo")

    persona_col = "persona" if "persona" in user_full.columns else None
    user_id_col = pick_user_id_column(user_full)

    if user_id_col is None:
        st.warning("Não foi possível identificar a coluna de usuário para montar a visão individual.")
        return

    selectable_df = user_full.copy()

    filtro1, filtro2 = st.columns([1, 2], gap="large")

    with filtro1:
        if persona_col:
            personas_disp = ["Todas"] + sorted([p for p in selectable_df[persona_col].dropna().unique().tolist()])
            persona_sel = st.selectbox("Filtrar por persona", personas_disp, index=0)
            if persona_sel != "Todas":
                selectable_df = selectable_df[selectable_df[persona_col] == persona_sel].copy()

    if selectable_df.empty:
        st.info("Não há usuários disponíveis para o filtro selecionado.")
        return

    selectable_df = selectable_df.sort_values(
        by=[c for c in ["eventos_total", "num_sessoes", "dias_ativos"] if c in selectable_df.columns],
        ascending=False,
    )

    selectable_df["user_label"] = selectable_df[user_id_col].apply(mask_user_id)

    with filtro2:
        selected_label = st.selectbox(
            "Selecionar usuário-exemplo",
            selectable_df["user_label"].tolist(),
            index=0,
        )

    selected_row = selectable_df.loc[selectable_df["user_label"] == selected_label].iloc[0]

    persona_value = selected_row.get("persona", "Não classificado")
    categoria_dom = selected_row.get("categoria_dominante_sem_login", selected_row.get("categoria_dominante", "-"))
    intensidade = selected_row.get("eventos_total", np.nan)

    st.markdown("## Ficha comportamental do usuário")

    st.markdown(
        f"""
        <div class="surface-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap;">
                <div>
                    <div class="small-muted">Usuário selecionado</div>
                    <h3 style="margin:4px 0 8px 0;">{selected_label}</h3>
                    <p style="margin:0;">
                        <strong>Persona:</strong> {persona_value}<br>
                        <strong>Categoria dominante:</strong> {categoria_dom}<br>
                        <strong>Intensidade observada:</strong> {format_metric_value(intensidade, "int")} eventos
                    </p>
                </div>
                <div class="persona-chip">{persona_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        st.metric("Sessões", format_metric_value(selected_row.get("num_sessoes"), "int"))
    with k2:
        st.metric("Eventos", format_metric_value(selected_row.get("eventos_total"), "int"))
    with k3:
        st.metric("Dias ativos", format_metric_value(selected_row.get("dias_ativos"), "int"))
    with k4:
        st.metric("Recência", format_metric_value(selected_row.get("recencia"), "int"))

    st.markdown("## Composição funcional do usuário")

    category_candidates = [
        "Login",
        "Apolices/Seguros",
        "Pagamento/Financeiro",
        "Home/Navegacao",
        "Outros",
        "NotSet",
    ]
    category_cols = pick_existing_cols(user_full, category_candidates)
    chart_df = build_user_category_chart_df(selected_row, category_cols)

    c1, c2 = st.columns([1.15, 1], gap="large")

    with c1:
        if not chart_df.empty:
            fig = px.bar(
                chart_df,
                x="Share",
                y="Categoria",
                orientation="h",
                text="Share",
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_tickformat=".0%",
                yaxis_title="",
                xaxis_title="Participação na navegação",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Não há composição funcional disponível para este usuário.")

    with c2:
        feature_table = build_user_feature_table(selected_row)
        st.dataframe(feature_table, use_container_width=True, hide_index=True)

    st.markdown("## Como essa agregação apoia a segmentação")
    st.markdown(
        """
        <div class="surface-card">
            <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap:12px;">
                <div class="mini-step">Eventos brutos</div>
                <div class="mini-step">Categorias funcionais</div>
                <div class="mini-step">Agregação por usuário</div>
                <div class="mini-step">Features comportamentais</div>
                <div class="mini-step">Persona final</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="surface-card">
            <h3>Leitura de negócio</h3>
            <p>
                A segmentação não trabalha com eventos isolados. Ela usa a estrutura consolidada do comportamento do usuário,
                permitindo separar intensidade de uso, foco funcional e diversidade de navegação em uma base comparável.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    ) 

def render_personas(artifacts):
    hero(
        "Personas finais",
        "A página foi reorganizada para privilegiar leitura e narrativa: cards, comparação visual e detalhamento do perfil selecionado.",
        kicker="Perfis comportamentais",
    )

    surface_start("Cards das personas", "Cada card resume a essência do grupo e substitui a antiga leitura tabular da página.")
    render_persona_cards(artifacts.persona_df)
    surface_end()

    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        surface_start("Heatmap comportamental", "Comparação visual entre os perfis nas principais categorias funcionais.")
        st.plotly_chart(fig_heatmap_personas(artifacts.persona_df, artifacts.cols_categoria), use_container_width=True)
        surface_end()
    with c2:
        surface_start("Principal funcionalidade", "Categoria que melhor representa cada persona na solução consolidada.")
        st.plotly_chart(
            fig_principal_funcionalidade(artifacts.principal_funcionalidade_df, DEFAULT_CONFIG.persona_color_map),
            use_container_width=True,
        )
        surface_end()

    persona = st.selectbox("Selecione uma persona", sorted(artifacts.persona_df["persona"].unique().tolist()))
    detalhe = artifacts.persona_df.loc[artifacts.persona_df["persona"] == persona].iloc[0]
    top_telas = artifacts.top_telas_persona.loc[artifacts.top_telas_persona["persona"] == persona].copy()
    cor = DEFAULT_CONFIG.persona_color_map.get(persona, "#334155")

    col1, col2 = st.columns([1.0, 1.0])
    with col1:
        st.markdown(persona_detail_card(detalhe, cor), unsafe_allow_html=True)
    with col2:
        surface_start("Top telas associadas", "Principais telas ou pontos de contato que ajudam a caracterizar a persona selecionada.")
        st.dataframe(top_telas, use_container_width=True, hide_index=True)
        surface_end()


def render_estabilidade(artifacts):
    hero(
        "Estabilidade da solução",
        "A robustez foi testada com múltiplas seeds para verificar consistência de silhouette, número de grupos finais e perfis médios.",
        kicker="Robustez",
    )
    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        surface_start("Resultado por seed", "Leitura consolidada do comportamento das execuções com diferentes seeds.")
        st.plotly_chart(fig_estabilidade(artifacts.estabilidade_df), use_container_width=True)
        surface_end()
    with col2:
        surface_start("Resumo consolidado", "Tabela executiva com silhouette, número de clusters finais e tamanhos relativos dos grupos.")
        st.dataframe(artifacts.estabilidade_df.sort_values("seed"), use_container_width=True, hide_index=True)
        surface_end()

    with st.expander("Abrir perfis médios por cluster e seed"):
        st.dataframe(artifacts.estabilidade_clusters_df, use_container_width=True, hide_index=True)


def render_fluxo(artifacts):
    hero(
        "Fluxo persona → categorias",
        "Visualização das relações entre personas consolidadas e os blocos funcionais que estruturam a navegação do app.",
        kicker="Estrutura funcional",
    )
    surface_start("Fluxo funcional", "Uma leitura visual da concentração relativa das categorias dentro de cada persona.")
    st.plotly_chart(fig_sankey(artifacts.sankey_df), use_container_width=True)
    surface_end()


def render_exportar(artifacts):
    hero(
        "Exportação dos outputs",
        "Gere os artefatos finais para documentação, validações adicionais e compartilhamento com o time.",
        kicker="Saídas do projeto",
    )
    col1, col2 = st.columns([0.95, 1.05])
    with col1:
        st.markdown(
            """
            <div class="note-card">
                <div class="section-title">Arquivos exportáveis</div>
                <div class="section-subtitle">Incluem base por usuário, resumo das personas, métricas de k, estabilidade por seed e tabelas auxiliares.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Exportar outputs", use_container_width=True, type="primary"):
            out_dir = export_outputs(artifacts, Path.cwd())
            st.success(f"Arquivos exportados em: {out_dir}")

        csv_bytes = make_csv(artifacts.persona_df)
        st.download_button(
            "Baixar resumo de personas (CSV)",
            data=csv_bytes,
            file_name="resumo_personas.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        surface_start("Prévia do resumo de personas", "A tabela segue disponível aqui apenas como artefato de apoio e exportação.")
        st.dataframe(artifacts.persona_df, use_container_width=True, hide_index=True)
        surface_end()


def main():
    inject_css()
    data_path, page = render_sidebar()

    if not Path(data_path).exists():
        hero(
            "Aplicação pronta para análise",
            "Informe o caminho do CSV bruto do projeto na barra lateral para carregar a pipeline e visualizar os resultados.",
            kicker="Próximo passo",
        )
        st.info("O app foi preparado para rodar sobre os dados originais do notebook.")
        st.stop()

    with st.spinner("Processando a base e calculando as personas..."):
        artifacts = cached_run_pipeline(data_path)

    if page == "Visão Geral":
        render_home(artifacts)
    elif page == "Metodologia":
        render_metodologia(artifacts)
    elif page == "Usuário":
        render_usuario(artifacts)
    elif page == "Personas":
        render_personas(artifacts)
    elif page == "Estabilidade":
        render_estabilidade(artifacts)
    elif page == "Fluxo":
        render_fluxo(artifacts)
    else:
        render_exportar(artifacts)


if __name__ == "__main__":
    main()
