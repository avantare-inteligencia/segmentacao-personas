from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def fig_metricas_k(metricas_k: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metricas_k["k"], y=metricas_k["inercia"], mode="lines+markers", name="Inércia"))
    fig.add_trace(go.Scatter(x=metricas_k["k"], y=metricas_k["silhouette"], mode="lines+markers", name="Silhouette", yaxis="y2"))
    fig.update_layout(
        title="Escolha do número de clusters",
        xaxis_title="k",
        yaxis=dict(title="Inércia"),
        yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
        template="plotly_white",
        height=430,
    )
    return fig


def fig_distribuicao_personas(persona_df: pd.DataFrame, color_map: dict[str, str]) -> go.Figure:
    df = persona_df.sort_values("% da base", ascending=True)
    fig = px.bar(
        df,
        x="% da base",
        y="persona",
        orientation="h",
        color="persona",
        color_discrete_map=color_map,
        text="% da base",
        title="Distribuição das personas",
    )
    fig.update_layout(showlegend=False, template="plotly_white", height=420)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    return fig


def fig_pca(pca_df: pd.DataFrame, color_col: str = "persona", color_map: dict[str, str] | None = None) -> go.Figure:
    fig = px.scatter(
        pca_df,
        x="pca_1",
        y="pca_2",
        color=color_col,
        color_discrete_map=color_map,
        opacity=0.6,
        title="Visualização 2D dos agrupamentos (PCA)",
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig


def fig_heatmap_personas(persona_df: pd.DataFrame, cols_categoria: list[str]) -> go.Figure:
    df = persona_df[["persona"] + cols_categoria].set_index("persona")
    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            x=df.columns.tolist(),
            y=df.index.tolist(),
            colorscale="Blues",
            hoverongaps=False,
        )
    )
    fig.update_layout(title="Composição média por persona", template="plotly_white", height=420)
    return fig


def fig_principal_funcionalidade(principal_df: pd.DataFrame, color_map: dict[str, str]) -> go.Figure:
    fig = px.bar(
        principal_df,
        x="persona",
        y="proporcao_media",
        color="persona",
        color_discrete_map=color_map,
        text="principal_funcionalidade",
        title="Principal funcionalidade por persona",
    )
    fig.update_layout(showlegend=False, template="plotly_white", height=430)
    fig.update_traces(textposition="outside")
    return fig


def fig_estabilidade(estabilidade_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=estabilidade_df["seed"], y=estabilidade_df["silhouette"], mode="lines+markers", name="Silhouette"))
    fig.add_trace(go.Bar(x=estabilidade_df["seed"], y=estabilidade_df["n_clusters_finais"], name="Clusters finais", opacity=0.45))
    fig.update_layout(title="Estabilidade por seed", template="plotly_white", barmode="overlay", height=430)
    return fig


def fig_sankey(sankey_df: pd.DataFrame) -> go.Figure:
    labels = list(dict.fromkeys(sankey_df["source"].tolist() + sankey_df["target"].tolist()))
    idx = {label: i for i, label in enumerate(labels)}
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(pad=18, thickness=18, label=labels),
                link=dict(
                    source=sankey_df["source"].map(idx),
                    target=sankey_df["target"].map(idx),
                    value=sankey_df["value"],
                ),
            )
        ]
    )
    fig.update_layout(title="Fluxo ponderado: persona → categorias", template="plotly_white", height=650)
    return fig
