from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PersonaConfig:
    data_path: str = "data-formatada.csv"
    n_clusters: int = 5
    random_state: int = 42
    k_range: Sequence[int] = tuple(range(2, 11))

    col_user: str = "CPF usuário"
    col_datetime: str = "Data e hora (AAAAMMDDHH)"
    col_events: str = "Contagem de eventos"
    col_screen: str = "Título da página e nome da tela"
    col_session: str = "GA Sessioin ID"

    exported_dir: str = "outputs"
    min_share_categoria_ativa: float = 0.05
    notset_merge_threshold: float = 0.50
    stability_seeds: Sequence[int] = (0, 1, 2, 3, 7, 13, 21, 42, 99, 123)

    categories_transacionais: Sequence[str] = (
        "Pagamento/Financeiro",
        "Apolices/Seguros",
        "Extrato/Consulta",
        "Beneficiarios",
    )
    categories_navegacionais: Sequence[str] = (
        "Home/Navegacao",
        "Documentos",
        "Perfil/Cadastro",
        "Suporte/Ajuda",
        "Love/Card",
    )

    persona_color_map: dict[str, str] = field(
        default_factory=lambda: {
            "Acesso Concentrado": "#F59E0B",
            "Engajado Financeiro": "#2563EB",
            "Engajado em Seguros": "#16A34A",
            "Explorador": "#7C3AED",
            "Explorador de Navegação": "#7C3AED",
            "Login + Financeiro": "#2563EB",
            "Login + Seguros": "#16A34A",
            "Login + Consulta": "#0EA5E9",
            "Login + Autosserviço": "#0891B2",
            "Multifuncional Engajado": "#7C3AED",
            "Base Ampla de Acesso": "#64748B",
            "Base Ampla de Baixa Intensidade": "#94A3B8",
            "Recorrente de Acesso": "#F97316",
        }
    )


DEFAULT_CONFIG = PersonaConfig()


def ensure_output_dir(base_dir: str | Path, config: PersonaConfig = DEFAULT_CONFIG) -> Path:
    output_dir = Path(base_dir) / config.exported_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
