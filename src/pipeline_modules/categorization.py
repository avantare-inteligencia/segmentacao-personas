from __future__ import annotations

import pandas as pd

from ..config import DEFAULT_CONFIG, PersonaConfig


def categorizar_tela(nome: str) -> str:
    nome = str(nome).lower().strip()

    if nome == "" or nome == "nan":
        return "Outros"
    
    if any(k in nome for k in ["login", "entrar", "acessar", "autentic", "senha", "token", "boas-vindas"]):
        return "Login"
    
    if any(k in nome for k in ["home", "inicio", "início", "menu", "dashboard", "painel"]):
        return "Home/Navegacao"
    
    if any(
        k in nome
        for k in [
            "pagamento",
            "fatura",
            "boleto",
            "pix",
            "cartao",
            "cartão",
            "parcela",
            "financeiro",
            "cobranca",
            "cobrança",
            "verificacao-seguranca:alterar-frequencia",
            "alterar-frequencia-revisao",
            "alterar-frequencia-finalizacao",
            "alterar-frequencia",
        ]
    ):
        return "Pagamento/Financeiro"
    
    if any(k in nome for k in ["apolice", "apólice", "seguro", "proposta", "cobertura", "plano"]):
        return "Apolices/Seguros"
    
    if any(k in nome for k in ["sinistro", "assistencia", "assistência", "guincho", "colisao", "colisão", "ocorrencia", "ocorrência"]):
        return "Sinistro/Assistencia"
    
    if any(
        k in nome
        for k in [
            "perfil",
            "cadastro",
            "dados pessoais",
            "meus dados",
            "conta",
            "endereco",
            "endereço",
            "telefone",
            "email",
            "dados-pessoais",
            "dados-cadastrais-vi",
            "dados-profissionais-vi",
            "dados-cadastrais",
            "editar-dados-profissionais-vi",
            "editar-estado-civil",
            "editar-dados",
            "editar-nome-completo",
            "revisao-dados-profissionais",
            "revisao-nome",
            "codigo-civil",
            "dados cadastrais",
            "modal:estado-civil",
        ]
    ):
        return "Perfil/Cadastro"
    
    if any(k in nome for k in ["beneficiarios", "beneficiários", "beneficiario", "beneficiário", "percentual"]):
        return "Beneficiarios"
    
    if any(
        k in nome
        for k in [
            "compartilhar-acesso",
            "retirar-acesso",
            "compartilhar-gerenciar",
            "compartilhar acesso",
            "gerenciar-acessos-guardioes-mais-detalhes",
            "/saiba-mais-sobre-papel-guardiao",
        ]
    ):
        return "Compartilhar/Acesso"
    
    if any(k in nome for k in ["documento", "comprovante", "contrato", "termo", "arquivo", "pdf"]):
        return "Documentos"
    
    if any(k in nome for k in ["ajuda", "faq", "suporte", "atendimento", "chat", "duvida", "dúvida"]):
        return "Suporte/Ajuda"
    
    if any(k in nome for k in ["love-card", "cartas", "love card"]):
        return "Love/Card"
    
    if any(k in nome for k in ["(not set)"]):
        return "NotSet"
    
    if any(k in nome for k in ["notificacoes", "notificações", "gerenciar-notificacoes"]):
        return "Notificacoes"
    
    if any(k in nome for k in ["extrato", "consulta", "historico", "histórico", "resumo", "saldo"]):
        return "Extrato/Consulta"
    
    return "Outros"


def aplicar_categorizacao(df: pd.DataFrame, config: PersonaConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    df = df.copy()
    df["categoria_tela"] = df[config.col_screen].apply(categorizar_tela)
    return df
