# App Streamlit — Segmentação de Personas

Este projeto converte o notebook `personas_att (2).ipynb` em uma estrutura modular Python integrada com Streamlit.

## Estrutura

- `app.py`: aplicação Streamlit
- `src/config.py`: parâmetros centrais
- `src/pipeline.py`: pipeline analítica ponta a ponta
- `src/charts.py`: gráficos Plotly reutilizáveis
- `requirements.txt`: dependências

## Como executar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Entrada esperada

Por padrão o app procura um arquivo chamado:

```text
data-formatada.csv
```

Você pode trocar esse caminho direto na sidebar do Streamlit.

## O que foi modularizado

1. leitura e validação do CSV bruto;
2. limpeza e tipagem;
3. categorização de telas;
4. agregação por usuário;
5. construção da matriz de categorias;
6. engenharia de features;
7. avaliação de `k`;
8. treinamento do KMeans;
9. fusão do cluster técnico dominado por `NotSet`;
10. nomeação das personas;
11. outputs analíticos e visuais.

## Correção aplicada em relação ao notebook

No notebook original, após a criação de `cluster_final`, a persona era associada usando `cluster` bruto. Aqui isso foi corrigido: o mapeamento usa `cluster_final` quando existir.
