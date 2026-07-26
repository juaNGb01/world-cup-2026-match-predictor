# ⚽ Predição de Resultados de Futebol e Simulação da Copa do Mundo FIFA 2026

Comparação de três algoritmos de classificação supervisionada (Random Forest, KNN e Regressão Logística) para prever resultados de partidas internacionais de futebol, com aplicação do melhor modelo em uma simulação completa da Copa do Mundo FIFA 2026 — 48 seleções, fase de grupos, mata-mata e premiação individual.

Projeto desenvolvido para a disciplina de Fundamentos de Sistemas Inteligentes (UTFPR — Câmpus Dois Vizinhos), pelo Prof. Dr. Marlon Marcon.

## 🔀 Duas versões da simulação

O repositório traz **dois scripts**, com o mesmo pipeline de dados e o mesmo modelo treinado, mas propósitos diferentes na etapa de simulação do torneio:

| Script | Fase de grupos | Chaveamento do mata-mata | Uso |
|---|---|---|---|
| **`model_training_and_report_simulation.py`** | Simulada (todos jogam todos, com desempate por pontos/saldo/gols) | Sorteado dinamicamente a cada execução | Reproduz **exatamente** o resultado documentado no relatório técnico (seed fixa = reprodutível) |
| **`world_cup_2026_bracket_simulation.py`** | Fixa, definida a partir da classificação real de grupos (coluna `group_position` em `national_teams_wc.csv`) | Fixo, seguindo a estrutura oficial de confrontos por posição da Copa do Mundo 2026 | Simula só o mata-mata em cima do chaveamento real da Copa — mais interessante pra explorar "quem seria campeão" |

Os dois compartilham a mesma base de treino, os mesmos atributos e o mesmo Random Forest — a diferença está inteiramente em como a fase eliminatória é estruturada.

## 📋 Sobre o projeto

O objetivo foi treinar e comparar três modelos de classificação na tarefa de prever o resultado de uma partida entre seleções — vitória do mandante, empate ou vitória do visitante — e usar o modelo vencedor como motor de uma simulação de torneio.

A base de treinamento foi construída a partir do dataset público [player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) (Transfermarkt/Kaggle), combinando dados de jogadores, partidas e seleções para gerar **20 atributos por confronto** (10 por seleção, replicados para mandante e visitante): valor de mercado do elenco, idade média dos jogadores, caps internacionais, gols por clube e seleção, assistências, média de gols marcados por elenco, cartões amarelos e vermelhos e ranking FIFA.

Cada jogador foi associado à sua seleção nacional pelo país de cidadania, considerando apenas atletas ativos na temporada de 2025. A convocação de cada seleção (26 jogadores) foi definida por uma função que ordena os jogadores por um escore combinando valor de mercado e número de partidas internacionais.

## 🧠 Modelos comparados

O conjunto de dados (310 partidas internacionais históricas — Copa do Mundo, Eurocopa, Copa América, Copa Africana de Nações e Copa da Ásia) foi dividido de forma estratificada em **248 partidas de treino e 62 de teste (80/20)**, com semente fixa para reprodutibilidade. Os atributos foram padronizados por z-score, ajustado apenas no conjunto de treino.

| Modelo | Parâmetros | Acurácia | Precisão (macro) | Revocação (macro) | F1 (macro) |
|---|---|---|---|---|---|
| **Random Forest** | 100 árvores, balanceamento de classes | **58,06%** | 0,43 | 0,45 | 0,44 |
| KNN | k = 5 | 51,61% | 0,42 | 0,41 | 0,40 |
| Regressão Logística | até 1000 iterações, balanceamento de classes | 37,10% | 0,33 | 0,28 | 0,31 |

Como referência, prever sempre a classe majoritária (vitória do mandante) renderia 46,8% de acurácia. Nesta execução documentada no relatório, o Random Forest foi o modelo com melhor equilíbrio entre as classes e foi escolhido como classificador final.

> ℹ️ **A escolha do modelo final é dinâmica no código** — o script sempre seleciona automaticamente o modelo de maior acurácia na comparação, e não fixa o Random Forest de antemão. Isso significa que, em outro ambiente (ex: versão diferente do scikit-learn — ver limitação de reprodutibilidade abaixo), um modelo diferente pode vencer a comparação e ser usado como motor da simulação da Copa, o que também mudaria os resultados do torneio, não só a tabela de métricas.

Vale destacar as limitações da base: das mais de 80 mil partidas presentes na base de dados original, apenas 310 representavam jogos internacionais de seleções (número a confirmar — não consta explicitamente no relatório técnico). Do total, apenas 62 partidas ficaram no conjunto de teste, das quais só 9 eram empates — o que torna os empates a classe mais difícil de prever para todos os modelos (nenhum acerto do Random Forest e da Regressão Logística, e apenas 1 do KNN), um comportamento coerente com a própria incerteza do futebol, mas que também limita a robustez estatística dos números acima.

Além disso, algumas seleções não possuíam jogadores suficientes para compor a convocação de 26 atletas, ou não constavam na base de dados. Para preservar o formato de 48 seleções, essas equipes foram substituídas por outras da mesma confederação e com ranking FIFA semelhante:

| Seleção da Copa de 2026 | Seleção utilizada (conjunto de dados) |
|---|---|
| Curaçao | Jamaica |
| Cabo Verde | Uganda |
| Congo | Etiópia |

## 🏆 Simulação da Copa do Mundo 2026

Com o Random Forest treinado, o projeto simula o torneio completo:

- Fase de grupos com as 48 seleções (simulada em `model_training_and_report_simulation.py`, ou fixa a partir da classificação real em `world_cup_2026_bracket_simulation.py`)
- Mata-mata desde os dezesseis-avos até a final, com pênaltis modelados para os empates (majoritariamente aleatório, com pequena vantagem para a seleção de melhor ranking FIFA)
- Distribuição de gols e assistências entre jogadores convocados, com base em escores ofensivos individuais
- Premiação final: Artilheiro, Luva de Ouro, Garçom, Revelação (sub-21) e Bola de Ouro

**Resultado documentado no relatório (`model_training_and_report_simulation.py`, seed fixa = 1781559800):** 🇪🇸 Espanha campeã, vencendo a 🇫🇷 França por 2 a 0 na final.

| Prêmio | Jogador | Seleção | Destaque |
|---|---|---|---|
| Artilheiro | Antoine Griezmann | França | 4 gols no torneio |
| Luva de Ouro | Unai Simón | Espanha | 8 jogos, 3 gols sofridos (média 0,38/jogo) |
| Garçom | Bradley Barcola | França | 3 assistências |
| Revelação (sub-21) | Lamine Yamal | Espanha | 17,9 anos, 2 gols e 1 assistência |
| Bola de Ouro | Antoine Griezmann | França | Ponta, 4 gols e 1 assistência |

A simulação preservou, em linhas gerais, a hierarquia esperada entre as seleções — Espanha (2ª no ranking FIFA) e França (1ª) chegaram à final —, mas também produziu eliminações precoces plausíveis, como a queda do Brasil diante dos Países Baixos já nos dezesseis-avos de final.

> ⚠️ Como o mata-mata em `world_cup_2026_bracket_simulation.py` é decidido por amostragem das probabilidades do modelo (não pelo resultado "mais provável" de forma determinística), essa versão pode gerar um campeão diferente a cada execução — isso é esperado e intencional, e reflete a incerteza real de um torneio eliminatório.

## 🛠️ Tecnologias

- **Python** (pandas, numpy)
- **scikit-learn** — RandomForestClassifier, KNeighborsClassifier, LogisticRegression, StandardScaler
- **matplotlib / seaborn** — matrizes de confusão
- Desenvolvido e executado em **Google Colab**

## 📂 Estrutura

```
data/
  players.csv                                  # jogadores (filtrado ao necessário para o projeto)
  appearances.csv                              # participações em partidas
  games.csv                                     # partidas
  national_teams.csv                            # seleções nacionais e ranking FIFA
  national_teams_wc.csv                         # as 48 seleções da Copa 2026, grupos e classificação final
model_training_and_report_simulation.py         # versão do relatório: treino/comparação dos modelos + simulação completa (grupos simulados, seed fixa)
world_cup_2026_bracket_simulation.py            # versão com chaveamento oficial: grupos fixos + mata-mata seguindo a estrutura real da Copa 2026
requirements.txt                                # dependências do projeto
relatorio_ml.pdf                                # relatório técnico completo (formato artigo/ABNT)
comparativo_modelos.png                         # matrizes de confusão dos 3 modelos
```

Os arquivos em `data/` já estão inclusos no repositório (apenas os dados efetivamente usados no projeto, extraídos do dataset [player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) do Kaggle) — não é necessário baixar nada separadamente.

## ▶️ Como executar

1. Clone o repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute um dos dois scripts, a partir da raiz do repositório:
   - `python model_training_and_report_simulation.py` — reproduz o resultado documentado no relatório
   - `python world_cup_2026_bracket_simulation.py` — simula o mata-mata no chaveamento oficial da Copa 2026


## 📄 Relatório completo

O relatório técnico com fundamentação teórica, metodologia detalhada e resultados completos da simulação (classificação de grupos, chaveamento do mata-mata e premiação) está disponível neste repositório.

## ⚠️ Nota sobre limitações

Este é um projeto acadêmico com fins de aprendizado sobre classificação supervisionada — não uma ferramenta de previsão real de apostas ou resultados esportivos. A acurácia de ~58% reflete a dificuldade genuína do problema (o futebol tem alta variância) e o tamanho reduzido da base de treinamento disponível (310 partidas, das quais apenas 62 no teste).

Outras limitações identificadas, algumas após a entrega do relatório:

- **Reprodutibilidade entre ambientes:** os resultados de acurácia e da simulação documentados no relatório foram obtidos no Google Colab com scikit-learn 1.6.1. Mesmo com a mesma seed, pequenas variações podem ocorrer em outros ambientes ou versões de bibliotecas — uma limitação conhecida de pipelines de machine learning, não um erro de execução.
- **Convocação de jogadores aposentados da seleção:** o critério de convocação usado no projeto é baseado em desempenho recente nos clubes, e não distingue jogadores que se aposentaram da seleção nacional mas seguem ativos em nível de clube. Como essa informação (aposentadoria da seleção) não é um atributo estruturado no dataset utilizado, esses atletas podem ser incluídos erroneamente na convocação simulada.
- **Dependência da qualidade do mapeamento seleção-jogador:** conforme apontado no relatório, a associação entre jogadores e seleções (pelo país de cidadania) pode subestimar a força de algumas equipes quando a cobertura de dados é incompleta.

Como trabalhos futuros (sugeridos no relatório): adoção de amostragem a partir das probabilidades do classificador para introduzir maior variabilidade nas simulações (já implementado em `world_cup_2026_bracket_simulation.py`), aprimoramento do mapeamento das seleções, inclusão de novos atributos (como forma recente) e uso de dados mais atualizados.
