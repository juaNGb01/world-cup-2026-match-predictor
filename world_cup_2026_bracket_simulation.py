
# ============================================================
# CÉLULA 1 — IMPORTAÇÕES E CARREGAMENTO DOS DADOS
# ============================================================

import pandas as pd
import numpy as np

pd.read_csv('./data/players.csv', sep=None, engine='python', on_bad_lines='warn')

# Carregando datasets do Kaggle
players     = pd.read_csv('./data/players.csv',  sep=None, engine='python', on_bad_lines='warn')
appearances = pd.read_csv('./data/appearances.csv' , sep=None, engine='python', on_bad_lines='warn')
games       = pd.read_csv('./data/games.csv' , sep=None, engine='python', on_bad_lines='warn')
nat_teams   = pd.read_csv('./data/national_teams.csv' , sep=None, engine='python', on_bad_lines='warn')

try:
    wc_teams = pd.read_csv('./data/national_teams_wc.csv', sep=None, engine='python', encoding='utf-8')
except UnicodeDecodeError:
    wc_teams = pd.read_csv('./data/national_teams_wc.csv', sep=None, engine='python', encoding='latin-1')

print("Arquivos carregados!")
print(f"   Jogadores:   {len(players):,}")
print(f"   Appearances: {len(appearances):,}")
print(f"   Jogos:       {len(games):,}")
print(f"   Seleções WC: {len(wc_teams)}")

# ============================================================
# CÉLULA 2 — PRÉ-PROCESSAMENTO E FILTROS
# ============================================================

# RN1.1 — Apenas jogadores ativos em 2025
players = players[players['last_season'] == 2025].copy()
print(f"Jogadores ativos em 2025: {len(players):,}")

# Filtrar jogos do ciclo pós-Copa 2022 para convocação
games_recentes = games[games['season'] >= 2022][['game_id']].copy()

# Manter apenas appearances desses jogos
appearances = appearances.merge(games_recentes, on='game_id', how='inner')
print(f"Appearances (season >= 2022): {len(appearances):,}")

# ============================================================
# CÉLULA 3 — AGREGAR STATS POR JOGADOR
# ============================================================

stats = appearances.groupby('player_id').agg(
    goals         = ('goals',          'sum'),
    assists       = ('assists',        'sum'),
    minutes       = ('minutes_played', 'sum'),
    yellow_cards  = ('yellow_cards',   'sum'),
    red_cards     = ('red_cards',      'sum'),
    caps_recentes = ('appearance_id',  'count')
).reset_index()

print(f"Jogadores com stats agregadas: {len(stats):,}")

# ============================================================
# CÉLULA 4 — MONTAR PERFIL COMPLETO DO JOGADOR
# ============================================================

perfil = players.merge(stats, on='player_id', how='left')

cols_stats = ['goals','assists','minutes','yellow_cards','red_cards','caps_recentes']
perfil[cols_stats] = perfil[cols_stats].fillna(0)

perfil['gols_por_90'] = np.where(
    perfil['minutes'] > 0,
    (perfil['goals'] / perfil['minutes']) * 90,
    0
)

# Associar jogador à seleção via country_of_citizenship
perfil = perfil.merge(
    wc_teams[['national_team_name', 'national_team_id']],
    left_on  = 'country_of_citizenship',
    right_on = 'national_team_name',
    how      = 'left'
)

# Trazer ranking FIFA do nat_teams
perfil = perfil.merge(
    nat_teams[['national_team_id', 'country_name', 'fifa_ranking']],
    on  = 'national_team_id',
    how = 'left'
)

associados = perfil['national_team_id'].notna().sum()
print(f"Jogadores associados a seleções da Copa: {associados:,}")
print(f"Jogadores sem seleção mapeada: {len(perfil) - associados:,}")

# ============================================================
# CÉLULA 5 — FUNÇÃO DE CONVOCAÇÃO (RN1.3)
# ============================================================

def normalizar(serie):
    min_v, max_v = serie.min(), serie.max()
    if max_v == min_v:
        return pd.Series([0.5] * len(serie), index=serie.index)
    return (serie - min_v) / (max_v - min_v)


def convocar_selecao(national_team_id, perfil_df):
    elenco = perfil_df[perfil_df['national_team_id'] == national_team_id].copy()

    if len(elenco) == 0:
        print(f"⚠️  Sem jogadores para national_team_id={national_team_id}")
        return pd.DataFrame()

    elenco['valor_norm'] = normalizar(elenco['market_value_in_eur'].fillna(0))
    elenco['caps_norm']  = normalizar(elenco['international_caps'].fillna(0))

    elenco['score_convocacao'] = (
        0.6 * elenco['valor_norm'] +
        0.4 * elenco['caps_norm']
    )

    goleiros = elenco[elenco['sub_position'] == 'Goalkeeper'].copy()
    linha    = elenco[elenco['sub_position'] != 'Goalkeeper'].copy()

    goleiros['score_convocacao'] = (goleiros['valor_norm'] + goleiros['caps_norm']) / 2

    gks   = goleiros.nlargest(3,  'score_convocacao')
    linha = linha.nlargest(23, 'score_convocacao')

    convocados = pd.concat([gks, linha])
    convocados['national_team_id'] = national_team_id
    return convocados

# ============================================================
# CÉLULA 6 — CONVOCAR AS 48 SELEÇÕES
# ============================================================

todas_convocacoes = []

for _, row in wc_teams.iterrows():
    team_id   = row['national_team_id']
    team_name = row['national_team_name']

    convocados = convocar_selecao(team_id, perfil)

    if len(convocados) > 0:
        todas_convocacoes.append(convocados)
        print(f"✅ {team_name}: {len(convocados)} jogadores convocados")
    else:
        print(f"❌ {team_name}: sem dados suficientes")

base_convocados = pd.concat(todas_convocacoes, ignore_index=True)
print(f"\nTotal de jogadores convocados: {len(base_convocados)}")
print(f"Seleções com convocação: {base_convocados['national_team_id'].nunique()}")

# ============================================================
# CÉLULA 7 — AGREGAR ATRIBUTOS POR SELEÇÃO (RN1.4)
# ============================================================

# Calcular idade
base_convocados['date_of_birth'] = pd.to_datetime(
    base_convocados['date_of_birth'], errors='coerce'
)
hoje = pd.Timestamp('2025-06-01')
base_convocados['age'] = (
    (hoje - base_convocados['date_of_birth']).dt.days / 365.25
).round(1)

base_selecoes = base_convocados.groupby('national_team_id').agg(
    valor_total       = ('market_value_in_eur', 'sum'),
    idade_media       = ('age',                 'mean'),
    caps_total        = ('international_caps',  'sum'),
    gols_selecao      = ('international_goals', 'sum'),
    gols_clube        = ('goals',               'sum'),
    assists_clube     = ('assists',             'sum'),
    minutos_total     = ('minutes',             'sum'),
    gols_por_90_med   = ('gols_por_90',         'mean'),
    cartoes_amarelos  = ('yellow_cards',        'sum'),
    cartoes_vermelhos = ('red_cards',           'sum'),
).reset_index()

# Trazer ranking FIFA — usar o menor ranking do elenco
# (ranking menor = melhor posição FIFA)
ranking_por_selecao = perfil[perfil['national_team_id'].notna()].groupby(
    'national_team_id'
)['fifa_ranking'].first().reset_index()

base_selecoes = base_selecoes.merge(ranking_por_selecao, on='national_team_id', how='left')

# Trazer nome e grupo
base_selecoes = base_selecoes.merge(
    nat_teams[['national_team_id', 'country_name']],
    on='national_team_id', how='left'
)
base_selecoes = base_selecoes.merge(
    wc_teams[['national_team_id', 'national_team_name', 'team_group']],
    on='national_team_id', how='left'
)

print("✅ Base de seleções pronta!")
print(f"   Seleções: {len(base_selecoes)}")
print(base_selecoes[['national_team_name','valor_total','fifa_ranking']].head(10))

# ============================================================
# CÉLULA 8 — MONTAR BASE DE TREINO COM JOGOS HISTÓRICOS
# ============================================================

# ids das competições internacionais
competicoes_internacionais = ['AFAC', 'AFCN', 'COPA', 'EURO', 'FIWC']

# Sem filtro de season — todo o histórico disponível de jogos entre seleções
jogos = games[
    games['competition_id'].isin(competicoes_internacionais)
].copy()

print(f"Jogos internacionais (histórico completo): {len(jogos)}")

def calcular_resultado(row):
    if row['home_club_goals'] > row['away_club_goals']:
        return 0
    elif row['home_club_goals'] == row['away_club_goals']:
        return 1
    else:
        return 2

jogos['resultado'] = jogos.apply(calcular_resultado, axis=1)

print("\nDistribuição de resultados:")
print(jogos['resultado'].value_counts().rename({
    0: 'Vitória Home',
    1: 'Empate',
    2: 'Vitória Away'
}))

# ============================================================
# CÉLULA 9 — JUNTAR ATRIBUTOS DAS SELEÇÕES AOS JOGOS
# ============================================================

# Colunas de features incluindo fifa_ranking
feature_cols = [
    'valor_total', 'idade_media', 'caps_total', 'gols_selecao',
    'gols_clube', 'assists_clube', 'gols_por_90_med',
    'cartoes_amarelos', 'cartoes_vermelhos', 'fifa_ranking'
]

home_attrs = base_selecoes.rename(columns={
    'national_team_id'  : 'home_club_id',
    'valor_total'       : 'home_valor_total',
    'idade_media'       : 'home_idade_media',
    'caps_total'        : 'home_caps_total',
    'gols_selecao'      : 'home_gols_selecao',
    'gols_clube'        : 'home_gols_clube',
    'assists_clube'     : 'home_assists_clube',
    'gols_por_90_med'   : 'home_gols_por_90',
    'cartoes_amarelos'  : 'home_cartoes_amarelos',
    'cartoes_vermelhos' : 'home_cartoes_vermelhos',
    'fifa_ranking'      : 'home_fifa_ranking',
})

away_attrs = base_selecoes.rename(columns={
    'national_team_id'  : 'away_club_id',
    'valor_total'       : 'away_valor_total',
    'idade_media'       : 'away_idade_media',
    'caps_total'        : 'away_caps_total',
    'gols_selecao'      : 'away_gols_selecao',
    'gols_clube'        : 'away_gols_clube',
    'assists_clube'     : 'away_assists_clube',
    'gols_por_90_med'   : 'away_gols_por_90',
    'cartoes_amarelos'  : 'away_cartoes_amarelos',
    'cartoes_vermelhos' : 'away_cartoes_vermelhos',
    'fifa_ranking'      : 'away_fifa_ranking',
})

home_cols = ['home_club_id','home_valor_total','home_idade_media',
             'home_caps_total','home_gols_selecao','home_gols_clube',
             'home_assists_clube','home_gols_por_90','home_cartoes_amarelos',
             'home_cartoes_vermelhos','home_fifa_ranking']

away_cols = ['away_club_id','away_valor_total','away_idade_media',
             'away_caps_total','away_gols_selecao','away_gols_clube',
             'away_assists_clube','away_gols_por_90','away_cartoes_amarelos',
             'away_cartoes_vermelhos','away_fifa_ranking']

jogos_ml = jogos.merge(home_attrs[home_cols], on='home_club_id', how='inner')
jogos_ml = jogos_ml.merge(away_attrs[away_cols], on='away_club_id', how='inner')

print(f"Jogos com atributos completos: {len(jogos_ml)}")
print(f"Distribuição de resultados na base final:")
print(jogos_ml['resultado'].value_counts())

# ============================================================
# CÉLULA 10 — TREINO E COMPARATIVO DOS 3 MODELOS
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

features = [
    'home_valor_total', 'home_idade_media', 'home_caps_total',
    'home_gols_selecao', 'home_gols_clube', 'home_assists_clube',
    'home_gols_por_90', 'home_cartoes_amarelos', 'home_cartoes_vermelhos',
    'home_fifa_ranking',
    'away_valor_total', 'away_idade_media', 'away_caps_total',
    'away_gols_selecao', 'away_gols_clube', 'away_assists_clube',
    'away_gols_por_90', 'away_cartoes_amarelos', 'away_cartoes_vermelhos',
    'away_fifa_ranking',
]

X = jogos_ml[features]
y = jogos_ml['resultado']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Treino: {len(X_train)} jogos | Teste: {len(X_test)} jogos")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

modelos = {
    'Random Forest'      : RandomForestClassifier(
                               n_estimators=100,
                               class_weight='balanced',
                               random_state=42
                           ),
    'KNN'                : KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(
                               max_iter=1000,
                               class_weight='balanced',
                               random_state=42
                           ),
}

resultados = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train_sc, y_train)
    y_pred = modelo.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    resultados[nome] = {
        'modelo' : modelo,
        'y_pred' : y_pred,
        'acc'    : acc,
    }

    print(f"\n{'='*50}")
    print(f"  {nome} — Accuracy: {acc:.2%}")
    print(f"{'='*50}")
    print(classification_report(
        y_test, y_pred,
        target_names=['Vitória Home','Empate','Vitória Away'],
        zero_division=0
    ))

# ============================================================
# CÉLULA 11 — MATRIZES DE CONFUSÃO
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
labels = ['Vit. Home', 'Empate', 'Vit. Away']

for ax, (nome, res) in zip(axes, resultados.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_title(f"{nome}\nAccuracy: {res['acc']:.2%}")
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Real')

plt.suptitle('Comparativo de Modelos — Matriz de Confusão', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('comparativo_modelos.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico salvo!")

# ============================================================
# CÉLULA 12 — FASE DE GRUPOS COM RASTREAMENTO DE GOLS
# ============================================================

import time

# ------------------------------------------------------------
# SEED DO TORNEIO
# seed -> mantem a seed fixa em 42 - descomentar caso queria usar seed diferente
# ------------------------------------------------------------
#SEED = int(time.time()) % (2**32 - 1)
SEED = 42
np.random.seed(SEED)
print(f"🎲 Seed do torneio: {SEED}")

# Temperatura da distribuição de gols/assistências.
TEMPERATURA_GOLS = 0.5
nome_modelo_final = max(resultados, key=lambda nome: resultados[nome]['acc'])
modelo_final = resultados[nome_modelo_final]['modelo']
print(f"🏆 Modelo final selecionado: {nome_modelo_final} (acurácia: {resultados[nome_modelo_final]['acc']:.2%})")

# Rastreamento individual
gols_por_jogador    = {}
assists_por_jogador = {}
gols_sofridos_gk    = {}

for _, jogador in base_convocados.iterrows():
    pid = jogador['player_id']
    gols_por_jogador[pid]    = 0
    assists_por_jogador[pid] = 0
    if jogador['sub_position'] == 'Goalkeeper':
        gols_sofridos_gk[pid] = {'gols_sofridos': 0, 'jogos': 0}

peso_posicao = {
    'Centre-Forward'    : 1.00,
    'Second Striker'    : 0.85,
    'Left Winger'       : 0.75,
    'Right Winger'      : 0.75,
    'Attacking Midfield': 0.60,
    'Left Midfield'     : 0.45,
    'Right Midfield'    : 0.45,
    'Central Midfield'  : 0.35,
    'Defensive Midfield': 0.15,
    'Centre-Back'       : 0.10,
    'Left-Back'         : 0.10,
    'Right-Back'        : 0.10,
}


def calcular_score_ofensivo(elenco_df):
    elenco = elenco_df.copy()
    elenco['peso_pos']      = elenco['sub_position'].map(peso_posicao).fillna(0.05)
    elenco['gols_sel_norm'] = normalizar(elenco['international_goals'].fillna(0))
    elenco['gols_90_norm']  = normalizar(elenco['gols_por_90'].fillna(0))
    elenco['pos_norm']      = normalizar(elenco['peso_pos'])
    elenco['score_ofensivo'] = (
        elenco['gols_sel_norm'] * 0.40 +
        elenco['gols_90_norm']  * 0.35 +
        elenco['pos_norm']      * 0.25
    )
    return elenco


def calcular_score_assistencia(elenco_df):
    elenco = elenco_df.copy()
    elenco['peso_pos']     = elenco['sub_position'].map(peso_posicao).fillna(0.05)
    elenco['assists_norm'] = normalizar(elenco['assists'].fillna(0))
    elenco['gols_90_norm'] = normalizar(elenco['gols_por_90'].fillna(0))
    elenco['pos_norm']     = normalizar(elenco['peso_pos'])
    elenco['score_assist'] = (
        elenco['assists_norm'] * 0.50 +
        elenco['gols_90_norm'] * 0.30 +
        elenco['pos_norm']     * 0.20
    )
    return elenco


def distribuir_gols_assists(team_id, n_gols, base_convocados):
    elenco = base_convocados[
        (base_convocados['national_team_id'] == team_id) &
        (base_convocados['sub_position'] != 'Goalkeeper')
    ].copy()

    if elenco.empty or n_gols == 0:
        return

    elenco_of  = calcular_score_ofensivo(elenco)
    elenco_ast = calcular_score_assistencia(elenco)

    scores_of  = elenco_of['score_ofensivo'].values
    scores_ast = elenco_ast['score_assist'].values
    pids       = elenco_of['player_id'].values

    # Temperatura achata a distribuição: as estrelas seguem favoritas,
    # mas jogadores medianos passam a ter chance real de marcar/assistir.
    probs_of  = np.power(scores_of  + 1e-6, TEMPERATURA_GOLS)
    probs_ast = np.power(scores_ast + 1e-6, TEMPERATURA_GOLS)
    probs_of  /= probs_of.sum()
    probs_ast /= probs_ast.sum()

    for _ in range(n_gols):
        marcador = np.random.choice(pids, p=probs_of)
        gols_por_jogador[marcador] += 1

        if np.random.random() < 0.80 and len(pids) > 1:
            mask       = pids != marcador
            pids_ast   = pids[mask]
            probs_ast2 = probs_ast[mask]
            probs_ast2 = probs_ast2 / probs_ast2.sum()
            assistente = np.random.choice(pids_ast, p=probs_ast2)
            assists_por_jogador[assistente] += 1


def registrar_goleiro(team_id, gols_sofridos, base_convocados):
    goleiros = base_convocados[
        (base_convocados['national_team_id'] == team_id) &
        (base_convocados['sub_position'] == 'Goalkeeper')
    ].copy()

    if goleiros.empty:
        return

    titular_pid = goleiros.nlargest(1, 'score_convocacao')['player_id'].values[0]
    if titular_pid in gols_sofridos_gk:
        gols_sofridos_gk[titular_pid]['gols_sofridos'] += gols_sofridos
        gols_sofridos_gk[titular_pid]['jogos']         += 1


def prever_jogo(home_id, away_id, base_selecoes, modelo, scaler):
    home = base_selecoes[base_selecoes['national_team_id'] == home_id]
    away = base_selecoes[base_selecoes['national_team_id'] == away_id]

    if home.empty or away.empty:
        return 1

    jogo = pd.DataFrame([{
        'home_valor_total'      : home['valor_total'].values[0],
        'home_idade_media'      : home['idade_media'].values[0],
        'home_caps_total'       : home['caps_total'].values[0],
        'home_gols_selecao'     : home['gols_selecao'].values[0],
        'home_gols_clube'       : home['gols_clube'].values[0],
        'home_assists_clube'    : home['assists_clube'].values[0],
        'home_gols_por_90'      : home['gols_por_90_med'].values[0],
        'home_cartoes_amarelos' : home['cartoes_amarelos'].values[0],
        'home_cartoes_vermelhos': home['cartoes_vermelhos'].values[0],
        'home_fifa_ranking'     : home['fifa_ranking'].values[0],
        'away_valor_total'      : away['valor_total'].values[0],
        'away_idade_media'      : away['idade_media'].values[0],
        'away_caps_total'       : away['caps_total'].values[0],
        'away_gols_selecao'     : away['gols_selecao'].values[0],
        'away_gols_clube'       : away['gols_clube'].values[0],
        'away_assists_clube'    : away['assists_clube'].values[0],
        'away_gols_por_90'      : away['gols_por_90_med'].values[0],
        'away_cartoes_amarelos' : away['cartoes_amarelos'].values[0],
        'away_cartoes_vermelhos': away['cartoes_vermelhos'].values[0],
        'away_fifa_ranking'     : away['fifa_ranking'].values[0],
    }])

    jogo_sc = scaler.transform(jogo)
    return modelo.predict(jogo_sc)[0]


def desempate_mata_mata(home_id, away_id, base_selecoes):
    home = base_selecoes[base_selecoes['national_team_id'] == home_id].iloc[0]
    away = base_selecoes[base_selecoes['national_team_id'] == away_id].iloc[0]

    # PÊNALTIS: decididos majoritariamente pela sorte.
    # A seleção melhor ranqueada (colocao no ranking FIFA) leva uma vantagem (55/45)
    # Esta função só é chamada no mata-mata (jogo eliminatório que empatou).
    r_home = home['fifa_ranking']
    r_away = away['fifa_ranking']

    if pd.isna(r_home) or pd.isna(r_away) or r_home == r_away:
        prob_home = 0.50
    else:
        prob_home = 0.55 if r_home < r_away else 0.45

    return home_id if np.random.random() < prob_home else away_id


def estimar_gols(home_id, away_id, resultado, base_selecoes):
    home = base_selecoes[base_selecoes['national_team_id'] == home_id].iloc[0]
    away = base_selecoes[base_selecoes['national_team_id'] == away_id].iloc[0]

    max_valor = base_selecoes['valor_total'].max()
    max_caps  = base_selecoes['caps_total'].max()

    forca_home = (home['valor_total'] / max_valor) * 0.6 + (home['caps_total'] / max_caps) * 0.4
    forca_away = (away['valor_total'] / max_valor) * 0.6 + (away['caps_total'] / max_caps) * 0.4
    diferenca  = abs(forca_home - forca_away)

    if diferenca > 0.5:
        gols_vencedor = np.random.randint(3, 6)
        gols_perdedor = np.random.randint(0, 2)
    elif diferenca > 0.3:
        gols_vencedor = np.random.randint(2, 4)
        gols_perdedor = np.random.randint(0, 2)
    elif diferenca > 0.15:
        gols_vencedor = np.random.randint(2, 3)
        gols_perdedor = np.random.randint(0, 2)
    else:
        gols_vencedor = np.random.randint(1, 3)
        gols_perdedor = np.random.randint(0, 2)

    # Jogo decidido pelo modelo NUNCA pode terminar empatado no placar.
    # Em partidas equilibradas o sorteio podia gerar, ex., 1 e 1; aqui
    # garantimos que o perdedor leve no máximo (gols_vencedor - 1).
    gols_perdedor = min(gols_perdedor, gols_vencedor - 1)

    if resultado == 0:
        return gols_vencedor, gols_perdedor
    elif resultado == 2:
        return gols_perdedor, gols_vencedor
    else:
        gols = np.random.randint(0, 3)
        return gols, gols


def simular_partida(home_id, away_id, base_selecoes, modelo, scaler, exibir=True):
    resultado = prever_jogo(home_id, away_id, base_selecoes, modelo, scaler)
    g_home, g_away = estimar_gols(home_id, away_id, resultado, base_selecoes)

    home_name = wc_teams[wc_teams['national_team_id'] == home_id]['national_team_name'].values[0]
    away_name = wc_teams[wc_teams['national_team_id'] == away_id]['national_team_name'].values[0]

    if exibir:
        print(f"  {home_name} {g_home} x {g_away} {away_name}")

    distribuir_gols_assists(home_id, g_home, base_convocados)
    distribuir_gols_assists(away_id, g_away, base_convocados)
    registrar_goleiro(home_id, g_away, base_convocados)
    registrar_goleiro(away_id, g_home, base_convocados)

    if resultado == 0:
        vencedor = home_id
    elif resultado == 2:
        vencedor = away_id
    else:
        vencedor = None
    
    return resultado, g_home, g_away, vencedor

def simular_confronto_mata_mata(home_id, away_id, base_selecoes, modelo, scaler):
    resultado, g_home, g_away, vencedor = simular_partida(
        home_id, away_id, base_selecoes, modelo, scaler, exibir=False
    )
 
    home_name = wc_teams[wc_teams['national_team_id'] == home_id]['national_team_name'].values[0]
    away_name = wc_teams[wc_teams['national_team_id'] == away_id]['national_team_name'].values[0]
 
    if vencedor is None:
        vencedor = desempate_mata_mata(home_id, away_id, base_selecoes)
        vencedor_name = wc_teams[wc_teams['national_team_id'] == vencedor]['national_team_name'].values[0]
        print(f"  {home_name} {g_home} x {g_away} {away_name} → Desempate: {vencedor_name} ✅")
    elif vencedor == home_id:
        print(f"  {home_name} ✅ {g_home} x {g_away} {away_name} ❌")
    else:
        print(f"  {home_name} ❌ {g_home} x {g_away} {away_name} ✅")
 
    return vencedor

# ============================================================
# CÉLULA 13 — CLASSIFICAÇÃO
# ============================================================
# ============================================================
# CÉLULA 13 — CLASSIFICAÇÃO FIXA DOS GRUPOS
# Lê a posição final de cada seleção diretamente da coluna
# 'group_position' de wc_teams (1, 2, 3 ou vazio = eliminado na
# fase de grupos). Não há simulação de jogos de grupo nesta versão
# — a fase de grupos é tratada como um dado de entrada fixo.
# ============================================================

wc_teams['group_position'] = pd.to_numeric(wc_teams['group_position'], errors='coerce')
wc_teams['team_group']     = wc_teams['team_group'].astype(str).str.strip()

pos_por_grupo        = {}   # {'A': {1: id_1º, 2: id_2º, 3: id_3º}, ...}
terceiros_por_grupo  = {}   # {'B': id_3º, 'D': id_3º, ...} — só os grupos com 3º classificado

for grupo in sorted(wc_teams['team_group'].dropna().unique()):
    grupo_df  = wc_teams[wc_teams['team_group'] == grupo]
    posicoes  = {}
    for _, row in grupo_df.iterrows():
        pos = row['group_position']
        if pd.notna(pos):
            posicoes[int(pos)] = row['national_team_id']
    pos_por_grupo[grupo] = posicoes
    if 3 in posicoes:
        terceiros_por_grupo[grupo] = posicoes[3]

# --- Validação: 24 classificados diretos (1º/2º) + 8 melhores terceiros = 32 ---
diretos       = sum(1 for posicoes in pos_por_grupo.values() for p in posicoes if p in (1, 2))
terceiros_qtd = len(terceiros_por_grupo)

print("=" * 60)
print("CLASSIFICAÇÃO FIXA — FASE DE GRUPOS (Copa 2026)")
print("=" * 60)

for grupo, posicoes in pos_por_grupo.items():
    print(f"\nGrupo {grupo}")
    for pos in sorted(posicoes):
        nome = wc_teams[wc_teams['national_team_id'] == posicoes[pos]]['national_team_name'].values[0]
        marcador = "✅" if pos <= 2 or grupo in terceiros_por_grupo else "❌"
        print(f"  {pos}º {nome:25s} {marcador}")

print(f"\n🔎 Classificados diretos (1º/2º): {diretos}")
print(f"🔎 Melhores terceiros classificados ({terceiros_qtd}): {sorted(terceiros_por_grupo.keys())}")

total_classificados = diretos + terceiros_qtd
print(f"\n✅ Total de classificados para o mata-mata: {total_classificados}")

assert total_classificados == 32, (
    f"Esperado 32 classificados (24 diretos + 8 terceiros), mas encontrei "
    f"{total_classificados}. Confira a coluna 'group_position' em wc_teams."
)


# ============================================================
# CÉLULA 14 — MATA-MATA (chaveamento fixo da Copa 2026)
# Baseado na estrutura oficial de posições por lado do chaveamento.
# A fase de grupos é fixa (Célula 13); apenas o mata-mata é simulado.
# ============================================================

np.random.seed(SEED)

# chaveamento do mata-mata conforme 
CHAVEAMENTO_LADO_A = [
    ('2B', '2A'), ('1F', '2C'), ('1E', '3D'), ('1I', '3F'), 
    ('1G', '3I'), ('1D', '3B'), ('2K', '2L'), ('1H', '2J')
]

CHAVEAMENTO_LADO_B = [
    ('1C', '2F'), ('2E', '2I'), ('1L', '3K'),  ('1A', '3E'),   
    ('1B', '3J'), ('1K', '3L'), ('2D', '2G'), ('1J', '2H')
]


def resolver_posicao(posicao, pos_por_grupo):
    """Converte um rótulo tipo '1A', '2C', '3F' no national_team_id
    correspondente, usando a classificação fixa dos grupos."""
    pos_num = int(posicao[0])
    grupo   = posicao[1:]
    if grupo not in pos_por_grupo or pos_num not in pos_por_grupo[grupo]:
        raise KeyError(
            f"Posição '{posicao}' não encontrada na classificação "
            f"(grupo='{grupo}', posição={pos_num}). Confira group_position em wc_teams."
        )
    return pos_por_grupo[grupo][pos_num]


def simular_lado(confrontos_fixos, pos_por_grupo, base_selecoes, modelo, scaler, nome_lado):
    """Simula um lado inteiro do chaveamento (16-avos → oitavas →
    quartas → semifinal daquele lado). Retorna o par final do lado
    (os 2 times que disputam a vaga na grande final)."""
    rodada_atual = [
        (resolver_posicao(p1, pos_por_grupo), resolver_posicao(p2, pos_por_grupo))
        for p1, p2 in confrontos_fixos
    ]

    nomes_fase = {8: "Dezesseis-avos", 4: "Oitavas", 2: "Quartas", 1: "Semifinal"}

    while len(rodada_atual) > 1:
        fase = nomes_fase.get(len(rodada_atual), f"Rodada ({len(rodada_atual)} jogos)")
        print(f"\n--- Lado {nome_lado} — {fase} ---")
        vencedores = [
            simular_confronto_mata_mata(h, a, base_selecoes, modelo, scaler)
            for h, a in rodada_atual
        ]
        rodada_atual = list(zip(vencedores[0::2], vencedores[1::2]))

    return rodada_atual[0]


print("=" * 60)
print("  MATA-MATA — LADO A")
print("=" * 60)
confronto_final_a = simular_lado(CHAVEAMENTO_LADO_A, pos_por_grupo, base_selecoes, modelo_final, scaler, "A")

print(f"\n--- Lado A — Semi Final do lado ---")
finalista_a = simular_confronto_mata_mata(*confronto_final_a, base_selecoes, modelo_final, scaler)

print(f"\n{'='*60}")
print("  MATA-MATA — LADO B")
print("=" * 60)
confronto_final_b = simular_lado(CHAVEAMENTO_LADO_B, pos_por_grupo, base_selecoes, modelo_final, scaler, "B")

print(f"\n--- Lado B — Semi Final do lado ---")
finalista_b = simular_confronto_mata_mata(*confronto_final_b, base_selecoes, modelo_final, scaler)

print(f"\n{'='*60}")
print("  🏆 GRANDE FINAL")
print("=" * 60)
campeao_id = simular_confronto_mata_mata(finalista_a, finalista_b, base_selecoes, modelo_final, scaler)
vice_id    = finalista_b if campeao_id == finalista_a else finalista_a

print(f"\n  🥇 CAMPEÃO: {wc_teams[wc_teams['national_team_id']==campeao_id]['national_team_name'].values[0]}")
print(f"  🥈 VICE:    {wc_teams[wc_teams['national_team_id']==vice_id]['national_team_name'].values[0]}")
finalistas_ids = [campeao_id, vice_id]

# ============================================================
# CÉLULA 15 — PRÊMIOS INDIVIDUAIS
# ============================================================

# Converter rastreamentos para DataFrame
df_gols = pd.DataFrame([
    {'player_id': pid, 'gols_torneio': gols}
    for pid, gols in gols_por_jogador.items()
])

df_assists = pd.DataFrame([
    {'player_id': pid, 'assists_torneio': assists}
    for pid, assists in assists_por_jogador.items()
])

df_gks = pd.DataFrame([
    {
        'player_id'     : pid,
        'gols_sofridos' : v['gols_sofridos'],
        'jogos'         : v['jogos'],
    }
    for pid, v in gols_sofridos_gk.items()
])

# Juntar com dados dos jogadores
base_premios = base_convocados.merge(df_gols,   on='player_id', how='left')
base_premios = base_premios.merge(df_assists,   on='player_id', how='left')
base_premios['gols_torneio']    = base_premios['gols_torneio'].fillna(0).astype(int)
base_premios['assists_torneio'] = base_premios['assists_torneio'].fillna(0).astype(int)

def get_nome_pais(team_id):
    r = wc_teams[wc_teams['national_team_id'] == team_id]['national_team_name']
    return r.values[0] if len(r) > 0 else 'Desconhecido'

# ============================================================
# ARTILHEIRO
# ============================================================

artilheiro = base_premios[
    base_premios['sub_position'] != 'Goalkeeper'
].nlargest(1, ['gols_torneio', 'assists_torneio']).iloc[0]

artilheiro_pais = get_nome_pais(artilheiro['national_team_id'])

print("=" * 60)
print("ARTILHEIRO DO TORNEIO")
print("=" * 60)
print(f"  {artilheiro['name']} ({artilheiro_pais})")
print(f"  Gols no torneio: {artilheiro['gols_torneio']}")

# ============================================================
# LUVA DE OURO
# Goleiro com menor média de gols sofridos por jogo
# Ponderado por jogos disputados (quem jogou mais fases vale mais)
# ============================================================

goleiros_premios = base_premios[
    base_premios['sub_position'] == 'Goalkeeper'
].merge(df_gks, on='player_id', how='left')

goleiros_premios['gols_sofridos'] = goleiros_premios['gols_sofridos'].fillna(0)
goleiros_premios['jogos']         = goleiros_premios['jogos'].fillna(0).astype(int)

# Apenas goleiros que jogaram ao menos 1 jogo
goleiros_ativos = goleiros_premios[goleiros_premios['jogos'] > 0].copy()

# Média gols sofridos por jogo (menor = melhor)
goleiros_ativos['media_gols_sofridos'] = (
    goleiros_ativos['gols_sofridos'] / goleiros_ativos['jogos']
)

# Score Luva: prioriza quem jogou mais jogos e sofreu menos gols
# Mais jogos = chegou mais longe = peso maior
max_jogos = goleiros_ativos['jogos'].max()
goleiros_ativos['score_luva'] = (
    (goleiros_ativos['jogos'] / max_jogos) * 0.5 +           # longevidade
    (1 - normalizar(goleiros_ativos['media_gols_sofridos'])) * 0.5  # menos gols sofridos
)

luva_de_ouro     = goleiros_ativos.nlargest(1, 'score_luva').iloc[0]
luva_pais        = get_nome_pais(luva_de_ouro['national_team_id'])

print(f"\n{'='*60}")
print("LUVA DE OURO — Melhor Goleiro")
print("=" * 60)
print(f"  {luva_de_ouro['name']} ({luva_pais})")
print(f"  Jogos: {int(luva_de_ouro['jogos'])} | "
      f"Gols sofridos: {int(luva_de_ouro['gols_sofridos'])} | "
      f"Média/jogo: {luva_de_ouro['media_gols_sofridos']:.2f}")

# ============================================================
# GARÇOM — Maior número de assistências no torneio
# ============================================================

garcom = base_premios[
    base_premios['sub_position'] != 'Goalkeeper'
].nlargest(1, ['assists_torneio', 'gols_torneio']).iloc[0]

garcom_pais = get_nome_pais(garcom['national_team_id'])

print(f"\n{'='*60}")
print("GARÇOM — Maior número de assistências")
print("=" * 60)
print(f"  {garcom['name']} ({garcom_pais})")
print(f"  Assistências no torneio: {garcom['assists_torneio']}")

# ============================================================
# REVELAÇÃO — Melhor jogador sub-21
# Score: 55% gols_torneio + 35% assists_torneio + 10% valor de mercado
# (valor de mercado reduzido para depender mais do desempenho no torneio)
# ============================================================

base_premios['date_of_birth'] = pd.to_datetime(
    base_premios['date_of_birth'], errors='coerce'
)
hoje = pd.Timestamp('2025-06-01')
base_premios['age'] = (
    (hoje - base_premios['date_of_birth']).dt.days / 365.25
)

jovens = base_premios[
    (base_premios['age'] <= 21) &
    (base_premios['sub_position'] != 'Goalkeeper')
].copy()

if len(jovens) > 0:
    jovens['score_revelacao'] = (
        normalizar(jovens['gols_torneio'])              * 0.55 +
        normalizar(jovens['assists_torneio'])           * 0.35 +
        normalizar(jovens['market_value_in_eur'].fillna(0)) * 0.10
    )

    revelacao      = jovens.nlargest(1, 'score_revelacao').iloc[0]
    revelacao_pais = get_nome_pais(revelacao['national_team_id'])

    print(f"\n{'='*60}")
    print("REVELAÇÃO — Melhor jogador sub-21")
    print("=" * 60)
    print(f"  {revelacao['name']} ({revelacao_pais})")
    print(f"  Idade: {revelacao['age']:.1f} anos | "
          f"Gols: {revelacao['gols_torneio']} | "
          f"Assistências: {revelacao['assists_torneio']}")
else:
    print("\n⚠️  Nenhum jogador sub-21 encontrado na base.")

# ============================================================
# BOLA DE OURO — Melhor do torneio
# Obrigatoriamente do campeão ou finalista (RN3.5)
# Score por posição (RN3.1): normalização separada por grupo
# ============================================================

finalistas_elenco = base_premios[
    base_premios['national_team_id'].isin(finalistas_ids)
].copy()

posicoes_grupos = {
    'Goleiro'  : ['Goalkeeper'],
    'Defensor' : ['Centre-Back', 'Left-Back', 'Right-Back'],
    'Meio'     : ['Central Midfield', 'Defensive Midfield',
                  'Attacking Midfield', 'Left Midfield', 'Right Midfield'],
    'Atacante' : ['Left Winger', 'Right Winger',
                  'Centre-Forward', 'Second Striker'],
}

# Fator de dificuldade por posição para a Bola de Ouro.
# Como a normalização é feita DENTRO de cada grupo, sem isto o melhor
# goleiro/zagueiro fica empatado com o melhor atacante. O fator reduz o
# score de goleiros e defensores: eles ainda podem ganhar, mas precisam
# ser MUITO mais dominantes na sua posição do que um atacante.
fator_posicao_bola = {
    'Atacante': 1.00,
    'Meio'    : 0.92,
    'Defensor': 0.65,
    'Goleiro' : 0.50,
}

grupos_scored = []
for grupo_nome, posicoes in posicoes_grupos.items():
    subset = finalistas_elenco[
        finalistas_elenco['sub_position'].isin(posicoes)
    ].copy()

    if len(subset) == 0:
        continue

    # Normalização dentro do grupo de posição (RN3.1)
    subset['gols_norm']    = normalizar(subset['gols_torneio'])
    subset['assists_norm'] = normalizar(subset['assists_torneio'])
    subset['valor_norm']   = normalizar(subset['market_value_in_eur'].fillna(0))

    if grupo_nome == 'Goleiro':
        # Goleiro: longevidade + gols sofridos
        subset = subset.merge(df_gks, on='player_id', how='left')
        subset['jogos']         = subset['jogos'].fillna(0)
        subset['gols_sofridos'] = subset['gols_sofridos'].fillna(0)
        subset['score_bola'] = (
            (subset['jogos'] / max_jogos)                          * 0.60 +
            (1 - normalizar(subset['gols_sofridos'].fillna(0)))    * 0.40
        )
    elif grupo_nome == 'Defensor':
        # Defensor: desempenho acima de valor (RN3.1 normalização por posição)
        subset['score_bola'] = (
            subset['gols_norm']    * 0.45 +
            subset['assists_norm'] * 0.35 +
            subset['valor_norm']   * 0.20
        )
    else:
        # Meio e Atacante: gols + assistências dominam, valor só desempata
        subset['score_bola'] = (
            subset['gols_norm']    * 0.55 +
            subset['assists_norm'] * 0.35 +
            subset['valor_norm']   * 0.10
        )

    # Penaliza posições defensivas (goleiro/defensor têm mais dificuldade)
    subset['score_bola'] *= fator_posicao_bola[grupo_nome]

    grupos_scored.append(subset)

finalistas_scored = pd.concat(grupos_scored, ignore_index=True)
bola_de_ouro      = finalistas_scored.nlargest(1, 'score_bola').iloc[0]
bola_pais         = get_nome_pais(bola_de_ouro['national_team_id'])

print(f"\n{'='*60}")
print("BOLA DE OURO — Melhor jogador do torneio")
print("=" * 60)
print(f"  {bola_de_ouro['name']} ({bola_pais})")
print(f"  Posição: {bola_de_ouro['sub_position']} | "
      f"Gols: {bola_de_ouro['gols_torneio']} | "
      f"Assistências: {bola_de_ouro['assists_torneio']}")

