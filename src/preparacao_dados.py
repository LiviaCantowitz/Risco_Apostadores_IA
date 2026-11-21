"""
Livia Negrucci Cantowitz RA 10389419 

Funções de preparação dos dados.
"""


from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

VARS_REGRA = ["n_bets", "avg_stake", "pct_multiple", "max_losing_streak", "total_ggr"]

ATRIBUTOS = [
    "n_bets", "total_stake", "avg_stake", "max_stake", "win_rate",
    "total_gain", "total_ggr", "avg_odds", "pct_multiple", "max_losing_streak",
]

PERFIS = ["baixo", "medio", "alto"]


def carregar_apostas(caminho=RAIZ / "bets.csv") -> pd.DataFrame:
    """Lê o dataset original e ordena as apostas pelo número do bet_id
    (proxy de ordem cronológica, já que o dataset não tem timestamp)."""
    bets = pd.read_csv(caminho, sep=";")
    bets["ordem"] = bets["bet_id"].str.replace("bet_", "", regex=False).astype(int)
    return bets.sort_values("ordem").reset_index(drop=True)


def _maior_sequencia_derrotas(vitorias) -> int:
    maior = atual = 0
    for venceu in vitorias:
        atual = 0 if venceu else atual + 1
        maior = max(maior, atual)
    return maior


def agregar_por_apostador(bets: pd.DataFrame) -> pd.DataFrame:
    """Agrega apostas (nível transação) em atributos por apostador."""
    bets = bets.sort_values("ordem")
    g = bets.groupby("user_id")
    df = pd.DataFrame({
        "n_bets": g.size(),
        "total_stake": g["stake"].sum(),
        "avg_stake": g["stake"].mean(),
        "max_stake": g["stake"].max(),
        "win_rate": g["is_win"].mean(),
        "total_gain": g["gain"].sum(),
        "total_ggr": g["GGR"].sum(),
        "avg_odds": g["odds"].mean(),
        "pct_multiple": g["bet_type"].apply(lambda s: (s == "multiple").mean()),
        "max_losing_streak": g["is_win"].apply(_maior_sequencia_derrotas),
    })
    return df.reset_index()


def historico_parcial(bets: pd.DataFrame, fracao: float) -> pd.DataFrame:
    """Mantém, para cada apostador, apenas as primeiras `fracao` das suas apostas
    (mínimo de 1 aposta)."""
    bets = bets.sort_values("ordem").copy()
    bets["pos"] = bets.groupby("user_id").cumcount()
    total = bets.groupby("user_id")["ordem"].transform("size")
    limite = np.maximum(1, np.floor(total * fracao)).astype(int)
    return bets[bets["pos"] < limite].drop(columns="pos")


def calcular_risk_score(df: pd.DataFrame, referencia: pd.DataFrame | None = None) -> pd.Series:
    """Soma dos z-scores das variáveis de risco. Média e desvio vêm de `referencia`
    (por padrão, o próprio df)."""
    ref = df if referencia is None else referencia
    score = 0
    for v in VARS_REGRA:
        score = score + (df[v] - ref[v].mean()) / ref[v].std(ddof=0)
    return score


def rotular_perfil(score: pd.Series, limites=None):
    """Divide o score em tercis (baixo/médio/alto). Se `limites` for informado
    (dois cortes), usa esses cortes em vez dos tercis do próprio score."""
    if limites is None:
        rotulos, limites = pd.qcut(score, 3, labels=PERFIS, retbins=True)
        return rotulos.astype(str), limites[1:3]
    rotulos = pd.cut(score, [-np.inf, limites[0], limites[1], np.inf], labels=PERFIS)
    return rotulos.astype(str), limites


if __name__ == "__main__":
    apostas = carregar_apostas()
    base = agregar_por_apostador(apostas)
    base["risk_score"] = calcular_risk_score(base)
    base["risk_profile"], _ = rotular_perfil(base["risk_score"])
    print(base["risk_profile"].value_counts())
