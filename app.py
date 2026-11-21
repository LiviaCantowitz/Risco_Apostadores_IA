"""
Projeto: Identificação de Perfis de Risco em Apostadores Online por Meio de Aprendizado de Máquina
Integrante: Livia Negrucci Cantowitz  RA 10389419 

Síntese: protótipo interativo. A partir do padrão de apostas recentes de um apostador,
estima o perfil de risco que ele tende a atingir, mostra as probabilidades,
os fatores que mais pesaram na previsão e uma orientação de proteção.
Usa o modelo treinado em 02_modelagem_avaliacao.ipynb.
Execução: streamlit run app.py

Histórico de alterações:
    20/11/2026 – Livia Negrucci Cantowitz - tudo
"""
    

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
from preparacao_dados import (ATRIBUTOS, PERFIS, agregar_por_apostador,  # noqa: E402
                              calcular_risk_score, carregar_apostas,
                              historico_parcial, rotular_perfil)

NOMES = {
    "n_bets": "Nº de apostas", "total_stake": "Total apostado", "avg_stake": "Valor médio apostado",
    "max_stake": "Maior aposta", "win_rate": "Taxa de vitória", "total_gain": "Total recebido",
    "total_ggr": "Prejuízo líquido", "avg_odds": "Odds médias", "pct_multiple": "% de apostas múltiplas",
    "max_losing_streak": "Maior sequência de derrotas",
}
CORES = {"baixo": "#2e7d32", "medio": "#f9a825", "alto": "#c62828"}
ROTULO = {"baixo": "BAIXO", "medio": "MÉDIO", "alto": "ALTO"}
ORIENTACAO = {
    "baixo": "Padrão de apostas sem sinais relevantes de risco. Manter limites de gasto definidos ajuda a continuar assim.",
    "medio": "Alguns sinais de atenção. Sugestões: definir limites de depósito e de tempo, e acompanhar o próprio histórico.",
    "alto": "Padrão associado a risco elevado. Sugestões: ativar limites ou autoexclusão na plataforma e buscar apoio "
            "especializado (ex.: CAPS, Jogadores Anônimos). Ninguém precisa enfrentar isso sozinho.",
}


@st.cache_resource
def carregar_modelo():
    meta = json.load(open(RAIZ / "modelos" / "metadados.json", encoding="utf-8"))
    modelo = joblib.load(RAIZ / "modelos" / "modelo_deteccao_precoce.joblib")
    return modelo, meta


@st.cache_data
def carregar_dados(fracao):
    apostas = carregar_apostas(RAIZ / "bets.csv")
    completo = agregar_por_apostador(apostas)
    completo["perfil_final"], _ = rotular_perfil(calcular_risk_score(completo))
    parcial = agregar_por_apostador(historico_parcial(apostas, fracao)).set_index("user_id")
    return apostas, completo.set_index("user_id"), parcial


def contribuicoes(modelo, x, classe):
    """Contribuição de cada atributo para o log-odds da classe (modelo linear: coef × valor padronizado)."""
    if not hasattr(modelo, "named_steps") or "logisticregression" not in modelo.named_steps:
        return None
    z = modelo.named_steps["standardscaler"].transform(x)[0]
    coef = modelo.named_steps["logisticregression"].coef_[classe]
    return pd.Series(coef * z, index=ATRIBUTOS).sort_values(key=np.abs, ascending=False)


def mostrar_resultado(modelo, x, perfil_real=None):
    proba = modelo.predict_proba(x[ATRIBUTOS])[0]
    i = int(np.argmax(proba))
    perfil = PERFIS[i]

    st.markdown(
        f"<div style='padding:16px;border-radius:10px;background:{CORES[perfil]}22;border-left:8px solid {CORES[perfil]}'>"
        f"<span style='font-size:15px'>Perfil de risco estimado</span><br>"
        f"<span style='font-size:34px;font-weight:700;color:{CORES[perfil]}'>{ROTULO[perfil]}</span></div>",
        unsafe_allow_html=True)
    st.write("")
    if perfil_real is not None:
        ok = "✅ acertou" if perfil_real == perfil else "❌ errou"
        st.info(f"Perfil real ao final do histórico completo: **{ROTULO[perfil_real]}** ({ok})")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Probabilidades")
        st.bar_chart(pd.DataFrame({"probabilidade": proba}, index=[ROTULO[p] for p in PERFIS]))
    with c2:
        st.subheader(f"O que pesou para '{ROTULO[perfil]}'")
        contrib = contribuicoes(modelo, x[ATRIBUTOS], i)
        if contrib is not None:
            top = contrib.head(6).rename(index=NOMES)
            st.bar_chart(top.rename("contribuição"), horizontal=True)
            st.caption("Barras positivas aumentam a chance deste perfil; negativas diminuem.")

    st.subheader("Orientação")
    st.write(ORIENTACAO[perfil])


def montar_atributos(n, media, maior, vitorias, odds, multiplas, seq, resultado):
    total = n * media
    return pd.DataFrame([{
        "n_bets": n, "total_stake": total, "avg_stake": media, "max_stake": max(maior, media),
        "win_rate": vitorias, "total_gain": max(total + resultado, 0), "total_ggr": -resultado,
        "avg_odds": odds, "pct_multiple": multiplas, "max_losing_streak": seq,
    }])


st.set_page_config(page_title="Perfil de Risco em Apostas", page_icon="🎲", layout="wide")
modelo, meta = carregar_modelo()
apostas, completo, parcial = carregar_dados(meta["fracao_historico"])

st.title("🎲 Identificação de Perfis de Risco em Apostadores Online")
st.caption(f"Modelo: {meta['modelo']} · detecção precoce com {int(meta['fracao_historico'] * 100)}% do histórico · "
           f"F1-macro no teste: {meta['f1_macro_teste']:.3f} · Projeto de IA – Mackenzie – Livia Negrucci Cantowitz")
st.warning("Protótipo acadêmico treinado com dados simulados. Não é diagnóstico. "
           "A classificação deve servir para proteger o apostador, e nunca para direcionar mais publicidade de apostas.")

modo = st.radio("Como você quer informar as apostas?",
                ["Simular um padrão de apostas", "Escolher um apostador do dataset", "Enviar um CSV de apostas"],
                horizontal=True)

if modo == "Simular um padrão de apostas":
    st.write("Informe o comportamento nas **apostas recentes** (equivalente à primeira metade do histórico).")
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Nº de apostas", 1, 25, 10)
    media = c1.number_input("Valor médio por aposta (R$)", 1.0, 1000.0, 80.0, step=10.0)
    maior = c1.number_input("Maior aposta feita (R$)", 1.0, 2000.0, 120.0, step=10.0)
    vitorias = c2.slider("Taxa de vitória (%)", 0, 100, 35) / 100
    odds = c2.slider("Odds médias", 1.1, 10.0, 4.5, step=0.1)
    multiplas = c2.slider("% de apostas múltiplas", 0, 100, 35) / 100
    seq = c3.slider("Maior sequência de derrotas seguidas", 0, 20, 4)
    resultado = c3.number_input("Resultado líquido no período (R$, negativo = prejuízo)", -20000.0, 20000.0, -100.0,
                                step=50.0)
    mostrar_resultado(modelo, montar_atributos(n, media, maior, vitorias, odds, multiplas, seq, resultado))

elif modo == "Escolher um apostador do dataset":
    st.write("O modelo vê só a **primeira metade** das apostas do usuário e prevê o perfil que ele terá ao final. "
             "Depois comparamos com o perfil real.")
    uid = st.selectbox("ID do apostador", parcial.index.tolist(), index=152)
    x = parcial.loc[[uid]]
    with st.expander("Atributos calculados (primeira metade do histórico)"):
        st.dataframe(x[ATRIBUTOS].rename(columns=NOMES).T.rename(columns={uid: "valor"}).round(2))
    mostrar_resultado(modelo, x, perfil_real=completo.loc[uid, "perfil_final"])

else:
    st.write("Envie um CSV no mesmo formato do `bets.csv` (separador `;`, colunas `bet_id; user_id; bet_type; sport; "
             "odds; is_win; stake; gain; GGR`). Se houver vários usuários, escolha qual analisar.")
    arquivo = st.file_uploader("Arquivo CSV", type="csv")
    if arquivo is not None:
        bets = pd.read_csv(arquivo, sep=";")
        if bets["is_win"].dtype == object:
            bets["is_win"] = bets["is_win"].astype(str).str.lower().isin(["true", "1"])
        bets["ordem"] = bets["bet_id"].astype(str).str.extract(r"(\d+)")[0].astype(int)
        agg = agregar_por_apostador(bets).set_index("user_id")
        uid = st.selectbox("Usuário", agg.index.tolist())
        mostrar_resultado(modelo, agg.loc[[uid]])
