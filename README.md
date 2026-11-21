# Identificação de Perfis de Risco em Apostadores Online por Meio de Aprendizado de Máquina

**Nome:** Livia Negrucci Cantowitz – RA 10389419 

## Sobre o projeto
Classificação de apostadores online em perfis de risco (**baixo**, **médio** e **alto**) com aprendizado de máquina supervisionado.
O experimento principal é de **detecção precoce**: o modelo vê apenas a primeira metade das apostas de cada usuário e prevê o perfil de risco que ele terá ao final do histórico.
Alinhado ao **ODS 3 – Saúde e Bem-Estar**.

## Principais resultados (conjunto de teste, 1.000 apostadores)
| Modelo | F1-macro | AUC (alto vs. demais) |
|---|---|---|
| Regra aplicada ao histórico parcial (linha de base) | 0,668 | – |
| **Regressão Logística** | **0,692** | **0,914** |
| MLP (rede neural) | 0,685 | 0,916 |
| XGBoost | 0,672 | 0,906 |

A diferença para a linha de base é estatisticamente significativa (McNemar, p = 0,031).

## Estrutura do repositório
| Arquivo / pasta | Conteúdo |
|---|---|
| `bets.csv` | Dataset original (100.000 apostas de 5.000 usuários) – [Sports Betting Profiling Dataset (Kaggle)](https://www.kaggle.com/datasets/emiliencoicaud/sports-betting-profiling-dataset) |
| `apostadores_perfil_risco.csv` | Base agregada por apostador com o perfil de risco (Parte 2) |
| `analise_exploratoria_apostadores .ipynb` | Análise exploratória e regra de rotulação (Parte 2) |
| `src/preparacao_dados.py` | Funções de agregação, histórico parcial e regra de risco |
| `02_modelagem_avaliacao.ipynb` | Comparação de algoritmos, ajuste de hiperparâmetros, detecção precoce, SHAP e k-means (Parte 3) |
| `app.py` | Protótipo interativo em Streamlit |
| `figuras/` | Gráficos usados no artigo |
| `resultados/` | Tabelas de resultados em CSV |
| `modelos/` | Modelo treinado e metadados usados pelo protótipo |
| `requirements.txt` | Dependências |

## Como executar
Requer Python 3.10 ou superior.

~~~bash
pip install -r requirements.txt

# reproduzir os experimentos: abrir e executar todas as células de
# 02_modelagem_avaliacao.ipynb (gera figuras/, resultados/ e modelos/)

# rodar o protótipo
streamlit run app.py
~~~

## Artigo e vídeo
- Artigo final: `Artigo_Final.pdf` (neste repositório)
- Vídeo de apresentação: https://youtu.be/EnhBmQsgl8k

## Limitações
O dataset é simulado e o perfil de risco vem de uma regra definida no projeto, e não de um diagnóstico clínico (ex.: PGSI). O protótipo é acadêmico e não deve ser usado como diagnóstico.
