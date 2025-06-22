from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import traceback
import os

from leitura import ler_arquivo
from Capabilidade import ANALISES as ANALISES_CAP
from Exploratoria import ANALISES as ANALISES_EXP
from Inferencial import ANALISES as ANALISES_INF
from Preditiva import ANALISES as ANALISES_PRED
from Controledeprocesso import ANALISES as ANALISES_PROC
from Analisesdiversas import ANALISES as ANALISES_DIVERSAS
from graficos import GRAFICOS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://educacaopelotrabalho-production.up.railway.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANALISES = {}
ANALISES.update(ANALISES_CAP)
ANALISES.update(ANALISES_EXP)
ANALISES.update(ANALISES_INF)
ANALISES.update(ANALISES_PRED)
ANALISES.update(ANALISES_PROC)
ANALISES.update(ANALISES_DIVERSAS)

CONFIG_ANALISES = {
    "Gráfico Sumario": ["df", "colunas_y"],
    "Análise de outliers": ["df", "lista_x"],
    "Correlação de person": ["df", "colunas_y", "lista_x"],
    "Matrix de dispersão": ["df", "colunas_y", "lista_x"],
    "Análise de estabilidade": ["df", "colunas_y", "subgrupo"],
    "Análise de distribuição estatística": ["df", "colunas_y"],
    "Análise de limpeza dos dados": ["df"],
    "Histograma": ["df", "lista_x", "subgrupo"],
    "Pareto": ["df", "lista_x", "subgrupo", "colunas_y"],
    "Setores (Pizza)": ["df", "lista_x", "colunas_y"],
    "Barras": ["df", "lista_x", "colunas_y", "subgrupo"],
    "BoxPlot": ["df", "lista_x", "subgrupo"],
    "Dispersão": ["df", "colunas_y", "lista_x", "subgrupo"],
    "Tendência": ["df", "colunas_y", "lista_x", "subgrupo"],
    "Bolhas - 3D": ["df", "lista_x", "colunas_y", "lista_z"],
    "Superfície - 3D": ["df", "lista_x", "colunas_y", "lista_z"],
    "Pareto simples": ["df", "lista_x", "subgrupo"],
    "Gráfifo de barras": ["df", "lista_x", "subgrupo"],
    "BoxPlot simples": ["df", "colunas_y", "subgrupo"],
    "Gráfico de disperao": ["df", "colunas_y", "lista_x"],
    "Gráfico de tendecias": ["df", "colunas_y", "lista_x"],
    "Gráficos de bolhas": ["df", "colunas_y", "lista_x", "lista_z"],
    "1 Sample T": ["df", "colunas_y", "field"],
    "2 Sample T": ["df", "colunas_y"],
    "2 Paired Test": ["df", "colunas_y"],
    "One way ANOVA": ["df", "colunas_y", "subgrupo"],
    "1 Wilcoxon": ["df", "colunas_y", "field"],
    "2 Mann-Whitney": ["df", "colunas_y"],
    "Kruskal-Wallis": ["df", "colunas_y", "subgrupo"],
    "Friedman Pareado": ["df", "colunas_y", "subgrupo"],
    "1 Intervalo de Confianca": ["df", "colunas_y", "field"],
    "1 Intervalo Interquartilico": ["df", "colunas_y"],
    "2 Variancas": ["df", "colunas_y"],
    "2 Variancas Brown-Forsythe": ["df", "colunas_y"],
    "Bartlett": ["df", "colunas_y"],
    "Brown-Forsythe": ["df", "colunas_y"],
    "1 Intervalo de Confianca Variancia": ["df", "colunas_y", "field"],
    "1 Proporcao": ["df", "colunas_y", "field"],
    "2 Proporcoes": ["df", "colunas_y", "field"],
    "K Proporcoes": ["df", "colunas_y"],
    "Qui-quadrado": ["df", "colunas_y", "lista_x"],
    "Tipo de modelo de regressão": ["df", "colunas_y", "lista_x"],
    "Regressão linear simples": ["df", "colunas_y", "lista_x"],
    "Regressão linear múltipla": ["df", "colunas_y", "lista_x"],
    "Regressão logística binária": ["df", "colunas_y", "lista_x"],
    "Regressão logística ordinal": ["df", "colunas_y", "lista_x"],
    "Regressão logística nominal": ["df", "colunas_y", "lista_x"],
    "Árvore de decisão": ["df", "colunas_y", "lista_x"],
    "Random Forest": ["df", "colunas_y", "lista_x"],
    "ARIMA": ["df", "colunas_y", "field"],
    "Holt-Winters": ["df", "colunas_y", "field"],
    "Carta I-MR": ["df", "colunas_y"],
    "Carta X-Barra R": ["df", "colunas_y", "subgrupo"],
    "Carta X-Barra S": ["df", "colunas_y", "subgrupo"],
    "Carta P": ["df", "colunas_y", "subgrupo"],
    "Carta NP": ["df", "colunas_y", "subgrupo"],
    "Carta C": ["df", "colunas_y"],
    "Carta U": ["df", "colunas_y", "subgrupo"],
    "Teste de normalidade": ["df", "colunas_y"],
    "Análise de estabilidade": ["df", "colunas_y"],
    "Análise de distribuição estatística": ["df", "colunas_y"],
    "Capabilidade - dados normais": ["df", "colunas_y", "field"],
    "Capabilidade - outras distribuições": ["df", "colunas_y", "field", "field_distribuicao"],
    "Capabilidade - com dados transformados": ["df", "colunas_y", "field"],
    "Capabilidade - com dados discretizados": ["df", "colunas_y", "field"],
    "Cálculo de Probabilidade": ["df", "colunas_y", "field"]
}

@app.post("/analise")
async def analisar(
    request: Request,
    arquivo: UploadFile = File(None),
    ferramenta: str = Form(None),
    grafico: str = Form(None),
    coluna_y: str = Form(None),
    colunas_x: str | list[str] = Form(None),
    coluna_z: str = Form(None),
    subgrupo: str = Form(None),
    field: str = Form(None),
    field_distribuicao: str = Form(None)
):
    try:
        df = await ler_arquivo(arquivo)
        colunas_y = [y.strip() for y in coluna_y.split(",")] if coluna_y else []
        lista_x = []
        if colunas_x:
            if isinstance(colunas_x, str):
                lista_x = [x.strip() for x in colunas_x.split(",")]
            else:
                for item in colunas_x:
                    lista_x.extend([x.strip() for x in item.split(",")])
        lista_z = [coluna_z.strip()] if coluna_z else []
        subgrupo_val = subgrupo.strip() if subgrupo else None
        colunas_usadas = colunas_y + lista_x + lista_z
        if subgrupo_val:
            colunas_usadas.append(subgrupo_val)
        resultado_texto = ""
        imagem_analise_base64 = None
        imagem_grafico_isolado_base64 = None
        if ferramenta:
            funcao = ANALISES.get(ferramenta.strip())
            if not funcao:
                return JSONResponse({"erro": f"Análise {ferramenta} desconhecida."}, status_code=400)
            disponiveis = {"df": df, "colunas_y": colunas_y, "lista_x": lista_x, "lista_z": lista_z, "subgrupo": subgrupo_val, "field": field, "field_distribuicao": field_distribuicao, "colunas_usadas": colunas_usadas}
            permitidos = CONFIG_ANALISES.get(ferramenta.strip(), ["df", "colunas_y", "lista_x", "lista_z", "subgrupo", "field"])
            args_to_pass = {k: disponiveis[k] for k in permitidos if k in disponiveis}
            resultado_texto, imagem_analise_base64 = funcao(**args_to_pass)
        if grafico:
            funcao_grafico = GRAFICOS.get(grafico.strip())
            if not funcao_grafico:
                return JSONResponse({"erro": f"Gráfico {grafico} não encontrado."}, status_code=400)
            imagem_grafico_isolado_base64 = funcao_grafico(df, colunas_usadas)
        return {"analise": resultado_texto, "grafico_base64": imagem_analise_base64 or [], "grafico_isolado_base64": imagem_grafico_isolado_base64, "colunas_utilizadas": colunas_usadas}
    except ValueError as e:
        return JSONResponse({"erro": str(e)}, status_code=400)
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"erro": "Erro interno ao processar a análise.", "detalhe": str(e), "traceback": tb}, status_code=500)

