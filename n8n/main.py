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

# Inicializa app
app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://educacaopelotrabalho-production.up.railway.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("✅ CORS configurado com domínio: https://educacaopelotrabalho-production.up.railway.app")

print("🚩 main.py carregado com PROJECT=", os.getenv("PROJECT"))

# Junta os ANALISES
ANALISES = {}
ANALISES.update(ANALISES_CAP)
ANALISES.update(ANALISES_EXP)
ANALISES.update(ANALISES_INF)
ANALISES.update(ANALISES_PRED)
ANALISES.update(ANALISES_PROC)
ANALISES.update(ANALISES_DIVERSAS)
print("✅ ANALISES carregados com sucesso")

ANALISES_COM_FIELD = {
    "1 Sample T",
    "1 Wilcoxon",
    "1 Intervalo de Confianca",
    "1 Intervalo de Confianca Variancia",
    "1 Proporcao",
    "2 Proporcoes",
    "ARIMA",
    "Holt-Winters",
    "Capabilidade - dados normais",
    "Capabilidade - outras distribuições",
    "Capabilidade - com dados transformados",
    "Capabilidade - com dados discretizados",
    "Cálculo de Probabilidade"
}

@app.get("/healthz")
def healthcheck():
    return JSONResponse({"status": "ok"})


ANALISES_COM_FIELD = {
    "1 Sample T",
    "1 Wilcoxon",
    "1 Intervalo de Confianca",
    "1 Intervalo de Confianca Variancia",
    "1 Proporcao",
    "2 Proporcoes",
    "ARIMA",
    "Holt-Winters",
    "Capabilidade - dados normais",
    "Capabilidade - outras distribuições",
    "Capabilidade - com dados transformados",
    "Capabilidade - com dados discretizados",
    "Cálculo de Probabilidade"
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
    field: str = Form(None)
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

            resultado_texto, imagem_analise_base64 = funcao(
                df, colunas_y, lista_x, lista_z, subgrupo_val, field=field
            )

        if grafico:
            funcao_grafico = GRAFICOS.get(grafico.strip())
            if not funcao_grafico:
                return JSONResponse({"erro": f"Gráfico {grafico} não encontrado."}, status_code=400)

            imagem_grafico_isolado_base64 = funcao_grafico(df, colunas_usadas)

        return {
            "analise": resultado_texto,
            "grafico_base64": imagem_analise_base64 or [],
            "grafico_isolado_base64": imagem_grafico_isolado_base64,
            "colunas_utilizadas": colunas_usadas
        }

    except ValueError as e:
        return JSONResponse({"erro": str(e)}, status_code=400)
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({
            "erro": "Erro interno ao processar a análise.",
            "detalhe": str(e),
            "traceback": tb
        }, status_code=500)
