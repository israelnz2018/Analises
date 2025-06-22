from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import traceback
import os

from leitura import ler_arquivo
from suporte import interpretar_coluna  # Supondo que você tem essa função no suporte.py
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
    # ... (sua lista completa de configurações de análises como já está no seu código)
    "Correlação de person": ["df", "colunas_y", "lista_x"],
    # Adicione as outras conforme já está no seu main
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

        # Coleta os valores enviados
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

        # 🔑 Mapeia para os nomes reais do df
        colunas_y_reais = [interpretar_coluna(df, y) for y in colunas_y] if colunas_y else []
        lista_x_reais = [interpretar_coluna(df, x) for x in lista_x] if lista_x else []
        lista_z_reais = [interpretar_coluna(df, z) for z in lista_z] if lista_z else []
        subgrupo_real = interpretar_coluna(df, subgrupo_val) if subgrupo_val else None

        colunas_usadas = colunas_y_reais + lista_x_reais + lista_z_reais
        if subgrupo_real:
            colunas_usadas.append(subgrupo_real)

        resultado_texto = ""
        imagem_analise_base64 = None
        imagem_grafico_isolado_base64 = None

        if ferramenta:
            funcao = ANALISES.get(ferramenta.strip())
            if not funcao:
                return JSONResponse({"erro": f"Análise {ferramenta} desconhecida."}, status_code=400)

            disponiveis = {
                "df": df,
                "colunas_y": colunas_y_reais,
                "lista_x": lista_x_reais,
                "lista_z": lista_z_reais,
                "subgrupo": subgrupo_real,
                "field": field,
                "field_distribuicao": field_distribuicao,
                "colunas_usadas": colunas_usadas
            }

            permitidos = CONFIG_ANALISES.get(ferramenta.strip(), ["df"])
            args_to_pass = {k: disponiveis[k] for k in permitidos if k in disponiveis}

            resultado_texto, imagem_analise_base64 = funcao(**args_to_pass)

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


