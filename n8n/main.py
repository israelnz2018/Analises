# main.py - atualizado 01/07/2025

from fastapi import FastAPI, File, UploadFile, Form, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  
import traceback
import os
import base64
import io
import matplotlib.pyplot as plt

# ✅ Importa agente IA com memória curta
from agente import perguntar_ia

# Tentativa segura de importação dos módulos 
try:
    from leitura import ler_arquivo
    from Capabilidade import ANALISES as ANALISES_CAP
    from Exploratoria import ANALISES as ANALISES_EXP 
    from Inferencial import ANALISES as ANALISES_INF
    from Preditiva import ANALISES as ANALISES_PRED
    from Controledeprocesso import ANALISES as ANALISES_PROC
    from Analisesdiversas import ANALISES as ANALISES_DIVERSAS
    from graficos import GRAFICOS
except ImportError as e:
    print(f"⚠ Erro de importação: {str(e)}")

# Iniciar app
app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajuste se necessário
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Responde preflight OPTIONS universalmente
@app.options("/{path:path}")
async def preflight_handler(path: str):
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Construção do dicionário de análises
ANALISES = {}
ANALISES.update(ANALISES_CAP if 'ANALISES_CAP' in locals() else {})
ANALISES.update(ANALISES_EXP if 'ANALISES_EXP' in locals() else {})
ANALISES.update(ANALISES_INF if 'ANALISES_INF' in locals() else {})
ANALISES.update(ANALISES_PRED if 'ANALISES_PRED' in locals() else {})
ANALISES.update(ANALISES_PROC if 'ANALISES_PROC' in locals() else {})
ANALISES.update(ANALISES_DIVERSAS if 'ANALISES_DIVERSAS' in locals() else {})

# ✅ Variável global para armazenar último resultado
ultimo_resultado = {
    "analise": None,
    "grafico": None
}

@app.post("/analise")
async def analisar(
    request: Request,
    arquivo: UploadFile = File(None),
    ferramenta: str = Form(None),
    grafico: str = Form(None),
    aba: str = Form(None),
    coluna_y: str = Form(None),
    coluna_x: str = Form(None),
    coluna_z: str = Form(None),
    lista_y: str = Form(None),
    lista_x: str = Form(None),
    subgrupo: str = Form(None),
    field: str = Form(None),
    field_conf: str = Form(None),
    field_dist: str = Form(None),
    field_LSE: str = Form(None),
    field_LIE: str = Form(None),
    Data: str = Form(None)
):
    try:
        global ultimo_resultado

        if not arquivo:
            return JSONResponse({"erro": "Nenhum arquivo recebido."}, status_code=400)

        df = await ler_arquivo(arquivo, aba)
        if df is None or df.empty:
            return JSONResponse({"erro": "Arquivo vazio ou aba inválida."}, status_code=400)

        # Processa listas
        lista_y_processada = [x.strip() for x in lista_y.split(",")] if lista_y else []
        lista_x_processada = [x.strip() for x in lista_x.split(",")] if lista_x else []
        lista_z = [coluna_z.strip()] if coluna_z else []
        subgrupo_val = subgrupo.strip() if subgrupo else None

        colunas_usadas = lista_y_processada + lista_x_processada + lista_z
        if subgrupo_val:
            colunas_usadas.append(subgrupo_val)

        resultado_texto = ""
        imagem_analise_base64 = None
        imagem_grafico_isolado_base64 = None

        if ferramenta:
            funcao = ANALISES.get(ferramenta.strip())
            if not funcao:
                return JSONResponse({"erro": f"Análise {ferramenta} desconhecida."}, status_code=400)

            disponiveis = {
                "df": df,
                "coluna_y": coluna_y.strip() if coluna_y else None,
                "coluna_x": coluna_x.strip() if coluna_x else None,
                "coluna_z": coluna_z.strip() if coluna_z else None,
                "Data": Data,
                "lista_y": lista_y_processada,
                "lista_x": lista_x_processada,
                "lista_z": lista_z,
                "subgrupo": subgrupo_val,
                "field": field,
                "field_conf": field_conf,
                "field_dist": field_dist,
                "field_LSE": field_LSE,
                "field_LIE": field_LIE
            }

            resultado_texto, imagem_analise_base64 = funcao(**disponiveis)

            # ✅ Salva no último resultado
            ultimo_resultado["analise"] = resultado_texto

        if grafico:
            funcao_grafico = GRAFICOS.get(grafico.strip())
            if not funcao_grafico:
                return JSONResponse({"erro": f"Gráfico {grafico} não encontrado."}, status_code=400)

            disponiveis = {
                "df": df,
                "coluna_y": coluna_y.strip() if coluna_y else None,
                "coluna_x": coluna_x.strip() if coluna_x else None,
                "coluna_z": coluna_z.strip() if coluna_z else None,
                "Data": Data,
                "lista_y": lista_y_processada,
                "lista_x": lista_x_processada,
                "lista_z": lista_z,
                "subgrupo": subgrupo_val,
                "field": field,
                "field_conf": field_conf,
                "field_dist": field_dist,
                "field_LSE": field_LSE,
                "field_LIE": field_LIE
            }

            imagem_grafico_isolado_base64 = funcao_grafico(**disponiveis)

            # ✅ Salva no último resultado
            ultimo_resultado["grafico"] = "[gráfico isolado gerado]"

        return {
            "analise": resultado_texto,
            "grafico_base64": imagem_analise_base64 or [],
            "grafico_isolado_base64": imagem_grafico_isolado_base64,
            "colunas_utilizadas": colunas_usadas
        }

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({
            "erro": "Erro interno ao processar a análise.",
            "detalhe": str(e),
            "traceback": tb
        }, status_code=500)

# ✅ Novo endpoint para perguntas ao agente IA
@app.post("/pergunta")
async def pergunta_agente(
    pergunta: str = Form(...),
    tipo: str = Form(...)  # "analise" ou "grafico"
):
    try:
        global ultimo_resultado

        if tipo not in ["analise", "grafico"]:
            return JSONResponse({"erro": "Tipo inválido. Use 'analise' ou 'grafico'."}, status_code=400)

        contexto = ultimo_resultado.get(tipo)
        if not contexto:
            return JSONResponse({"erro": f"Nenhum {tipo} disponível para responder."}, status_code=400)

        resposta = perguntar_ia(pergunta, tipo)

        return {"resposta": resposta}

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({
            "erro": "Erro interno ao consultar agente IA.",
            "detalhe": str(e),
            "traceback": tb
        }, status_code=500)
