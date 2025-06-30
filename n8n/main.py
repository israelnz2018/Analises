# Atualizacao forçada - 27/06/2025

from fastapi import FastAPI, File, UploadFile, Form, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  
import traceback
import os
from fastapi import Form
import base64
import io
import matplotlib.pyplot as plt

# 🧠 Importa o agente de IA
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

# Middleware de CORS atualizado para incluir seu novo domínio
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://educacaopelotrabalho-production.up.railway.app",
        "https://app.educacaopelotrabalho.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Responde preflight OPTIONS universalmente
@app.options("/{path:path}")
async def preflight_handler(path: str):
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = ",".join(allowed_origins)
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

# ✅ Variáveis globais para armazenar última análise e último gráfico
ultima_analise = None
ultimo_grafico = None

@app.post("/pergunta")
async def pergunta_ia(request: Request):
    try:
        dados = await request.json()
        pergunta = dados.get("prompt", "").strip()
        if not pergunta:
            return JSONResponse({"erro": "Nenhuma pergunta fornecida."}, status_code=400)

        # Prioridade: usa última análise se existir, senão último gráfico
        contexto = ultima_analise if ultima_analise else ultimo_grafico
        if not contexto:
            return JSONResponse({"erro": "Nenhuma análise ou gráfico anterior encontrado para contexto."}, status_code=400)

        resposta = perguntar_ia(pergunta, contexto)
        return {"resposta": resposta}

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({
            "erro": "Erro interno ao processar a pergunta.",
            "detalhe": str(e),
            "traceback": tb
        }, status_code=500)

# ✅ Variável global para armazenar df atual
df_global = None

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
        if not arquivo:
            return JSONResponse({"erro": "Nenhum arquivo recebido."}, status_code=400)

        df = await ler_arquivo(arquivo, aba)
        if df is None or df.empty:
            return JSONResponse({"erro": "Arquivo vazio ou aba inválida."}, status_code=400)

        # 🔧 Processa lista_y
        if lista_y:
            if isinstance(lista_y, str):
                lista_y_processada = [x.strip() for x in lista_y.split(",")] if "," in lista_y else [lista_y.strip()]
            elif isinstance(lista_y, list):
                lista_y_processada = [x.strip() for x in lista_y]
            else:
                lista_y_processada = []
        else:
            lista_y_processada = []

        # 🔧 Processa lista_x
        if lista_x:
            if isinstance(lista_x, str):
                lista_x_processada = [x.strip() for x in lista_x.split(",")] if "," in lista_x else [lista_x.strip()]
            elif isinstance(lista_x, list):
                lista_x_processada = [x.strip() for x in lista_x]
            else:
                lista_x_processada = []
        else:
            lista_x_processada = []

        # 🔧 Processa coluna_z
        lista_z = [coluna_z.strip()] if coluna_z else []

        subgrupo_val = subgrupo.strip() if subgrupo else None

        # 🔧 Concatena colunas usadas
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

            permitidos = CONFIG_ANALISES.get(ferramenta.strip(), ["df"])
            args_to_pass = {k: disponiveis[k] for k in permitidos if k in disponiveis}
            resultado_texto, imagem_analise_base64 = funcao(**args_to_pass)

            # ✅ Atualiza variável global de última análise
            global ultima_analise
            ultima_analise = resultado_texto

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

            import inspect
            params_aceitos = inspect.signature(funcao_grafico).parameters
            args_filtrados = {k: v for k, v in disponiveis.items() if k in params_aceitos}

            imagem_grafico_isolado_base64 = funcao_grafico(**args_filtrados)

            # ✅ Atualiza variável global de último gráfico
            global ultimo_grafico
            ultimo_grafico = "[gráfico isolado gerado]"

        # ✅ Atualiza a variável global com o DataFrame carregado
        global df_global
        df_global = df

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
