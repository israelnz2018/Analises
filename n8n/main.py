from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.templating import Jinja2Templates
import os
import traceback
from pathlib import Path
import pandas as pd
from io import BytesIO

from leitura import ler_arquivo
from suporte import interpretar_coluna
from estatistica import ANALISES
from graficos import GRAFICOS
from agente import perguntar_ia  # ainda importado, mas não usado

# Cria app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("🚩 main.py carregado com PROJECT=", os.getenv("PROJECT"))

# Templates e path raiz
pasta_raiz = Path(__file__).parent
templates = Jinja2Templates(directory=str(pasta_raiz))

# Define variável de ambiente para controlar UI
PROJECT_MODE = os.getenv("PROJECT", "analises").lower()
SERVE_UI = PROJECT_MODE == "html"

# Healthcheck
@app.get("/healthz")
def healthcheck():
    return JSONResponse({"status": "ok"})

# Monta UI apenas se estiver no modo html
if SERVE_UI:
    app.mount(
        "/n8n",
        StaticFiles(directory=pasta_raiz),
        name="n8n_static"
    )

    @app.get("/", response_class=HTMLResponse)
    async def raiz(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/visualizar")
    async def visualizar_planilha(file: UploadFile = File(...)):
        try:
            conteudo = await file.read()
            df = pd.read_excel(BytesIO(conteudo))
            preview = df.head(10).to_dict(orient="records")
            return JSONResponse({"status": "ok", "preview": preview, "colunas": list(df.columns)})
        except Exception as e:
            return JSONResponse({"status": "erro", "mensagem": str(e)})

# Endpoint de análise
@app.post("/analise")
async def analisar(
    request: Request,
    arquivo: UploadFile = File(None),
    ferramenta: str = Form(None),
    grafico: str = Form(None),
    coluna_y: str = Form(None),
    colunas_x: str | list[str] = Form(None)
):
    try:
        # Leitura
        df = await ler_arquivo(arquivo)
        colunas_usadas = []

        # Interpretação de coluna Y
        nome_coluna_y = None
        if coluna_y and coluna_y.strip():
            try:
                nome_coluna_y = interpretar_coluna(df, coluna_y.strip())
                if nome_coluna_y:
                    colunas_usadas.append(nome_coluna_y)
            except Exception as e:
                print(f"Erro ao interpretar coluna_y: {e}")

        # Interpretação de colunas X
        colunas_x_lista = []
        if colunas_x:
            if isinstance(colunas_x, str):
                colunas_x_lista = [x.strip() for x in colunas_x.split(",") if x.strip()]
            else:
                for item in colunas_x:
                    colunas_x_lista.extend([x.strip() for x in item.split(",") if x.strip()])
            for letra in colunas_x_lista:
                try:
                    nome_coluna = interpretar_coluna(df, letra)
                    if nome_coluna:
                        colunas_usadas.append(nome_coluna)
                except Exception as e:
                    print(f"Erro ao interpretar colunas_x: {e}")

        # Validação
        if not colunas_usadas:
            return JSONResponse(content={"erro": "Informe ao menos coluna_y ou colunas_x."}, status_code=422)
        for col in colunas_usadas:
            if col not in df.columns:
                return JSONResponse(content={"erro": f"Coluna '{col}' não encontrada no arquivo."}, status_code=400)

        # Processa análise
        resultado_texto = None
        imagem_analise_base64 = None
        imagem_grafico_isolado_base64 = None
        explicacao_ia = None  # ✅ Agente IA desativado no botão "Enviar"

        if ferramenta and ferramenta.strip():
            funcao = ANALISES.get(ferramenta.strip())
            if not funcao:
                return JSONResponse(content={"erro": "Análise estatística desconhecida."}, status_code=400)
            resultado_texto, imagem_analise_base64 = funcao(df, colunas_usadas)
            explicacao_ia = None  # agente removido daqui

        # Processa gráfico
        if grafico and grafico.strip():
            funcao = GRAFICOS.get(grafico.strip())
            if not funcao:
                return JSONResponse(content={"erro": "Gráfico desconhecido."}, status_code=400)
            imagem_grafico_isolado_base64 = funcao(
                df,
                colunas_usadas,
                coluna_y=nome_coluna_y
            )

        return {
            "analise": resultado_texto or "",
            "explicacao_ia": explicacao_ia,
            "grafico_base64": imagem_analise_base64 or [],
            "grafico_isolado_base64": imagem_grafico_isolado_base64,
            "colunas_utilizadas": colunas_usadas
        }

    except ValueError as e:
        return JSONResponse(content={"erro": str(e)}, status_code=400)
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            content={
                "erro": "Erro interno ao processar a análise.",
                "detalhe": str(e),
                "traceback": tb
            },
            status_code=500
        )







