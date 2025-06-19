from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import traceback
import os

from leitura import ler_arquivo 
from estatistica import ANALISES
from graficos import GRAFICOS

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚩 main.py carregado com PROJECT=", os.getenv("PROJECT"))

@app.get("/healthz")
def healthcheck():
    return JSONResponse({"status": "ok"})

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
        df = await ler_arquivo(arquivo)
        colunas_usadas = []

        if coluna_y and coluna_y.strip():
            colunas_usadas.append(coluna_y.strip())

        if colunas_x:
            if isinstance(colunas_x, str):
                colunas_usadas.extend([x.strip() for x in colunas_x.split(",") if x.strip()])
            else:
                for item in colunas_x:
                    colunas_usadas.extend([x.strip() for x in item.split(",") if x.strip()])

        resultado_texto = None
        imagem_analise_base64 = None
        imagem_grafico_isolado_base64 = None

        # Executa análise
        if ferramenta and ferramenta.strip():
            funcao = ANALISES.get(ferramenta.strip())
            if not funcao:
                return JSONResponse(content={"erro": "Análise estatística desconhecida."}, status_code=400)
            resultado_texto, imagem_analise_base64 = funcao(df, colunas_usadas)

        # Executa gráfico
        if grafico and grafico.strip():
            print(f"🎨 Gráfico solicitado: {grafico.strip()}")
            print(f"📊 Colunas usadas: {colunas_usadas}")
            funcao = GRAFICOS.get(grafico.strip())
            if not funcao:
                print(f"❌ Gráfico {grafico.strip()} não encontrado no GRAFICOS.")
                return JSONResponse(content={"erro": f"Gráfico {grafico.strip()} não encontrado."}, status_code=400)
            imagem_grafico_isolado_base64 = funcao(
                df,
                colunas_usadas,
                coluna_y=coluna_y.strip() if coluna_y else None
            )

        return {
            "analise": resultado_texto or "",
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
