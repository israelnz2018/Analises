from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import traceback
from pathlib import Path

from leitura import ler_arquivo
from suporte import interpretar_coluna
from estatistica import ANALISES
from graficos import GRAFICOS
from agente import interpretar_analise

app = FastAPI()
print("🚩 main.py carregado")

@app.get("/healthz")
def healthcheck():
    print("🚩 /healthz chamada")
    return JSONResponse({"status": "ok"})

pasta_raiz = Path(__file__).parent

app.mount(
    "/n8n",
    StaticFiles(directory=pasta_raiz),
    name="n8n_static"
)
print(f"🚩 Arquivos estáticos montados em /n8n a partir de {pasta_raiz}")

templates = Jinja2Templates(directory=str(pasta_raiz))

@app.get("/", response_class=HTMLResponse)
async def raiz(request: Request):
    print("🚩 / chamada – enviando index.html")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analise")
async def analisar(
    request: Request,
    arquivo: UploadFile = File(None),
    ferramenta: str = Form(None),
    grafico: str = Form(None),
    coluna_y: str = Form(None),
    colunas_x: str | list[str] = Form(None)
):
    print("🚩 /analise chamada")
    print(f"  - ferramenta: {ferramenta}")
    print(f"  - grafico:   {grafico}")
    print(f"  - coluna_y:  {coluna_y}")
    print(f"  - colunas_x: {colunas_x}")
    try:
        df = await ler_arquivo(arquivo)
        print(f"🚩 Arquivo lido: {len(df)} linhas, colunas: {list(df.columns)}")
        colunas_usadas = []

        nome_coluna_y = None
        if coluna_y and coluna_y.strip():
            try:
                nome_coluna_y = interpretar_coluna(df, coluna_y.strip())
                print(f"  → coluna_y interpretada como: {nome_coluna_y}")
                if nome_coluna_y:
                    colunas_usadas.append(nome_coluna_y)
            except Exception as e:
                print(f"❌ Erro ao interpretar coluna_y: {e}")

        colunas_x_lista = []
        if colunas_x:
            if isinstance(colunas_x, str):
                colunas_x_lista = [x.strip() for x in colunas_x.split(",") if x.strip()]
            else:
                for item in colunas_x:
                    colunas_x_lista.extend([x.strip() for x in item.split(",") if x.strip()])
            print(f"  → lista de colunas_x: {colunas_x_lista}")

            for letra in colunas_x_lista:
                try:
                    nome_coluna = interpretar_coluna(df, letra)
                    print(f"    • coluna_x '{letra}' → '{nome_coluna}'")
                    if nome_coluna:
                        colunas_usadas.append(nome_coluna)
                except Exception as e:
                    print(f"❌ Erro ao interpretar colunas_x '{letra}': {e}")

        print(f"🚩 colunas_usadas finais: {colunas_usadas}")
        if not colunas_usadas:
            print("⚠️ Nenhuma coluna válida informada")
            return JSONResponse(content={"erro": "Informe ao menos coluna_y ou colunas_x."}, status_code=422)

        for col in colunas_usadas:
            if col not in df.columns:
                print(f"⚠️ Coluna não encontrada no DataFrame: {col}")
                return JSONResponse(content={"erro": f"Coluna '{col}' não encontrada no arquivo."}, status_code=400)

        resultado_texto = None
        imagem_analise_base64 = None
        imagem_grafico_isolado_base64 = None
        explicacao_ia = None

        if ferramenta and ferramenta.strip():
            print(f"🚩 Executando análise '{ferramenta.strip()}'")
            funcao = ANALISES.get(ferramenta.strip())
            if not funcao:
                print(f"❌ Análise desconhecida: {ferramenta.strip()}")
                return JSONResponse(content={"erro": "Análise estatística desconhecida."}, status_code=400)
            resultado_texto, imagem_analise_base64 = funcao(df, colunas_usadas)
            print(f"🚩 Resultado texto obtido, iniciando explicação IA")
            explicacao_ia = interpretar_analise(resultado_texto)

        if grafico and grafico.strip():
            print(f"🚩 Gerando gráfico '{grafico.strip()}'")
            funcao = GRAFICOS.get(grafico.strip())
            if not funcao:
                print(f"❌ Gráfico desconhecido: {grafico.strip()}")
                return JSONResponse(content={"erro": "Gráfico desconhecido."}, status_code=400)
            imagem_grafico_isolado_base64 = funcao(
                df,
                colunas_usadas,
                coluna_y=nome_coluna_y
            )

        print("🚩 /analise concluída com sucesso")
        return {
            "analise": resultado_texto or "",
            "explicacao_ia": explicacao_ia,
            "grafico_base64": imagem_analise_base64 or [],
            "grafico_isolado_base64": imagem_grafico_isolado_base64,
            "colunas_utilizadas": colunas_usadas
        }

    except ValueError as e:
        print(f"❌ ValueError: {e}")
        return JSONResponse(content={"erro": str(e)}, status_code=400)

    except Exception as e:
        tb = traceback.format_exc()
        print("🔴 ERRO COMPLETO:\n", tb)
        return JSONResponse(
            content={
                "erro": "Erro interno ao processar a análise.",
                "detalhe": str(e),
                "traceback": tb
            },
            status_code=500
        )

