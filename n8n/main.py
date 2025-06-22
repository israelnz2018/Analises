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

ANALISES_COM_FIELD = {"1 Sample T", "1 Wilcoxon", "1 Teste de Sinal", "1 Proporcao", "Intervalo de Confianca"}


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
    colunas_x: str | list[str] = Form(None),
    coluna_z: str = Form(None)
):
    try:
        form = await request.form()
        field = form.get("field") or None
        if field == "":
            field = None

        df = await ler_arquivo(arquivo)
        colunas_usadas = []

        # ✅ Trata coluna_y
        colunas_y = []
        if coluna_y and coluna_y.strip():
            colunas_y = [y.strip() for y in coluna_y.split(",") if y.strip()]
            colunas_usadas.extend(colunas_y)

        # ✅ Trata colunas_x
        if colunas_x:
            if isinstance(colunas_x, str):
                colunas_usadas.extend([x.strip() for x in colunas_x.split(",") if x.strip()])
            else:
                for item in colunas_x:
                    colunas_usadas.extend([x.strip() for x in item.split(",") if x.strip()])

        # ✅ Trata coluna_z
        if coluna_z and coluna_z.strip():
            colunas_usadas.append(coluna_z.strip())

        resultado_texto = None
        imagem_analise_base64 = None
        imagem_grafico_isolado_base64 = None

        # ✅ Executa análise
        if ferramenta and ferramenta.strip():
            funcao = ANALISES.get(ferramenta.strip())
            if not funcao:
                return JSONResponse(content={"erro": "Análise estatística desconhecida."}, status_code=400)

            if ferramenta.strip() in ANALISES_COM_FIELD:
                resultado_texto, imagem_analise_base64 = funcao(df, colunas_y, field=field)
            else:
                resultado_texto, imagem_analise_base64 = funcao(df, colunas_y or colunas_usadas)

        # ✅ Executa gráfico
        if grafico and grafico.strip():
            print(f"🎨 Gráfico solicitado: {grafico.strip()}")
            print(f"📊 Colunas usadas: {colunas_usadas}")
            funcao = GRAFICOS.get(grafico.strip())
            if not funcao:
                return JSONResponse(content={"erro": f"Gráfico {grafico.strip()} não encontrado."}, status_code=400)

            imagem_grafico_isolado_base64 = funcao(df, colunas_usadas)

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

