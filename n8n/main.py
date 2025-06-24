from fastapi import FastAPI, File, UploadFile, Form, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import traceback
import os

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

# Middleware de CORS seguro com variável ambiente opcional
allowed_origins = os.getenv(
    "CORS_ORIGINS", 
    "https://educacaopelotrabalho-production.up.railway.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
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


@app.post("/analise")
async def analisar(
    request: Request,
    arquivo: UploadFile = File(None),
    ferramenta: str = Form(None),
    grafico: str = Form(None),
    aba: str = Form(None),
    coluna_y: str = Form(None),
    colunas_x: str | list[str] = Form(None),
    coluna_z: str = Form(None),
    subgrupo: str = Form(None),
    field: str = Form(None),
    field_conf: str = Form(None),
    field_distribuicao: str = Form(None),
    field_LSE: str = Form(None),
    field_LIE: str = Form(None),
    Data: str = Form(None)
):
    try:
        if not arquivo:
            return JSONResponse({"erro": "Nenhum arquivo recebido."}, status_code=400)

        print("🚀 Nome do arquivo recebido:", arquivo.filename)
        print("🚀 Aba solicitada:", aba)

        df = await ler_arquivo(arquivo, aba)
        if df is None or df.empty:
            return JSONResponse({"erro": "Arquivo vazio ou aba inválida."}, status_code=400)

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

        print("🔍 Colunas no arquivo recebido:", df.columns.tolist())
        print("🔍 Colunas Y:", colunas_y)
        print("🔍 Colunas X:", lista_x)
        print("🔍 Colunas Z:", lista_z)
        print("🔍 Subgrupo:", subgrupo_val)
        print("🔍 Field:", field)
        print("🔍 Field_conf:", field_conf)
        print("🔍 Field distribuição:", field_distribuicao)

        resultado_texto = ""
        imagem_analise_base64 = None
        imagem_grafico_isolado_base64 = None

        if ferramenta:
            funcao = ANALISES.get(ferramenta.strip())
            if not funcao:
                return JSONResponse({"erro": f"Análise {ferramenta} desconhecida."}, status_code=400)

            disponiveis = {
                "df": df,
                "coluna_y": colunas_y[0] if colunas_y else None,
                "coluna_x": lista_x[0] if lista_x else None,
                "coluna_z": lista_z[0] if lista_z else None,
                "Data": Data,
                "lista_y": colunas_y,
                "lista_x": lista_x,
                "lista_z": lista_z,
                "subgrupo": subgrupo_val,
                "field": field,
                "field_conf": field_conf,
                "field_dist": field_distribuicao,
                "field_LSE": field_LSE,
                "field_LIE": field_LIE
            }

            permitidos = CONFIG_ANALISES.get(ferramenta.strip(), ["df"])
            args_to_pass = {k: disponiveis[k] for k in permitidos if k in disponiveis}
            resultado_texto, imagem_analise_base64 = funcao(**args_to_pass)

        if grafico:
            funcao_grafico = GRAFICOS.get(grafico.strip())
            if not funcao_grafico:
                return JSONResponse({"erro": f"Gráfico {grafico} não encontrado."}, status_code=400)

            disponiveis = {
                "df": df,
                "coluna_y": colunas_y[0] if colunas_y else None,
                "coluna_x": lista_x[0] if lista_x else None,
                "coluna_z": lista_z[0] if lista_z else None,
                "Data": Data,
                "lista_y": colunas_y,
                "lista_x": lista_x,
                "lista_z": lista_z,
                "subgrupo": subgrupo_val,
                "field": field,
                "field_conf": field_conf,
                "field_dist": field_distribuicao,
                "field_LSE": field_LSE,
                "field_LIE": field_LIE
            }

            permitidos = CONFIG_ANALISES.get(grafico.strip(), ["df"])
            args_to_pass = [df] + [disponiveis.get(p) for p in permitidos[1:]]
            imagem_grafico_isolado_base64 = funcao_grafico(*args_to_pass)

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


