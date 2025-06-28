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

CONFIG_ANALISES = {
    "Gráfico Sumario": ["df", "coluna_y"],
    "Análise de outliers": ["df", "lista_y"],
    "Correlação de person": ["df", "coluna_y", "lista_x"],
    "Matrix de dispersão": ["df", "coluna_y", "lista_x"],
    "Análise de estabilidade": ["df", "coluna_y"],
    "Análise de limpeza dos dados": ["df"],
    "Histograma": ["df", "coluna_y", "subgrupo"],
    "Pareto": ["df", "coluna_x","coluna_y", "subgrupo"],
    "Setores (Pizza)": ["df", "coluna_x","coluna_y", "subgrupo"],
    "Barras": ["df", "coluna_x","coluna_y", "subgrupo"],
    "BoxPlot": ["df", "lista_y", "subgrupo"],
    "Dispersão": ["df", "coluna_y", "coluna_x", "subgrupo"],
    "Tendência": ["df", "coluna_y", "Data", "subgrupo"],
    "Bolhas - 3D": ["df", "coluna_y", "coluna_x", "coluna_z"],
    "Superfície - 3D": ["df", "coluna_y", "coluna_x", "coluna_z"],
    "Dispersão 3D com Regressão": ["df", "coluna_y", "coluna_x", "coluna_z"],
    "1 Sample T": ["df", "coluna_y", "field", "field_conf"],
    "2 Sample T": ["df", "lista_y", "field_conf"],
    "2 Paired Test": ["df", "lista_y", "field_conf"],
    "One way ANOVA": ["df", "lista_y", "subgrupo", "field_conf"],
    "1 Wilcoxon": ["df", "coluna_y", "field", "field_conf"],
    "2 Mann-Whitney": ["df", "lista_y", "field_conf"],
    "Kruskal-Wallis": ["df", "lista_y", "subgrupo", "field_conf"],
    "Friedman Pareado": ["df", "lista_y", "subgrupo", "field_conf"],
    "1 Intervalo de Confianca": ["df", "coluna_y", "field_conf"],
    "1 Intervalo Interquartilico": ["df", "coluna_y", "field_conf"],
    "2 Varianças":["df", "lista_y", "field_conf"],
    "2 Variancas Brown-Forsythe": ["df", "lista_y", "field_conf"],
    "Bartlett": ["df", "lista_y", "subgrupo", "field_conf"],
    "Brown-Forsythe": ["df", "lista_y", "subgrupo", "field_conf"],
    "1 Intervalo de Confianca Variancia": ["df", "coluna_y", "field_conf"],
    "1 Proporcao": ["df", "coluna_y", "field_conf"],
    "2 Proporcoes": ["df", "lista_y", "field_conf"],
    "K Proporcoes": ["df", "lista_y", "field_conf"],
    "Qui-quadrado": ["df", "coluna_y", "lista_x", "subgrupo"],
    "Tipo de modelo de regressão": ["df", "coluna_y"],
    "Regressão linear simples": ["df", "coluna_y", "coluna_x"],
    "Regressão linear múltipla": ["df", "coluna_y", "lista_x"],
    "Regressão logística binária": ["df", "coluna_y", "lista_x"],
    "Regressão logística ordinal": ["df", "coluna_y", "lista_x"],
    "Regressão logística nominal": ["df", "coluna_y", "lista_x"],
    "Árvore de decisão": ["df", "coluna_y", "lista_x"],
    "Random Forest": ["df", "coluna_y", "lista_x"],
    "ARIMA": ["df", "coluna_y", "field"],
    "Holt-Winters": ["df", "coluna_y", "field"],
    "Carta I-MR": ["df", "coluna_y"],
    "Carta X-BarraR": ["df", "coluna_y", "subgrupo"],
    "Carta X-BarraS": ["df", "coluna_y", "subgrupo"], 
    "Carta P": ["df", "coluna_y", "subgrupo"],
    "Carta NP": ["df", "coluna_y", "subgrupo"],
    "Carta C": ["df", "coluna_y"],
    "Carta U": ["df", "coluna_y", "subgrupo"],
    "Teste de normalidade": ["df", "coluna_y"],
    "Análise de distribuição estatística": ["df", "coluna_y"],
    "Capabilidade - dados normais": ["df", "coluna_y", "subgrupo", "field_LIE", "field_LSE"],
    "Capabilidade - outras distribuições": ["df", "coluna_y", "subgrupo", "field_dist", "field_LIE", "field_LSE"],
    "Capabilidade - com dados transformados": ["df", "coluna_y", "subgrupo", "field_LIE", "field_LSE"],
    "Capabilidade - com dados discretizados": ["df", "coluna_y", "field_LIE", "field_LSE"],
    "Cálculo de Probabilidade": ["df", "coluna_y", "field"]
}


DICIONARIO_TERMOS = {
    "coluna_y": "Uma coluna numérica ou categórica Y selecionada pelo usuário",
    "coluna_x": "Uma coluna numérica ou categórica X selecionada pelo usuário",
    "coluna_z": "Uma coluna numérica Z selecionada pelo usuário",
    "Data": "Coluna de data ou tempo selecionada pelo usuário",
    "lista_y": "Lista de colunas numéricas Y selecionadas pelo usuário; pode conter apenas um valor se subgrupo for acionado",
    "lista_x": "Lista de colunas numéricas ou categóricas X selecionadas pelo usuário; pode conter apenas um valor se subgrupo for acionado",
    "lista_z": "Lista de colunas numéricas Z selecionadas pelo usuário; pode conter apenas um valor se subgrupo for acionado",
    "subgrupo": "Coluna categórica de subgrupos selecionada pelo usuário",
    "field": "Valor de referência ou parâmetro extra (exemplo: valor de H0)",
    "field_conf": "Nível de confiança em porcentagem (exemplo: 95%)",
    "field_dist": "Tipo de distribuição (exemplo: Normal, Weibull)",
    "field_LSE": "Valor do limite superior de engenharia",
    "field_LIE": "Valor do limite inferior de engenharia"
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

        # ✅ PRINT DEBUG COMPLETO
        print("\n====================== INÍCIO DEBUG /ANALISE ======================")
        print("📂 Nome do arquivo recebido:", arquivo.filename)
        print("📑 Aba solicitada:", aba)
        print("🧾 Colunas no DataFrame:", df.columns.tolist())
        print("➡ coluna_y recebida:", coluna_y)
        print("➡ lista_y recebida:", lista_y_processada)
        print("➡ coluna_x recebida:", coluna_x)
        print("➡ lista_x recebida:", lista_x_processada)
        print("➡ coluna_z recebida:", coluna_z)
        print("➡ lista_z recebida:", lista_z)
        print("➡ subgrupo recebido:", subgrupo_val)
        print("➡ field:", field)
        print("➡ field_conf:", field_conf)
        print("➡ field_dist:", field_dist)
        print("➡ field_LSE:", field_LSE)
        print("➡ field_LIE:", field_LIE)
        print("➡ Data:", Data)
        print("🔧 ferramenta:", ferramenta)
        print("🔧 grafico:", grafico)
        print("======================= FIM DEBUG /ANALISE ==========================\n")

        resultado_texto = ""
        imagem_analise_base64 = None
        imagem_grafico_isolado_base64 = None

        if ferramenta:
            funcao = ANALISES.get(ferramenta.strip())
            if not funcao:
                return JSONResponse({"erro": f"Análise {ferramenta} desconhecida."}, status_code=400)

            disponiveis = {
                "df": df,
                "coluna_y": coluna_y.strip() if coluna_y else None,  # ✅ garante coluna_y como string única
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

            permitidos = CONFIG_ANALISES.get(grafico.strip(), ["df", "coluna_y"])
            args_to_pass = {k: disponiveis[k] for k in permitidos if k in disponiveis}

            import inspect
            params_aceitos = inspect.signature(funcao_grafico).parameters
            args_filtrados = {k: v for k, v in args_to_pass.items() if k in params_aceitos}

            imagem_grafico_isolado_base64 = funcao_grafico(**args_filtrados)

        return {
            "analise": resultado_texto,
            "grafico_base64": imagem_analise_base64 or [],
            "grafico_isolado_base64": imagem_grafico_isolado_base64,
            "colunas_utilizadas": colunas_usadas
        }

    
@app.post("/personalizar-grafico")
async def personalizar_grafico(
    request: Request,
    arquivo: UploadFile = File(None),
    aba: str = Form(None),
    grafico: str = Form(None),
    coluna_y: str = Form(None),
    coluna_x: str = Form(None),
    coluna_z: str = Form(None),
    subgrupo: str = Form(None),
    field: str = Form(None),
    field_conf: str = Form(None),
    field_dist: str = Form(None),
    field_LSE: str = Form(None),
    field_LIE: str = Form(None),
    Data: str = Form(None),
    cor: str = Form(None),
    titulo_x: str = Form(None),
    titulo_y: str = Form(None),
    titulo_grafico: str = Form(None),
    tamanho_fonte: str = Form(None),
    inclinacao_x: str = Form(None),
    inclinacao_y: str = Form(None),
    espessura: str = Form(None)
):
    try:
        if not arquivo:
            return JSONResponse({"erro": "Nenhum arquivo recebido."}, status_code=400)

        # 🔍 DEBUG INICIO
        print("\n====================== INÍCIO DEBUG /PERSONALIZAR-GRAFICO ======================")
        print("📂 Nome do arquivo recebido:", arquivo.filename)
        print("📑 Aba solicitada:", aba)
        print("🎨 Gráfico solicitado:", grafico)
        print("➡ coluna_y:", coluna_y)
        print("➡ coluna_x:", coluna_x)
        print("➡ coluna_z:", coluna_z)
        print("➡ subgrupo:", subgrupo)
        print("➡ field:", field)
        print("➡ field_conf:", field_conf)
        print("➡ field_dist:", field_dist)
        print("➡ field_LSE:", field_LSE)
        print("➡ field_LIE:", field_LIE)
        print("➡ Data:", Data)
        print("➡ cor:", cor)
        print("➡ titulo_x:", titulo_x)
        print("➡ titulo_y:", titulo_y)
        print("➡ titulo_grafico:", titulo_grafico)
        print("➡ tamanho_fonte:", tamanho_fonte)
        print("➡ inclinacao_x:", inclinacao_x)
        print("➡ inclinacao_y:", inclinacao_y)
        print("➡ espessura:", espessura)
        print("======================= FIM DEBUG /PERSONALIZAR-GRAFICO ==========================\n")

        from leitura import ler_arquivo  # ✅ garante import dentro da função se necessário
        df = await ler_arquivo(arquivo, aba)
        if df is None or df.empty:
            return JSONResponse({"erro": "Arquivo vazio ou aba inválida."}, status_code=400)

        # 🔍 Busca a função do gráfico personalizado diretamente no dicionário
        funcao_grafico = GRAFICOS.get(grafico.strip())
        if not funcao_grafico:
            return JSONResponse({"erro": f"Gráfico {grafico} não encontrado no dicionário GRAFICOS."}, status_code=400)

        # 🔧 Filtra apenas os parâmetros aceitos pela função do gráfico
        params_aceitos = inspect.signature(funcao_grafico).parameters
        args_to_pass = {k: v for k, v in {
            "df": df,
            "coluna_y": coluna_y,
            "coluna_x": coluna_x,
            "coluna_z": coluna_z,
            "subgrupo": subgrupo,
            "field": field,
            "field_conf": field_conf,
            "field_dist": field_dist,
            "field_LSE": field_LSE,
            "field_LIE": field_LIE,
            "Data": Data,
            "cor": cor,
            "titulo_x": titulo_x,
            "titulo_y": titulo_y,
            "titulo_grafico": titulo_grafico,
            "tamanho_fonte": tamanho_fonte,
            "inclinacao_x": inclinacao_x,
            "inclinacao_y": inclinacao_y,
            "espessura": espessura
        }.items() if k in params_aceitos}

        # 🖼️ Chama a função do gráfico com os parâmetros filtrados
        imagem_grafico_isolado_base64 = funcao_grafico(**args_to_pass)

        return {
            "grafico_isolado_base64": imagem_grafico_isolado_base64
        }

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({
            "erro": "Erro interno ao personalizar gráfico.",
            "detalhe": str(e),
            "traceback": tb
        }, status_code=500)
