# graficos_interativos.py
# ════════════════════════════════════════════════════════════════════
# Módulo de gráficos interativos (retornam JSON para Plotly.js no front)
#
# Segue o padrão modular dos demais módulos do projeto:
#   - Inferencial.py, Capabilidade.py, etc. retornam (texto, imagem)
#   - Este módulo retorna JSON estruturado para renderização interativa
#
# Para adicionar um novo gráfico:
#   1) Crie a função (ex: pareto_interativo)
#   2) Registre no dicionário GRAFICOS_INTERATIVOS no final do arquivo
#   3) Pronto. O endpoint /v2/grafico-interativo nunca precisa mudar.
# ════════════════════════════════════════════════════════════════════
 
import pandas as pd
 
 
# ====================================================================
# UTILITÁRIOS COMPARTILHADOS
# ====================================================================
 
def _validar_coluna(df, coluna, nome_campo):
    """Verifica se a coluna existe no DataFrame."""
    if not coluna:
        return f"⚠ Campo '{nome_campo}' é obrigatório."
    if coluna not in df.columns:
        return f"⚠ Coluna '{coluna}' não encontrada na planilha."
    return None
 
 
def _converter_numerico(serie, nome_coluna):
    """Tenta converter uma série pandas para numérico."""
    try:
        convertida = pd.to_numeric(serie, errors='coerce').dropna()
        if len(convertida) == 0:
            return None, f"⚠ Coluna '{nome_coluna}' não tem valores numéricos válidos."
        return convertida, None
    except Exception:
        return None, f"⚠ Coluna '{nome_coluna}' não pode ser convertida para número."
 
 
def _estatisticas_basicas(serie_numerica):
    """Retorna estatísticas descritivas de uma série numérica."""
    return {
        "n": int(len(serie_numerica)),
        "media": float(serie_numerica.mean()),
        "mediana": float(serie_numerica.median()),
        "desvio_padrao": float(serie_numerica.std()),
        "minimo": float(serie_numerica.min()),
        "maximo": float(serie_numerica.max()),
    }
 
 
# ====================================================================
# HISTOGRAMA INTERATIVO
# ====================================================================
 
def histograma_interativo(df, coluna_y, subgrupo=None):
    """
    Retorna dados estruturados para renderizar Histograma no Plotly.js.
    
    Estrutura de retorno:
      {
        "tipo": "histograma",
        "series": [{nome, valores}, ...],  # 1 série sem subgrupo, várias com
        "labels": {x, y, titulo},
        "estatisticas": {n, media, mediana, dp, min, max}
      }
    """
    # Validações
    erro = _validar_coluna(df, coluna_y, "Variável Y")
    if erro:
        return {"erro": erro}
 
    # Converte Y para numérico
    dados_y, erro = _converter_numerico(df[coluna_y], coluna_y)
    if erro:
        return {"erro": erro}
 
    # Caso 1: sem subgrupo — um histograma único
    if not subgrupo or subgrupo not in df.columns:
        return {
            "tipo": "histograma",
            "series": [
                {
                    "nome": str(coluna_y),
                    "valores": dados_y.tolist(),
                }
            ],
            "labels": {
                "x": str(coluna_y),
                "y": "Frequência",
                "titulo": f"Histograma de {coluna_y}"
            },
            "estatisticas": _estatisticas_basicas(dados_y)
        }
 
    # Caso 2: com subgrupo — uma série por nível
    series = []
    for nivel in df[subgrupo].dropna().unique():
        valores_nivel = df[df[subgrupo] == nivel][coluna_y]
        valores_nivel = pd.to_numeric(valores_nivel, errors='coerce').dropna()
        if len(valores_nivel) > 0:
            series.append({
                "nome": str(nivel),
                "valores": valores_nivel.tolist(),
            })
 
    if not series:
        return {"erro": "Não há dados válidos para nenhum subgrupo."}
 
    return {
        "tipo": "histograma",
        "series": series,
        "labels": {
            "x": str(coluna_y),
            "y": "Frequência",
            "titulo": f"Histograma de {coluna_y} por {subgrupo}"
        },
        "estatisticas": _estatisticas_basicas(dados_y)
    }
 
 
# ====================================================================
# DICIONÁRIO DE EXPORTAÇÃO
# Registre cada novo gráfico interativo aqui. O endpoint usa este
# dicionário para encontrar a função certa.
# ====================================================================
GRAFICOS_INTERATIVOS = {
    "Histograma": histograma_interativo,
    # Adicione aqui conforme migrar mais gráficos:
    # "Pareto": pareto_interativo,
    # "BoxPlot": boxplot_interativo,
    # "Barras": barras_interativo,
    # "Setores (Pizza)": pizza_interativo,
    # "Dispersão": dispersao_interativo,
    # "Tendência": tendencia_interativo,
    # ...
}
 
 
# ====================================================================
# CONFIGURAÇÃO DE PARÂMETROS
# Diz ao endpoint quais campos do form passar para cada função.
# Mesmo padrão do CONFIG_ANALISES do main.py.
# ====================================================================
CONFIG_GRAFICOS_INTERATIVOS = {
    "Histograma": ["df", "coluna_y", "subgrupo"],
    # Adicione aqui conforme registrar novos:
    # "Pareto": ["df", "coluna_x", "coluna_y", "subgrupo"],
    # "BoxPlot": ["df", "lista_y", "subgrupo"],
    # ...
}
 

Passo 2 — Adicionar import + endpoint no main.py
Adicione o import no topo do main.py (junto com os outros imports de módulos):
from graficos_interativos import GRAFICOS_INTERATIVOS, CONFIG_GRAFICOS_INTERATIVOS
 
E cole o endpoint abaixo em qualquer lugar após as outras rotas /v2/:
 
# ════════════════════════════════════════════════════════════════════
# ENDPOINT — GRÁFICO INTERATIVO (versão modular)
# Adicionar este bloco ao main.py do Railway.
#
# IMPORTANTE: este endpoint NÃO MUDA quando você adiciona novos gráficos.
# Para adicionar um gráfico novo, basta editar o graficos_interativos.py.
# ════════════════════════════════════════════════════════════════════
 
# === IMPORT (adicionar junto com os outros imports no topo do main.py) ===
 
from graficos_interativos import GRAFICOS_INTERATIVOS, CONFIG_GRAFICOS_INTERATIVOS
 
 
# === ENDPOINT (adicionar em qualquer lugar após as outras rotas /v2/) ===
 
@app.post("/v2/grafico-interativo")
async def grafico_interativo(
    request: Request,
    arquivo: UploadFile = File(None),
    aba: str = Form(None),
    grafico: str = Form(None),
    coluna_y: str = Form(None),
    coluna_x: str = Form(None),
    coluna_z: str = Form(None),
    lista_y: str = Form(None),
    lista_x: str = Form(None),
    subgrupo: str = Form(None),
    field: str = Form(None),
    Data: str = Form(None),
):
    """
    Endpoint enxuto que delega a renderização para o módulo graficos_interativos.
    Retorna JSON estruturado para o Plotly.js renderizar no front.
    """
    try:
        # Validações básicas
        if arquivo is None:
            return JSONResponse({"erro": "Envie um arquivo Excel."}, status_code=400)
 
        if not grafico:
            return JSONResponse({"erro": "Informe o tipo de gráfico."}, status_code=400)
 
        # Verifica se o gráfico está registrado
        if grafico not in GRAFICOS_INTERATIVOS:
            disponiveis = ", ".join(GRAFICOS_INTERATIVOS.keys())
            return JSONResponse(
                {"erro": f"Gráfico '{grafico}' não suportado em modo interativo. Disponíveis: {disponiveis}"},
                status_code=400
            )
 
        # Lê o arquivo
        import pandas as pd
        import io
        file_bytes = await arquivo.read()
        if arquivo.filename.endswith(".xlsx") or arquivo.filename.endswith(".xlsm"):
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl", sheet_name=aba)
        elif arquivo.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            return JSONResponse({"erro": "Formato não suportado. Use .xlsx, .xlsm ou .csv."}, status_code=400)
 
        # Processa lista_y / lista_x (vêm como string separada por vírgula)
        lista_y_proc = [v.strip() for v in lista_y.split(",")] if lista_y else None
        lista_x_proc = [v.strip() for v in lista_x.split(",")] if lista_x else None
 
        # Monta o dicionário com todos os parâmetros disponíveis
        disponiveis = {
            "df": df,
            "coluna_y": coluna_y.strip() if coluna_y else None,
            "coluna_x": coluna_x.strip() if coluna_x else None,
            "coluna_z": coluna_z.strip() if coluna_z else None,
            "subgrupo": subgrupo.strip() if subgrupo else None,
            "lista_y": lista_y_proc,
            "lista_x": lista_x_proc,
            "field": field,
            "Data": Data,
        }
 
        # Pega a config de quais parâmetros este gráfico precisa
        parametros_necessarios = CONFIG_GRAFICOS_INTERATIVOS.get(grafico, ["df"])
        argumentos = {p: disponiveis.get(p) for p in parametros_necessarios}
 
        # Chama a função do gráfico interativo
        funcao = GRAFICOS_INTERATIVOS[grafico]
        resultado = funcao(**argumentos)
 
        # Retorna o resultado (que é um dict)
        return JSONResponse(resultado)
 
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return JSONResponse({
            "erro": "Erro interno ao gerar gráfico interativo.",
            "detalhe": str(e),
            "traceback": tb
        }, status_code=500)
 
