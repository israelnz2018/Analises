import pandas as pd


def _validar_coluna(df, coluna, nome_campo):
    if not coluna:
        return f"Campo '{nome_campo}' eh obrigatorio."
    if coluna not in df.columns:
        return f"Coluna '{coluna}' nao encontrada na planilha."
    return None


def _converter_numerico(serie, nome_coluna):
    try:
        convertida = pd.to_numeric(serie, errors='coerce').dropna()
        if len(convertida) == 0:
            return None, f"Coluna '{nome_coluna}' nao tem valores numericos validos."
        return convertida, None
    except Exception:
        return None, f"Coluna '{nome_coluna}' nao pode ser convertida para numero."


def _estatisticas_basicas(serie_numerica):
    return {
        "n": int(len(serie_numerica)),
        "media": float(serie_numerica.mean()),
        "mediana": float(serie_numerica.median()),
        "desvio_padrao": float(serie_numerica.std()),
        "minimo": float(serie_numerica.min()),
        "maximo": float(serie_numerica.max()),
    }


def histograma_interativo(df, coluna_y, subgrupo=None):
    erro = _validar_coluna(df, coluna_y, "Variavel Y")
    if erro:
        return {"erro": erro}

    dados_y, erro = _converter_numerico(df[coluna_y], coluna_y)
    if erro:
        return {"erro": erro}

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
                "y": "Frequencia",
                "titulo": f"Histograma de {coluna_y}"
            },
            "estatisticas": _estatisticas_basicas(dados_y)
        }

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
        return {"erro": "Nao ha dados validos para nenhum subgrupo."}

    return {
        "tipo": "histograma",
        "series": series,
        "labels": {
            "x": str(coluna_y),
            "y": "Frequencia",
            "titulo": f"Histograma de {coluna_y} por {subgrupo}"
        },
        "estatisticas": _estatisticas_basicas(dados_y)
    }


GRAFICOS_INTERATIVOS = {
    "Histograma": histograma_interativo,
}


CONFIG_GRAFICOS_INTERATIVOS = {
    "Histograma": ["df", "coluna_y", "subgrupo"],
}
