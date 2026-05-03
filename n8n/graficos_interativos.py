# -*- coding: utf-8 -*-
import math
import pandas as pd

MAX_NIVEIS_SUBGRUPO = 12


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
    s = pd.Series(serie_numerica).dropna()
    if len(s) == 0:
        return None
    dp = float(s.std()) if len(s) > 1 else 0.0
    return {
        "n": int(len(s)),
        "media": float(s.mean()),
        "mediana": float(s.median()),
        "desvio_padrao": dp,
        "minimo": float(s.min()),
        "maximo": float(s.max()),
    }


def _normalizar_subgrupo(df, coluna_y, subgrupo, max_niveis=MAX_NIVEIS_SUBGRUPO):
    avisos = []
    if not subgrupo:
        return None, [], avisos
    if subgrupo not in df.columns:
        avisos.append(f"Subgrupo '{subgrupo}' nao encontrado. Ignorado.")
        return None, [], avisos
    if subgrupo == coluna_y:
        avisos.append("Subgrupo igual a coluna Y. Ignorado.")
        return None, [], avisos

    niveis = df[subgrupo].dropna().astype(str).unique().tolist()
    if len(niveis) == 0:
        avisos.append(f"Subgrupo '{subgrupo}' nao tem valores. Ignorado.")
        return None, [], avisos
    if len(niveis) == 1:
        avisos.append(f"Subgrupo '{subgrupo}' tem apenas 1 nivel. Ignorado.")
        return None, [], avisos
    if len(niveis) > max_niveis:
        contagem = df[subgrupo].dropna().astype(str).value_counts()
        topN = contagem.head(max_niveis - 1).index.tolist()
        avisos.append(
            f"Subgrupo '{subgrupo}' tem {len(niveis)} niveis. "
            f"Mostrando os {max_niveis - 1} mais frequentes; demais agrupados em 'Outros'."
        )
        return subgrupo, topN + ["__OUTROS__"], avisos

    try:
        niveis_ordenados = sorted(niveis, key=lambda x: float(x))
    except (ValueError, TypeError):
        niveis_ordenados = sorted(niveis)
    return subgrupo, niveis_ordenados, avisos


def _serie_filtrada_por_grupo(df, coluna_y, subgrupo, nivel, niveis_explicitos=None):
    if nivel == "__OUTROS__":
        if not niveis_explicitos:
            return pd.Series([], dtype=float)
        mask = ~df[subgrupo].astype(str).isin(niveis_explicitos)
    else:
        mask = df[subgrupo].astype(str) == str(nivel)
    return pd.to_numeric(df.loc[mask, coluna_y], errors='coerce').dropna()


def _bins_compartilhados(serie_global, n_bins=None):
    s = pd.Series(serie_global).dropna()
    if len(s) == 0:
        return None
    minimo, maximo = float(s.min()), float(s.max())
    if minimo == maximo:
        return {"start": minimo - 0.5, "end": maximo + 0.5, "size": 1.0}
    if n_bins is None:
        n_bins = max(5, min(30, int(math.ceil(math.log2(len(s)) + 1))))
    size = (maximo - minimo) / n_bins
    return {"start": minimo, "end": maximo + size * 0.001, "size": size}


def _estatisticas_por_grupo(df, coluna_y, subgrupo, niveis):
    out = {}
    niveis_explicitos = [n for n in niveis if n != "__OUTROS__"]
    for nivel in niveis:
        valores = _serie_filtrada_por_grupo(df, coluna_y, subgrupo, nivel, niveis_explicitos)
        st = _estatisticas_basicas(valores)
        if st:
            rotulo = "Outros" if nivel == "__OUTROS__" else str(nivel)
            out[rotulo] = st
    return out


def _payload(tipo, series, labels, estat_global=None, estat_grupo=None, config=None, avisos=None):
    return {
        "tipo": tipo,
        "series": series,
        "labels": labels,
        "config": config or {},
        "estatisticas": {
            "global": estat_global,
            "por_grupo": estat_grupo or {},
        },
        "avisos": avisos or [],
    }


def histograma_interativo(df, coluna_y, subgrupo=None):
    erro = _validar_coluna(df, coluna_y, "Variavel Y")
    if erro:
        return {"erro": erro}
    dados_y, erro = _converter_numerico(df[coluna_y], coluna_y)
    if erro:
        return {"erro": erro}

    bins = _bins_compartilhados(dados_y)
    sg, niveis, avisos = _normalizar_subgrupo(df, coluna_y, subgrupo)

    if sg is None:
        series = [{"nome": str(coluna_y), "valores": dados_y.tolist()}]
        titulo = f"Histograma de {coluna_y}"
        estat_grupo = {}
    else:
        series = []
        niveis_explicitos = [n for n in niveis if n != "__OUTROS__"]
        for nivel in niveis:
            valores = _serie_filtrada_por_grupo(df, coluna_y, sg, nivel, niveis_explicitos)
            if len(valores) == 0:
                continue
            rotulo = "Outros" if nivel == "__OUTROS__" else str(nivel)
            series.append({"nome": rotulo, "valores": valores.tolist()})
        if not series:
            return {"erro": "Nao ha dados validos para nenhum subgrupo."}
        titulo = f"Histograma de {coluna_y} por {sg}"
        estat_grupo = _estatisticas_por_grupo(df, coluna_y, sg, niveis)

    return _payload(
        tipo="histograma",
        series=series,
        labels={"x": str(coluna_y), "y": "Frequencia", "titulo": titulo},
        estat_global=_estatisticas_basicas(dados_y),
        estat_grupo=estat_grupo,
        config={
            "bins": bins,
            "barmode": "overlay" if len(series) > 1 else "group",
        },
        avisos=avisos,
    )


GRAFICOS_INTERATIVOS = {
    "Histograma": histograma_interativo,
}

CONFIG_GRAFICOS_INTERATIVOS = {
    "Histograma": ["df", "coluna_y", "subgrupo"],
}
