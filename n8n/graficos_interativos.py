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

# ============================================================
# HISTOGRAMA
# ============================================================
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

# ============================================================
# PARETO
# ============================================================

def pareto_interativo(df, coluna_x, coluna_y=None, subgrupo=None):
    # 1) validar X
    erro = _validar_coluna(df, coluna_x, "Categoria X")
    if erro:
        return {"erro": erro}

    # 2) decide se usa contagem (sem Y) ou soma (com Y)
    if coluna_y and coluna_y in df.columns:
        df_calc = df[[coluna_x, coluna_y]].copy()
        df_calc[coluna_y] = pd.to_numeric(df_calc[coluna_y], errors='coerce')
        df_calc = df_calc.dropna(subset=[coluna_y, coluna_x])
        if len(df_calc) == 0:
            return {"erro": f"Coluna Y '{coluna_y}' nao tem valores numericos validos."}
        agregado = df_calc.groupby(df_calc[coluna_x].astype(str))[coluna_y].sum()
        rotulo_y = f"Soma de {coluna_y}"
    else:
        agregado = df[coluna_x].dropna().astype(str).value_counts()
        rotulo_y = "Frequencia"

    if len(agregado) == 0:
        return {"erro": "Sem dados validos para o Pareto."}

    # 3) ordena decrescente
    agregado = agregado.sort_values(ascending=False)
    categorias = [str(c) for c in agregado.index.tolist()]
    valores_total = [float(v) for v in agregado.values.tolist()]

    # 4) cumulativa percentual
    total = sum(valores_total)
    cumulativa_pct = []
    acumulado = 0.0
    for v in valores_total:
        acumulado += v
        cumulativa_pct.append(round((acumulado / total) * 100, 2) if total > 0 else 0)

    # 5) subgrupo (opcional) -> empilhamento por nivel
    series = []
    sg, niveis, avisos = _normalizar_subgrupo(df, coluna_x, subgrupo)

    if sg is None:
        series.append({
            "nome": rotulo_y,
            "categorias": categorias,
            "valores": valores_total,
        })
    else:
        niveis_explicitos = [n for n in niveis if n != "__OUTROS__"]
        for nivel in niveis:
            if nivel == "__OUTROS__":
                mask = ~df[sg].astype(str).isin(niveis_explicitos)
            else:
                mask = df[sg].astype(str) == str(nivel)
            sub_df = df[mask]
            if coluna_y and coluna_y in df.columns:
                sub_df = sub_df[[coluna_x, coluna_y]].copy()
                sub_df[coluna_y] = pd.to_numeric(sub_df[coluna_y], errors='coerce')
                sub_df = sub_df.dropna(subset=[coluna_y, coluna_x])
                agg = sub_df.groupby(sub_df[coluna_x].astype(str))[coluna_y].sum()
            else:
                agg = sub_df[coluna_x].dropna().astype(str).value_counts()
            valores_nivel = [float(agg.get(c, 0)) for c in categorias]
            if sum(valores_nivel) > 0:
                rotulo = "Outros" if nivel == "__OUTROS__" else str(nivel)
                series.append({
                    "nome": rotulo,
                    "categorias": categorias,
                    "valores": valores_nivel,
                })
        if not series:
            return {"erro": "Sem dados validos por subgrupo."}

    # 6) serie da linha cumulativa (sempre por ultimo)
    series.append({
        "nome": "% Cumulativa",
        "categorias": categorias,
        "valores": cumulativa_pct,
        "tipo_serie": "cumulativa",
    })

    titulo = f"Pareto de {coluna_x}" + (f" por {sg}" if sg else "")

    return _payload(
        tipo="pareto",
        series=series,
        labels={"x": str(coluna_x), "y": rotulo_y, "titulo": titulo},
        estat_global=None,
        estat_grupo=None,
        config={
            "categorias": categorias,
            "total": total,
            "barmode": "stack" if len(series) > 2 else "group",
        },
        avisos=avisos,
    )


# ============================================================
# SETORES (PIZZA)
# ============================================================

def pizza_interativo(df, coluna_x, coluna_y=None, subgrupo=None):
    erro = _validar_coluna(df, coluna_x, "Categoria X")
    if erro:
        return {"erro": erro}

    avisos = []
    sg, niveis, av_sg = _normalizar_subgrupo(df, coluna_x, subgrupo)
    avisos.extend(av_sg)

    df_work = df.copy()
    df_work = df_work.dropna(subset=[coluna_x])

    if sg:
        df_work['_chave'] = df_work[coluna_x].astype(str) + ' | ' + df_work[sg].astype(str)
        chave = '_chave'
        avisos.append("Subgrupo combinado com X como rotulo de cada fatia.")
    else:
        df_work['_chave'] = df_work[coluna_x].astype(str)
        chave = '_chave'

    if coluna_y and coluna_y in df.columns:
        df_work[coluna_y] = pd.to_numeric(df_work[coluna_y], errors='coerce')
        df_work = df_work.dropna(subset=[coluna_y])
        if len(df_work) == 0:
            return {"erro": f"Coluna Y '{coluna_y}' nao tem valores numericos validos."}
        agregado = df_work.groupby(chave)[coluna_y].sum().sort_values(ascending=False)
        rotulo_y = f"Soma de {coluna_y}"
    else:
        agregado = df_work[chave].value_counts()
        rotulo_y = "Frequencia"

    if len(agregado) == 0:
        return {"erro": "Sem dados validos para o grafico de Pizza."}

    labels = [str(l) for l in agregado.index.tolist()]
    valores = [float(v) for v in agregado.values.tolist()]

    titulo = f"Setores de {coluna_x}" + (f" por {sg}" if sg else "")

    return _payload(
        tipo="pizza",
        series=[{"labels": labels, "valores": valores, "nome": rotulo_y}],
        labels={"titulo": titulo},
        estat_global=None,
        estat_grupo=None,
        config={"n_fatias": len(labels)},
        avisos=avisos,
    )


# ============================================================
# BARRAS
# ============================================================

def barras_interativo(df, coluna_x, coluna_y=None, subgrupo=None):
    erro = _validar_coluna(df, coluna_x, "Categoria X")
    if erro:
        return {"erro": erro}

    avisos = []
    sg, niveis, av_sg = _normalizar_subgrupo(df, coluna_x, subgrupo)
    avisos.extend(av_sg)

    df_work = df.copy()
    df_work = df_work.dropna(subset=[coluna_x])
    df_work['_chave'] = df_work[coluna_x].astype(str)

    # ordenacao das categorias globais
    if coluna_y and coluna_y in df.columns:
        df_work[coluna_y] = pd.to_numeric(df_work[coluna_y], errors='coerce')
        df_work = df_work.dropna(subset=[coluna_y])
        if len(df_work) == 0:
            return {"erro": f"Coluna Y '{coluna_y}' nao tem valores numericos validos."}
        ordem = df_work.groupby('_chave')[coluna_y].sum().sort_values(ascending=False).index.tolist()
        rotulo_y = f"Soma de {coluna_y}"
    else:
        ordem = df_work['_chave'].value_counts().index.tolist()
        rotulo_y = "Frequencia"

    categorias = [str(c) for c in ordem]

    series = []

    if sg is None:
        if coluna_y and coluna_y in df.columns:
            agg = df_work.groupby('_chave')[coluna_y].sum()
        else:
            agg = df_work['_chave'].value_counts()
        valores = [float(agg.get(c, 0)) for c in categorias]
        series.append({"nome": rotulo_y, "categorias": categorias, "valores": valores})
    else:
        niveis_explicitos = [n for n in niveis if n != "__OUTROS__"]
        for nivel in niveis:
            if nivel == "__OUTROS__":
                mask = ~df_work[sg].astype(str).isin(niveis_explicitos)
            else:
                mask = df_work[sg].astype(str) == str(nivel)
            sub_df = df_work[mask]
            if coluna_y and coluna_y in df.columns:
                agg = sub_df.groupby('_chave')[coluna_y].sum()
            else:
                agg = sub_df['_chave'].value_counts()
            valores = [float(agg.get(c, 0)) for c in categorias]
            if sum(valores) > 0:
                rotulo = "Outros" if nivel == "__OUTROS__" else str(nivel)
                series.append({"nome": rotulo, "categorias": categorias, "valores": valores})
        if not series:
            return {"erro": "Sem dados validos por subgrupo."}

    titulo = f"Barras de {coluna_x}" + (f" por {sg}" if sg else "")

    return _payload(
        tipo="barras",
        series=series,
        labels={"x": str(coluna_x), "y": rotulo_y, "titulo": titulo},
        estat_global=None,
        estat_grupo=None,
        config={"barmode": "group" if len(series) > 1 else "group"},
        avisos=avisos,
    )


# ============================================================
# BOXPLOT
# ============================================================

def boxplot_interativo(df, lista_y, subgrupo=None):
    if not lista_y:
        return {"erro": "Informe ao menos uma variavel Y."}

    avisos = []
    series = []

    colunas_validas = []
    for col in lista_y:
        erro = _validar_coluna(df, col, f"Y ({col})")
        if erro:
            avisos.append(erro)
            continue
        dados, erro = _converter_numerico(df[col], col)
        if erro:
            avisos.append(erro)
            continue
        colunas_validas.append((col, dados))

    if not colunas_validas:
        return {"erro": "Nenhuma variavel Y valida."}

    sg, niveis, av_sg = _normalizar_subgrupo(df, lista_y[0], subgrupo)
    avisos.extend(av_sg)

    if sg is None:
        for nome, dados in colunas_validas:
            series.append({
                "nome": nome,
                "valores": dados.tolist(),
            })
    else:
        niveis_explicitos = [n for n in niveis if n != "__OUTROS__"]
        for nome, _ in colunas_validas:
            for nivel in niveis:
                valores = _serie_filtrada_por_grupo(
                    df, nome, sg, nivel, niveis_explicitos
                )
                if len(valores) > 0:
                    rotulo = "Outros" if nivel == "__OUTROS__" else str(nivel)
                    label = f"{nome} | {rotulo}" if len(colunas_validas) > 1 else rotulo
                    series.append({"nome": label, "valores": valores.tolist()})

    if not series:
        return {"erro": "Sem dados validos para o BoxPlot."}

    titulo = "BoxPlot de " + ", ".join([c for c, _ in colunas_validas])
    if sg:
        titulo += f" por {sg}"

    # estatisticas globais usando a primeira coluna valida
    estat_global = _estatisticas_basicas(colunas_validas[0][1])

    return _payload(
        tipo="boxplot",
        series=series,
        labels={
            "x": sg or "Variavel",
            "y": colunas_validas[0][0],
            "titulo": titulo,
        },
        estat_global=estat_global,
        estat_grupo=None,
        config={"n_series": len(series)},
        avisos=avisos,
    )


# Atualize os dois dicionários assim:

GRAFICOS_INTERATIVOS = {
    "Histograma":      histograma_interativo,
    "Pareto":          pareto_interativo,
    "Setores (Pizza)": pizza_interativo,
    "Barras":          barras_interativo,
    "BoxPlot":         boxplot_interativo,
}

CONFIG_GRAFICOS_INTERATIVOS = {
    "Histograma":      ["df", "coluna_y", "subgrupo"],
    "Pareto":          ["df", "coluna_x", "coluna_y", "subgrupo"],
    "Setores (Pizza)": ["df", "coluna_x", "coluna_y", "subgrupo"],
    "Barras":          ["df", "coluna_x", "coluna_y", "subgrupo"],
    "BoxPlot":         ["df", "lista_y", "subgrupo"],
}
