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


# ============================================================
# DISPERSAO
# ============================================================

def dispersao_interativo(df, coluna_y, coluna_x, subgrupo=None):
    erro = _validar_coluna(df, coluna_y, "Variavel Y")
    if erro:
        return {"erro": erro}
    erro = _validar_coluna(df, coluna_x, "Variavel X")
    if erro:
        return {"erro": erro}

    cols = [coluna_x, coluna_y]
    if subgrupo and subgrupo in df.columns and subgrupo not in cols:
        cols.append(subgrupo)
    df_work = df[cols].copy()
    df_work[coluna_x] = pd.to_numeric(df_work[coluna_x], errors='coerce')
    df_work[coluna_y] = pd.to_numeric(df_work[coluna_y], errors='coerce')
    df_work = df_work.dropna(subset=[coluna_x, coluna_y])

    if len(df_work) < 2:
        return {"erro": "Dados insuficientes (precisa de ao menos 2 pontos numericos)."}

    avisos = []
    sg, niveis, av_sg = _normalizar_subgrupo(df, coluna_x, subgrupo)
    avisos.extend(av_sg)

    def _tendencia(xv, yv):
        n = len(xv)
        if n < 2:
            return None
        x = pd.Series(xv).astype(float)
        y = pd.Series(yv).astype(float)
        if x.std() == 0 or y.std() == 0:
            return None
        denom_x = ((x - x.mean()) ** 2).sum()
        if denom_x == 0:
            return None
        slope = float(((x - x.mean()) * (y - y.mean())).sum() / denom_x)
        intercept = float(y.mean() - slope * x.mean())
        denom_r = (((x - x.mean()) ** 2).sum() ** 0.5) * (((y - y.mean()) ** 2).sum() ** 0.5)
        r = float(((x - x.mean()) * (y - y.mean())).sum() / denom_r) if denom_r > 0 else 0.0
        x_min, x_max = float(x.min()), float(x.max())
        return {
            "x": [x_min, x_max],
            "y": [slope * x_min + intercept, slope * x_max + intercept],
            "slope": slope,
            "intercept": intercept,
            "r": r,
            "r2": r * r,
            "n": int(n),
        }

    series = []
    if sg is None:
        x_vals = df_work[coluna_x].tolist()
        y_vals = df_work[coluna_y].tolist()
        series.append({
            "nome": f"{coluna_y} vs {coluna_x}",
            "x": x_vals,
            "y": y_vals,
            "tendencia": _tendencia(x_vals, y_vals),
        })
    else:
        niveis_explicitos = [n for n in niveis if n != "__OUTROS__"]
        for nivel in niveis:
            if nivel == "__OUTROS__":
                mask = ~df_work[sg].astype(str).isin(niveis_explicitos)
            else:
                mask = df_work[sg].astype(str) == str(nivel)
            sub = df_work[mask]
            if len(sub) == 0:
                continue
            rotulo = "Outros" if nivel == "__OUTROS__" else str(nivel)
            x_vals = sub[coluna_x].tolist()
            y_vals = sub[coluna_y].tolist()
            series.append({
                "nome": rotulo,
                "x": x_vals,
                "y": y_vals,
                "tendencia": _tendencia(x_vals, y_vals),
            })
        if not series:
            return {"erro": "Sem dados validos por subgrupo."}

    tend_global = _tendencia(df_work[coluna_x].tolist(), df_work[coluna_y].tolist())
    titulo = f"Dispersao de {coluna_y} vs {coluna_x}" + (f" por {sg}" if sg else "")

    return _payload(
        tipo="dispersao",
        series=series,
        labels={"x": str(coluna_x), "y": str(coluna_y), "titulo": titulo},
        estat_global=None,
        estat_grupo=None,
        config={"tendencia_global": tend_global},
        avisos=avisos,
    )


# ============================================================
# TENDENCIA
# ============================================================

def tendencia_interativo(df, coluna_y, Data=None, subgrupo=None):
    erro = _validar_coluna(df, coluna_y, "Variavel Y")
    if erro:
        return {"erro": erro}

    avisos = []

    cols = [coluna_y]
    if Data and Data in df.columns:
        cols.insert(0, Data)
    if subgrupo and subgrupo in df.columns and subgrupo not in cols:
        cols.append(subgrupo)
    df_work = df[cols].copy()

    # tratar a coluna de Data
    usa_data = bool(Data and Data in df.columns)
    if usa_data:
        try:
            convertida = pd.to_datetime(df_work[Data], errors='coerce', dayfirst=False)
            if convertida.isna().sum() > len(convertida) / 2:
                convertida = pd.to_datetime(df_work[Data], errors='coerce', dayfirst=True)
            if convertida.isna().sum() > len(convertida) / 2:
                avisos.append(f"Coluna '{Data}' nao reconhecida como data; usando como sequencia.")
                df_work['_eixo_x'] = df_work[Data].astype(str)
                tipo_x = "categoria"
            else:
                df_work['_eixo_x'] = convertida
                tipo_x = "data"
        except Exception:
            df_work['_eixo_x'] = df_work[Data].astype(str)
            tipo_x = "categoria"
    else:
        # sem coluna de data: usa indice da linha como eixo X
        df_work['_eixo_x'] = range(len(df_work))
        tipo_x = "indice"
        avisos.append("Sem coluna de Data informada; usando indice da linha como eixo X.")

    df_work[coluna_y] = pd.to_numeric(df_work[coluna_y], errors='coerce')
    df_work = df_work.dropna(subset=['_eixo_x', coluna_y])

    if len(df_work) < 1:
        return {"erro": "Sem dados validos para tendencia."}

    df_work = df_work.sort_values(by='_eixo_x')

    sg, niveis, av_sg = _normalizar_subgrupo(df, coluna_y, subgrupo)
    avisos.extend(av_sg)

    def _x_para_lista(serie_x):
        if tipo_x == "data":
            return [pd.Timestamp(d).isoformat() for d in serie_x]
        return [str(v) for v in serie_x.tolist()]

    series = []
    if sg is None:
        x_vals = _x_para_lista(df_work['_eixo_x'])
        y_vals = df_work[coluna_y].tolist()
        media = float(df_work[coluna_y].mean()) if len(y_vals) > 0 else 0.0
        series.append({
            "nome": str(coluna_y),
            "x": x_vals,
            "valores": y_vals,
            "media": media,
        })
    else:
        niveis_explicitos = [n for n in niveis if n != "__OUTROS__"]
        for nivel in niveis:
            if nivel == "__OUTROS__":
                mask = ~df_work[sg].astype(str).isin(niveis_explicitos)
            else:
                mask = df_work[sg].astype(str) == str(nivel)
            sub = df_work[mask].sort_values(by='_eixo_x')
            if len(sub) == 0:
                continue
            rotulo = "Outros" if nivel == "__OUTROS__" else str(nivel)
            x_vals = _x_para_lista(sub['_eixo_x'])
            y_vals = sub[coluna_y].tolist()
            media = float(sub[coluna_y].mean()) if len(y_vals) > 0 else 0.0
            series.append({
                "nome": rotulo,
                "x": x_vals,
                "valores": y_vals,
                "media": media,
            })
        if not series:
            return {"erro": "Sem dados validos por subgrupo."}

    estat_global = _estatisticas_basicas(df_work[coluna_y])
    rotulo_x = str(Data) if usa_data else "Sequencia"
    titulo = f"Tendencia de {coluna_y}"
    if usa_data:
        titulo += f" ao longo de {Data}"
    if sg:
        titulo += f" por {sg}"

    return _payload(
        tipo="tendencia",
        series=series,
        labels={"x": rotulo_x, "y": str(coluna_y), "titulo": titulo},
        estat_global=estat_global,
        estat_grupo=None,
        config={"tipo_x": tipo_x},
        avisos=avisos,
    )


# ============================================================
# BOLHAS - 3D (bubble chart: X, Y posicao, Z = tamanho da bolha)
# ============================================================

def bolhas_interativo(df, coluna_y, coluna_x, coluna_z=None):
    erro = _validar_coluna(df, coluna_y, "Variavel Y")
    if erro:
        return {"erro": erro}
    erro = _validar_coluna(df, coluna_x, "Variavel X")
    if erro:
        return {"erro": erro}

    avisos = []

    cols = [coluna_x, coluna_y]
    tem_z = bool(coluna_z and coluna_z in df.columns)
    if tem_z:
        cols.append(coluna_z)

    df_work = df[cols].copy()
    df_work[coluna_x] = pd.to_numeric(df_work[coluna_x], errors='coerce')
    df_work[coluna_y] = pd.to_numeric(df_work[coluna_y], errors='coerce')

    if tem_z:
        df_work[coluna_z] = pd.to_numeric(df_work[coluna_z], errors='coerce')
        df_work = df_work.dropna(subset=[coluna_x, coluna_y, coluna_z])
    else:
        df_work = df_work.dropna(subset=[coluna_x, coluna_y])
        avisos.append("Coluna Z nao informada. Usando tamanho fixo para as bolhas.")

    if len(df_work) < 1:
        return {"erro": "Sem dados validos para o grafico de Bolhas."}

    x_vals = df_work[coluna_x].tolist()
    y_vals = df_work[coluna_y].tolist()

    if tem_z:
        z_vals = df_work[coluna_z].tolist()
        z_min = float(df_work[coluna_z].min())
        z_max = float(df_work[coluna_z].max())
        # normaliza Z para range de tamanho [10, 55]
        z_range = z_max - z_min if z_max != z_min else 1.0
        tamanhos = [10 + 45 * ((v - z_min) / z_range) for v in z_vals]
    else:
        z_vals = []
        tamanhos = [20] * len(x_vals)

    titulo = f"Bolhas: {coluna_y} vs {coluna_x}"
    if tem_z:
        titulo += f" (tamanho = {coluna_z})"

    return _payload(
        tipo="bolhas",
        series=[{
            "nome": f"{coluna_y} vs {coluna_x}",
            "x": x_vals,
            "y": y_vals,
            "z": z_vals,
            "tamanhos": tamanhos,
        }],
        labels={
            "x": str(coluna_x),
            "y": str(coluna_y),
            "z": str(coluna_z) if tem_z else "",
            "titulo": titulo,
        },
        estat_global=_estatisticas_basicas(df_work[coluna_y]),
        estat_grupo=None,
        config={"tem_z": tem_z, "coluna_z": str(coluna_z) if tem_z else ""},
        avisos=avisos,
    )


# ============================================================
# DISPERSAO 3D (scatter3d com 3 eixos cartesianos)
# ============================================================

def dispersao3d_interativo(df, coluna_y, coluna_x, coluna_z=None):
    erro = _validar_coluna(df, coluna_y, "Variavel Y")
    if erro:
        return {"erro": erro}
    erro = _validar_coluna(df, coluna_x, "Variavel X")
    if erro:
        return {"erro": erro}
    erro = _validar_coluna(df, coluna_z, "Variavel Z")
    if erro:
        return {"erro": erro}

    df_work = df[[coluna_x, coluna_y, coluna_z]].copy()
    df_work[coluna_x] = pd.to_numeric(df_work[coluna_x], errors='coerce')
    df_work[coluna_y] = pd.to_numeric(df_work[coluna_y], errors='coerce')
    df_work[coluna_z] = pd.to_numeric(df_work[coluna_z], errors='coerce')
    df_work = df_work.dropna(subset=[coluna_x, coluna_y, coluna_z])

    if len(df_work) < 1:
        return {"erro": "Sem dados validos para Dispersao 3D."}

    titulo = f"Dispersao 3D: {coluna_y} vs {coluna_x} vs {coluna_z}"

    return _payload(
        tipo="dispersao3d",
        series=[{
            "nome": f"{coluna_y} vs {coluna_x} vs {coluna_z}",
            "x": df_work[coluna_x].tolist(),
            "y": df_work[coluna_y].tolist(),
            "z": df_work[coluna_z].tolist(),
        }],
        labels={
            "x": str(coluna_x),
            "y": str(coluna_y),
            "z": str(coluna_z),
            "titulo": titulo,
        },
        estat_global=_estatisticas_basicas(df_work[coluna_y]),
        estat_grupo=None,
        config={},
        avisos=[],
    )


# ============================================================
# SUPERFICIE 3D (interpola X,Y,Z em uma malha tridimensional)
# ============================================================

def superficie3d_interativo(df, coluna_y, coluna_x, coluna_z=None):
    erro = _validar_coluna(df, coluna_y, "Variavel Y")
    if erro:
        return {"erro": erro}
    erro = _validar_coluna(df, coluna_x, "Variavel X")
    if erro:
        return {"erro": erro}
    erro = _validar_coluna(df, coluna_z, "Variavel Z")
    if erro:
        return {"erro": erro}

    avisos = []
    df_work = df[[coluna_x, coluna_y, coluna_z]].copy()
    df_work[coluna_x] = pd.to_numeric(df_work[coluna_x], errors='coerce')
    df_work[coluna_y] = pd.to_numeric(df_work[coluna_y], errors='coerce')
    df_work[coluna_z] = pd.to_numeric(df_work[coluna_z], errors='coerce')
    df_work = df_work.dropna(subset=[coluna_x, coluna_y, coluna_z])

    if len(df_work) < 4:
        return {"erro": "Sem dados suficientes para Superficie 3D (minimo 4 pontos)."}

    # construir malha regular interpolando Z em funcao de X e Y
    try:
        from scipy.interpolate import griddata
        import numpy as np
    except ImportError:
        return {"erro": "Modulo scipy nao disponivel no servidor para interpolacao."}

    x = df_work[coluna_x].values
    y = df_work[coluna_y].values
    z = df_work[coluna_z].values

    n_grid = 40
    xi = np.linspace(x.min(), x.max(), n_grid)
    yi = np.linspace(y.min(), y.max(), n_grid)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # tenta cubic; se faltar densidade, cai para linear; depois nearest
    zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
    if zi is None or np.isnan(zi).all():
        zi = griddata((x, y), z, (xi_grid, yi_grid), method='linear')
        avisos.append("Interpolacao cubica falhou; usando linear.")
    if zi is None or np.isnan(zi).all():
        zi = griddata((x, y), z, (xi_grid, yi_grid), method='nearest')
        avisos.append("Interpolacao linear falhou; usando vizinho mais proximo.")

    # converte NaN remanescentes em None para JSON
    zi_list = []
    for row in zi:
        zi_list.append([None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in row])

    titulo = f"Superficie 3D: {coluna_z} = f({coluna_x}, {coluna_y})"

    return _payload(
        tipo="superficie3d",
        series=[{
            "nome": f"{coluna_z} = f({coluna_x}, {coluna_y})",
            "x": xi.tolist(),
            "y": yi.tolist(),
            "z": zi_list,
            # pontos originais sobrepostos a superficie
            "pontos_x": x.tolist(),
            "pontos_y": y.tolist(),
            "pontos_z": z.tolist(),
        }],
        labels={
            "x": str(coluna_x),
            "y": str(coluna_y),
            "z": str(coluna_z),
            "titulo": titulo,
        },
        estat_global=_estatisticas_basicas(df_work[coluna_z]),
        estat_grupo=None,
        config={"n_pontos": int(len(df_work))},
        avisos=avisos,
    )


# ============================================================
# INTERVALO (intervalos de confianca da media estilo Minitab)
# ============================================================

def intervalo_interativo(df, lista_y, subgrupo=None, field_conf=None):
    if not lista_y:
        return {"erro": "Informe ao menos uma variavel Y."}

    # nivel de confianca (default 95)
    try:
        nivel = float(field_conf) if field_conf else 95.0
        if nivel <= 0 or nivel >= 100:
            nivel = 95.0
    except (ValueError, TypeError):
        nivel = 95.0
    alpha = 1.0 - nivel / 100.0

    avisos = []

    # valida cada Y
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

    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return {"erro": "Modulo scipy nao disponivel para calcular intervalos de confianca."}

    def _ic(valores):
        v = pd.Series(valores).dropna()
        n = len(v)
        if n < 2:
            return None
        media = float(v.mean())
        dp = float(v.std(ddof=1))
        se = dp / (n ** 0.5) if n > 0 else 0.0
        t_crit = float(scipy_stats.t.ppf(1 - alpha / 2, n - 1)) if n > 1 else 0.0
        margem = t_crit * se
        return {
            "n": int(n),
            "media": media,
            "dp": dp,
            "se": float(se),
            "margem": float(margem),
            "ic_inferior": float(media - margem),
            "ic_superior": float(media + margem),
        }

    categorias = []
    medias, ic_inf, ic_sup = [], [], []
    ns, dps, ses, margens = [], [], [], []

    if sg is None:
        for nome, dados in colunas_validas:
            ic = _ic(dados)
            if not ic:
                avisos.append(f"'{nome}' tem n<2 e foi ignorado.")
                continue
            categorias.append(nome)
            medias.append(ic["media"]);     ic_inf.append(ic["ic_inferior"]); ic_sup.append(ic["ic_superior"])
            ns.append(ic["n"]);              dps.append(ic["dp"]);              ses.append(ic["se"])
            margens.append(ic["margem"])
    else:
        niveis_explicitos = [n for n in niveis if n != "__OUTROS__"]
        for nome, _ in colunas_validas:
            for nivel_sg in niveis:
                valores = _serie_filtrada_por_grupo(df, nome, sg, nivel_sg, niveis_explicitos)
                if len(valores) < 2:
                    continue
                ic = _ic(valores)
                if not ic:
                    continue
                rotulo = "Outros" if nivel_sg == "__OUTROS__" else str(nivel_sg)
                cat = f"{nome} | {rotulo}" if len(colunas_validas) > 1 else rotulo
                categorias.append(cat)
                medias.append(ic["media"]);     ic_inf.append(ic["ic_inferior"]); ic_sup.append(ic["ic_superior"])
                ns.append(ic["n"]);              dps.append(ic["dp"]);              ses.append(ic["se"])
                margens.append(ic["margem"])

    if not categorias:
        return {"erro": "Sem dados validos para calcular intervalos (cada grupo precisa de n>=2)."}

    titulo = f"Intervalos de Confianca {nivel:.0f}% - " + ", ".join([c for c, _ in colunas_validas])
    if sg:
        titulo += f" por {sg}"

    return _payload(
        tipo="intervalo",
        series=[{
            "nome": f"IC {nivel:.0f}%",
            "categorias": categorias,
            "medias": medias,
            "ic_inferior": ic_inf,
            "ic_superior": ic_sup,
            "ns": ns,
            "dps": dps,
            "ses": ses,
            "margens": margens,
        }],
        labels={
            "x": str(sg) if sg else "Variavel",
            "y": colunas_validas[0][0],
            "titulo": titulo,
        },
        estat_global=None,
        estat_grupo=None,
        config={
            "nivel_confianca": nivel,
            "n_grupos": len(categorias),
        },
        avisos=avisos,
    )


# Atualize os dois dicionários assim:

GRAFICOS_INTERATIVOS = {
    "Histograma":      histograma_interativo,
    "Pareto":          pareto_interativo,
    "Setores (Pizza)": pizza_interativo,
    "Barras":          barras_interativo,
    "BoxPlot":         boxplot_interativo,
    "Dispersão":       dispersao_interativo,
    "Tendência":       tendencia_interativo,
    "Bolhas - 3D":     bolhas_interativo,
    "Superfície - 3D": superficie3d_interativo,
    "Dispersão 3D":    dispersao3d_interativo,
    "Intervalo":       intervalo_interativo,
}

CONFIG_GRAFICOS_INTERATIVOS = {
    "Histograma":      ["df", "coluna_y", "subgrupo"],
    "Pareto":          ["df", "coluna_x", "coluna_y", "subgrupo"],
    "Setores (Pizza)": ["df", "coluna_x", "coluna_y", "subgrupo"],
    "Barras":          ["df", "coluna_x", "coluna_y", "subgrupo"],
    "BoxPlot":         ["df", "lista_y", "subgrupo"],
    "Dispersão":       ["df", "coluna_y", "coluna_x", "subgrupo"],
    "Tendência":       ["df", "coluna_y", "Data", "subgrupo"],
    "Bolhas - 3D":     ["df", "coluna_y", "coluna_x", "coluna_z"],
    "Superfície - 3D": ["df", "coluna_y", "coluna_x", "coluna_z"],
    "Dispersão 3D":    ["df", "coluna_y", "coluna_x", "coluna_z"],
    "Intervalo":       ["df", "lista_y", "subgrupo", "field_conf"],
}
