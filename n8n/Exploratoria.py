from suporte import *

def grafico_sumario(df, coluna_y):
    if not coluna_y or coluna_y not in df.columns:
        return (
            "❌ A coluna selecionada para o Gráfico Sumario não foi encontrada.",
            None
        )

    serie = df[coluna_y].dropna()
    if serie.empty:
        return (
            "❌ A coluna selecionada não contém dados numéricos válidos.",
            None
        )

    media = serie.mean()
    mediana = serie.median()
    desvio = serie.std()
    variancia = serie.var()
    minimo = serie.min()
    maximo = serie.max()
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    assimetria = serie.skew()
    curtose = serie.kurtosis()
    n = serie.count()

    resumo = f"""📊 **Gráfico Sumario da coluna '{coluna_y}'**  
- Média: {media:.2f}  
- Mediana: {mediana:.2f}  
- Desvio Padrão: {desvio:.2f}  
- Variância: {variancia:.2f}  
- Mínimo: {minimo:.2f}  
- 1º Quartil (Q1): {q1:.2f}  
- 3º Quartil (Q3): {q3:.2f}  
- Máximo: {maximo:.2f}  
- Assimetria: {assimetria:.2f}  
- Curtose: {curtose:.2f}  
- N: {n}"""

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(serie, bins=10, kde=True, ax=ax, color='gray')
    ax.set_title(f"Gráfico Sumario - {coluna_y}")
    ax.set_xlabel(coluna_y)
    ax.set_ylabel("Frequência")

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return resumo, imagem_base64


def analise_de_outliers(df, lista_y):
    resultado_texto = "📊 **Análise de Outliers**\n"
    aplicar_estilo_minitab()

    dados_plot = {}
    encontrou_outliers = False

    for coluna in lista_y:
        if coluna not in df.columns:
            resultado_texto += f"- ❌ A coluna '{coluna}' não foi encontrada.\n"
            continue

        serie = df[coluna].dropna()
        if serie.empty:
            resultado_texto += f"- ❌ A coluna '{coluna}' não contém dados numéricos válidos.\n"
            continue

        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        outliers = serie[(serie < limite_inferior) | (serie > limite_superior)]

        dados_plot[coluna] = serie

        if not outliers.empty:
            encontrou_outliers = True
            resultado_texto += f"- ⚠ A coluna '{coluna}' possui {len(outliers)} outlier(s): {list(outliers.values)}\n"
        else:
            resultado_texto += f"- ✅ A coluna '{coluna}' não possui outliers detectados.\n"

    if not dados_plot:
        return resultado_texto, None

    df_plot = pd.DataFrame(dados_plot)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_plot, ax=ax, orient="v", flierprops=dict(marker='*', markersize=8, markerfacecolor='red'))
    ax.set_title("Boxplot - Outliers nas colunas analisadas")
    ax.set_ylabel("Valores")
    ax.set_xlabel("Colunas")

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return resultado_texto, imagem_base64

def analise_correlacao_person(df, coluna_y, lista_x):
    nomes_df = df.columns.tolist()

    if not coluna_y:
        return "❌ É necessário informar uma coluna Y.", None
    if not lista_x or len(lista_x) < 1:
        return "❌ É necessário ao menos uma coluna X.", None

    if coluna_y not in nomes_df:
        return f"❌ A coluna Y '{coluna_y}' não foi encontrada no arquivo.", None

    # Verifica se coluna Y é numérica
    if not pd.api.types.is_numeric_dtype(df[coluna_y]):
        return f"❌ A coluna Y '{coluna_y}' contém dados não numéricos e não pode ser usada na correlação de Pearson.", None

    serie_y = df[coluna_y].dropna()
    if serie_y.empty:
        return f"❌ A coluna Y '{coluna_y}' não contém dados válidos.", None

    linhas = []
    conclusoes = []
    for nome_x in lista_x:
        if nome_x not in nomes_df:
            linhas.append(f"{coluna_y} x {nome_x}\n❌ A coluna X não foi encontrada no arquivo.")
            continue

        # Verifica se coluna X é numérica
        if not pd.api.types.is_numeric_dtype(df[nome_x]):
            linhas.append(f"{coluna_y} x {nome_x}\n❌ A coluna X '{nome_x}' contém dados não numéricos e não pode ser usada na correlação de Pearson.")
            continue

        serie_x = df[nome_x].dropna()
        if serie_x.empty:
            linhas.append(f"{coluna_y} x {nome_x}\n❌ Dados X inválidos.")
            continue

        data = pd.concat([serie_y, serie_x], axis=1).dropna()
        if data.empty or len(data) < 2:
            linhas.append(f"{coluna_y} x {nome_x}\n❌ Sem dados pareados suficientes.")
            continue

        r, p = stats.pearsonr(data.iloc[:, 0], data.iloc[:, 1])
        forca = "fraca" if abs(r) < 0.3 else "média" if abs(r) < 0.7 else "forte"
        sentido = "positiva" if r >= 0 else "negativa"
        significancia = "são estatisticamente correlacionadas" if p < 0.05 else "não são estatisticamente correlacionadas"

        if p < 0.05:
            conclusoes.append(nome_x)

        linhas.append(
            f"{coluna_y} x {nome_x}\n"
            f"Coeficiente de Pearson = {r:.2f} → Correlação {forca}, {sentido}.\n"
            f"P-value: {p:.4f} → As variáveis {coluna_y} e {nome_x} {significancia}."
        )

    if conclusoes:
        conclusao_final = f"🔎 **Conclusão**: Apenas as variáveis {', '.join(conclusoes)} são estatisticamente correlacionadas com {coluna_y}."
    else:
        conclusao_final = f"🔎 **Conclusão**: Nenhuma das variáveis apresenta correlação estatisticamente significativa com {coluna_y}."

    # 🔷 BLOCO DE RELATÓRIO PARA EDIÇÃO FUTURA
    resumo = (
        f"📊 **Análise de Correlação de Pearson**\n"
        f"Variáveis analisadas:\n\n"
        + "\n\n".join(linhas)
        + "\n\n"
        + conclusao_final
    )

    return resumo, None


def analise_matrix_correlacao(df, coluna_y, lista_x):
    colunas = ([coluna_y] if coluna_y else []) + (lista_x or [])

    if len(colunas) < 2:
        return "❌ É necessário ao menos duas colunas para gerar a matriz de correlação.", None

    for col in colunas:
        if col not in df.columns:
            return f"❌ A coluna '{col}' não foi encontrada no arquivo.", None
        # Verifica se coluna é numérica
        if not pd.api.types.is_numeric_dtype(df[col]):
            return f"❌ A coluna '{col}' contém dados não numéricos e não pode ser usada na matriz de correlação.", None

    dados = df[colunas].dropna()
    if dados.empty:
        return "❌ Dados insuficientes após remoção de valores ausentes.", None

    matriz_cor = dados.corr(method='pearson')

    linhas_resumo = []
    relevantes = []
    nao_relevantes = []

    for i in range(len(colunas)):
        for j in range(i + 1, len(colunas)):
            col1 = colunas[i]
            col2 = colunas[j]
            r = matriz_cor.loc[col1, col2]

            if abs(r) < 0.3:
                forca = "fraca"
                nao_relevantes.append(f"{col1} x {col2}")
            elif abs(r) < 0.7:
                forca = "moderada"
                relevantes.append(f"{col1} x {col2}")
            else:
                forca = "forte"
                relevantes.append(f"{col1} x {col2}")

            sentido = "positiva" if r >= 0 else "negativa"

            linhas_resumo.append(
                f"{col1} x {col2}\n"
                f"Coeficiente de Pearson = {r:.2f} → Correlação {forca}, {sentido}."
            )

    # 🔷 BLOCO DE RELATÓRIO PARA EDIÇÃO FUTURA
    conclusao = "🔎 **Conclusão:**\n"
    if relevantes:
        conclusao += f"- Correlações relevantes: {', '.join(relevantes)}\n"
    else:
        conclusao += "- Não há correlações relevantes.\n"
    if nao_relevantes:
        conclusao += f"- Correlações não relevantes: {', '.join(nao_relevantes)}"

    resumo = (
        f"📊 **Matriz de Correlação de Pearson**\n"
        f"Variáveis analisadas:\n\n"
        + "\n\n".join(linhas_resumo)
        + "\n\n"
        + conclusao
    )

    aplicar_estilo_minitab()
    sns.pairplot(dados, kind='reg', plot_kws={'line_kws': {'color': 'red'}, 'scatter_kws': {'s': 20}})

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return resumo, img_base64



def analise_estabilidade(df, coluna_y):
    import matplotlib.pyplot as plt
    import numpy as np
    from io import BytesIO
    import base64
    import pandas as pd

    aplicar_estilo_minitab()

    nome_coluna_y = coluna_y if isinstance(coluna_y, str) else (coluna_y[0] if coluna_y else None)
    if not nome_coluna_y or nome_coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    if not pd.api.types.is_numeric_dtype(df[nome_coluna_y]):
        return f"❌ A coluna '{nome_coluna_y}' contém dados não numéricos e não pode ser usada na análise de estabilidade.", None

    dados = df[[nome_coluna_y]].dropna().copy()
    dados["Subgrupo"] = range(1, len(dados) + 1)

    if dados.empty:
        return "❌ Dados insuficientes para análise.", None

    # Estatísticas
    media = dados[nome_coluna_y].mean()
    sigma = dados[nome_coluna_y].std()
    UCL_I = media + 3 * sigma
    LCL_I = media - 3 * sigma

    mr = dados[nome_coluna_y].diff().abs()
    mr_mean = mr[1:].mean()
    UCL_MR = mr_mean * 3.267

    # Gráfico estilo Minitab
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Carta Individual (I)
    y = dados[nome_coluna_y].values
    x = dados["Subgrupo"].values
    pontos_i = axs[0].scatter(x, y, color="black")
    axs[0].plot(x, y, color="black", linestyle="-")
    axs[0].axhline(media, color="green", linestyle="-")
    axs[0].axhline(UCL_I, color="red", linestyle="-")
    axs[0].axhline(LCL_I, color="red", linestyle="-")
    axs[0].set_title(f"Carta I de {nome_coluna_y}", fontsize=12)
    axs[0].set_ylabel("Valor Individual")

    # Labels no lado direito
    axs[0].text(len(x)+1, media, f"X̄={media:.3f}", va='center', fontsize=8)
    axs[0].text(len(x)+1, UCL_I, f"LSC={UCL_I:.3f}", va='center', fontsize=8, color="red")
    axs[0].text(len(x)+1, LCL_I, f"LIC={LCL_I:.3f}", va='center', fontsize=8, color="red")

    # Destacar pontos fora dos limites
    for xi, yi in zip(x, y):
        if yi > UCL_I or yi < LCL_I:
            axs[0].scatter(xi, yi, color="red")

    # Carta MR
    x_mr = dados["Subgrupo"].values[1:]
    y_mr = mr[1:].values
    axs[1].plot(x_mr, y_mr, color="black", linestyle="-")
    axs[1].scatter(x_mr, y_mr, color="black")
    axs[1].axhline(mr_mean, color="green", linestyle="-")
    axs[1].axhline(UCL_MR, color="red", linestyle="-")
    axs[1].set_title("Carta MR", fontsize=12)
    axs[1].set_ylabel("Amplitude Móvel")

    # Labels no lado direito
    axs[1].text(len(x)+1, mr_mean, f"MR̄={mr_mean:.3f}", va='center', fontsize=8)
    axs[1].text(len(x)+1, UCL_MR, f"LSC={UCL_MR:.3f}", va='center', fontsize=8, color="red")
    axs[1].text(len(x)+1, 0, f"LIC=0.000", va='center', fontsize=8, color="red")

    # Destacar pontos fora dos limites MR
    for xi, yi in zip(x_mr, y_mr):
        if yi > UCL_MR:
            axs[1].scatter(xi, yi, color="red")

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    # 🔷 BLOCO DE RELATÓRIO PARA EDIÇÃO FUTURA
    texto_resumo = f"📊 **Carta I-MR ({nome_coluna_y}) gerada com sucesso.**"

    return texto_resumo, img_base64




def analise_limpeza_dados(df):
    import numpy as np
    import pandas as pd
    import string

    linhas_total = len(df)
    colunas_total = df.shape[1]
    linhas_duplicadas = df.duplicated().sum()

    # Mapeia colunas para letras A, B, C...
    letras_colunas = {col: string.ascii_uppercase[i] for i, col in enumerate(df.columns)}

    resultado = [
        "📊 <strong>Análise de Limpeza de Dados</strong><br>",
        f"<strong>Total de linhas esperadas:</strong> {linhas_total}",
        f"<strong>Total de colunas:</strong> {colunas_total}",
        f"<strong>Linhas duplicadas detectadas:</strong> {linhas_duplicadas}<br>"
    ]

    colunas_aptas = []
    colunas_problemas = []

    for coluna in df.columns:
        letra = letras_colunas[coluna]
        n_valores = df[coluna].notnull().sum()
        n_gaps = linhas_total - n_valores
        problemas = []

        # Linhas faltando
        if n_gaps > 0:
            idx_gaps = df[df[coluna].isnull()].index
            primeira_linha_gap = idx_gaps[0] + 2 if len(idx_gaps) > 0 else "?"
            problemas.append(f"{n_gaps} célula(s) vazia(s) (primeira ocorrência na linha {primeira_linha_gap})")

        # Coluna totalmente vazia
        if n_valores == 0:
            problemas.append("⚠ Totalmente vazia")

        # Coluna sem variação
        if df[coluna].nunique(dropna=True) <= 1:
            problemas.append("⚠ Sem variação (todos os valores iguais)")

        # Dados não numéricos em colunas object
        if df[coluna].dtype == object:
            try:
                pd.to_numeric(df[coluna].dropna())
            except:
                problemas.append("⚠ Contém dados não numéricos suspeitos")

        # Outliers extremos
        if pd.api.types.is_numeric_dtype(df[coluna]):
            media = df[coluna].mean()
            sigma = df[coluna].std()
            if sigma > 0:
                extremos = df[(df[coluna] > media + 10 * sigma) | (df[coluna] < media - 10 * sigma)]
                if not extremos.empty:
                    problemas.append(f"⚠ {len(extremos)} valor(es) extremamente fora do padrão")

        if problemas:
            resultado.append(f"Coluna {letra}: " + "; ".join(problemas))
            colunas_problemas.append(letra)
        else:
            colunas_aptas.append(letra)

    # Colunas duplicadas
    colunas_duplicadas = []
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            if df[df.columns[i]].equals(df[df.columns[j]]):
                colunas_duplicadas.append(f"{letras_colunas[df.columns[i]]} / {letras_colunas[df.columns[j]]}")
    if colunas_duplicadas:
        resultado.append(f"⚠ Colunas duplicadas detectadas: {colunas_duplicadas}")
        colunas_problemas.extend([c.split(' / ')[0] for c in colunas_duplicadas])
        colunas_problemas.extend([c.split(' / ')[1] for c in colunas_duplicadas])
    else:
        resultado.append("✅ Nenhuma coluna duplicada")

    # 🔷 BLOCO DE RELATÓRIO PARA EDIÇÃO FUTURA
    conclusao = "🔎 <strong>Conclusão:</strong><br>"
    conclusao += f"✅ <strong>Colunas aptas para análises:</strong> {', '.join(colunas_aptas) if colunas_aptas else 'Nenhuma'}<br>"
    conclusao += f"⚠ <strong>Colunas para serem analisadas/melhoradas:</strong> {', '.join(set(colunas_problemas)) if colunas_problemas else 'Nenhuma'}"

    resultado.append("<br>" + conclusao)

    texto_final = "<br>".join(resultado)
    return texto_final, None




ANALISES = {
    "Gráfico Sumario": grafico_sumario,
    "Análise de outliers": analise_de_outliers,
    "Correlação de person": analise_correlacao_person,
    "Matrix de dispersão": analise_matrix_correlacao,
    "Análise de estabilidade": analise_estabilidade,
    "Análise de limpeza dos dados": analise_limpeza_dados
}


