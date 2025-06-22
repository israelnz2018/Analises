from suporte import *

def grafico_sumario(df, colunas_usadas):
    coluna_y = colunas_usadas[0]
    if coluna_y not in df.columns:
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

def analise_de_outliers(df, colunas_usadas):
    resultado_texto = "📊 **Análise de Outliers**\n"
    aplicar_estilo_minitab()
    fig, axs = plt.subplots(len(colunas_usadas), 1, figsize=(6, 4 * len(colunas_usadas)))
    if len(colunas_usadas) == 1:
        axs = [axs]

    encontrou_outliers = False

    for ax, coluna in zip(axs, colunas_usadas):
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

        sns.boxplot(x=serie, orient="h", ax=ax, flierprops=dict(marker='*', markersize=8, markerfacecolor='red'))
        ax.set_title(f"Boxplot - {coluna}")
        ax.set_xlabel(coluna)

        if not outliers.empty:
            encontrou_outliers = True
            resultado_texto += f"- ⚠ A coluna '{coluna}' possui {len(outliers)} outlier(s): {list(outliers.values)}\n"
        else:
            resultado_texto += f"- ✅ A coluna '{coluna}' não possui outliers detectados.\n"

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return resultado_texto, imagem_base64

def analise_correlacao_person(df, colunas_y, colunas_x, field=None):
    if not colunas_y or len(colunas_y) != 1:
        return "❌ É necessário exatamente uma variável Y.", None
    if not colunas_x or len(colunas_x) < 1:
        return "❌ É necessário ao menos uma variável X.", None

    nome_coluna_y = colunas_y[0]
    nomes_colunas_x = colunas_x

    if nome_coluna_y not in df.columns:
        return f"❌ A coluna Y '{nome_coluna_y}' não foi encontrada.", None

    for col in nomes_colunas_x:
        if col not in df.columns:
            return f"❌ A coluna X '{col}' não foi encontrada.", None

    serie_y = df[nome_coluna_y].dropna()
    if serie_y.empty:
        return f"❌ A coluna Y '{nome_coluna_y}' não contém dados válidos.", None

    linhas = []
    for nome_x in nomes_colunas_x:
        serie_x = df[nome_x].dropna()
        if serie_x.empty:
            linhas.append(f"- {nome_x}: ❌ Dados X inválidos.")
            continue

        data = pd.concat([serie_y, serie_x], axis=1).dropna()
        if data.empty or len(data) < 2:
            linhas.append(f"- {nome_x}: ❌ Sem dados pareados suficientes.")
            continue

        r, p = stats.pearsonr(data.iloc[:, 0], data.iloc[:, 1])

        forca = "fraca" if abs(r) < 0.3 else "moderada" if abs(r) < 0.7 else "forte"
        dependencia = "existe dependência estatística" if p < 0.05 else "não há dependência estatística"

        linhas.append(
            f"- {nome_x}: Coeficiente de Pearson = {r:.2f}, p-valor = {p:.4f} → Correlação {forca}, {dependencia}."
        )

    resumo = f"""📊 **Análise de Correlação de Pearson**
Coluna Y: **{nome_coluna_y}**
Resultados:
""" + "\n".join(linhas)

    return resumo, None




def analise_matrix_correlacao(df, colunas_usadas):
    if len(colunas_usadas) < 2:
        return "❌ É necessário ao menos duas colunas para gerar a matriz de correlação.", None

    for col in colunas_usadas:
        if col not in df.columns:
            return f"❌ Coluna '{col}' não encontrada no dataframe.", None

    dados = df[colunas_usadas].dropna()
    if dados.empty:
        return "❌ Dados insuficientes após remoção de valores ausentes.", None

    matriz_cor = dados.corr(method='pearson')

    linhas_resumo = []
    for i in range(len(colunas_usadas)):
        for j in range(i + 1, len(colunas_usadas)):
            col1 = colunas_usadas[i]
            col2 = colunas_usadas[j]
            r = matriz_cor.loc[col1, col2]

            if abs(r) < 0.3:
                forca = "fraca"
            elif abs(r) < 0.7:
                forca = "moderada"
            else:
                forca = "forte"

            linhas_resumo.append(f"- {col1} vs {col2}: correlação {forca} (r={r:.2f})")

    resumo = "📊 **Matriz de Correlação (Pearson)**\n" + "\n".join(linhas_resumo)

    aplicar_estilo_minitab()
    sns.pairplot(dados, kind='reg', plot_kws={'line_kws': {'color': 'red'}, 'scatter_kws': {'s': 20}})

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return resumo, img_base64

def analise_estabilidade(df, colunas_usadas):
    if not colunas_usadas or colunas_usadas[0] not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no dataframe.", None

    nome_coluna_y = colunas_usadas[0]
    nome_coluna_subgrupo = colunas_usadas[1] if len(colunas_usadas) > 1 else None

    dados = df[[nome_coluna_y]].copy()
    if nome_coluna_subgrupo and nome_coluna_subgrupo in df.columns:
        dados['Subgrupo'] = df[nome_coluna_subgrupo]
    else:
        dados['Subgrupo'] = range(1, len(dados) + 1)

    if dados.empty or dados[nome_coluna_y].dropna().empty:
        return "❌ Dados insuficientes para análise.", None

    aplicar_estilo_minitab()
    texto_resumo = f"📊 **Análise de Estabilidade da coluna '{nome_coluna_y}'**\n"

    if nome_coluna_subgrupo and nome_coluna_subgrupo in df.columns:
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        sns.lineplot(x="Subgrupo", y=nome_coluna_y, data=dados, ax=axs[0], marker="o")
        axs[0].set_title("Carta X-Barra")
        axs[0].set_ylabel("Média")

        grupo_stats = dados.groupby('Subgrupo')[nome_coluna_y].agg(['mean', 'std'])
        r_values = grupo_stats['std'] * (2 ** 0.5)

        axs[1].plot(grupo_stats.index, r_values, marker="o")
        axs[1].set_title("Carta R")
        axs[1].set_ylabel("Amplitude (aprox)")

        texto_resumo += "- Carta X-BarraR usada (subgrupos detectados).\n"
    else:
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        sns.lineplot(x="Subgrupo", y=nome_coluna_y, data=dados, ax=axs[0], marker="o")
        axs[0].set_title("Carta Individual")
        axs[0].set_ylabel("Valor")

        mr = dados[nome_coluna_y].diff().abs()
        axs[1].plot(dados['Subgrupo'][1:], mr[1:], marker="o")
        axs[1].set_title("Carta MR")
        axs[1].set_ylabel("Movimento Range")

        texto_resumo += "- Carta I-MR usada (sem subgrupos).\n"

    media = dados[nome_coluna_y].mean()
    sigma = dados[nome_coluna_y].std()
    outliers = dados[(dados[nome_coluna_y] > media + 3 * sigma) | (dados[nome_coluna_y] < media - 3 * sigma)]

    if not outliers.empty:
        texto_resumo += f"- ⚠ Detectados {len(outliers)} pontos fora dos limites (3 sigma). Processo potencialmente instável.\n"
    else:
        texto_resumo += "- ✅ Nenhum ponto fora dos limites (3 sigma). Processo aparentemente estável.\n"

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return texto_resumo, img_base64

def analise_limpeza_dados(df, colunas_usadas):
    linhas_total = len(df)
    colunas_total = df.shape[1]
    linhas_duplicadas = df.duplicated().sum()

    resultado = [
        f"<strong>Total de linhas esperadas:</strong> {linhas_total}",
        f"<strong>Total de colunas:</strong> {colunas_total}",
        f"<strong>Linhas duplicadas detectadas:</strong> {linhas_duplicadas}<br>"
    ]

    for coluna in df.columns:
        n_valores = df[coluna].notnull().sum()
        n_gaps = linhas_total - n_valores

        if n_gaps > 0:
            idx_gaps = df[df[coluna].isnull()].index
            primeira_linha_gap = idx_gaps[0] + 2 if len(idx_gaps) > 0 else "?"
            resultado.append(
                f"Coluna <strong>{coluna}</strong>: {n_gaps} linhas faltando (primeiro gap na linha {primeira_linha_gap})"
            )

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


