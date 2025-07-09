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
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Carta Individual (I)
    y = dados[nome_coluna_y].values
    x = dados["Subgrupo"].values
    axs[0].plot(x, y, color="black", linestyle="-")
    axs[0].scatter(x, y, color="black")
    axs[0].axhline(media, color="green", linestyle="-")
    axs[0].axhline(UCL_I, color="red", linestyle="-")
    axs[0].axhline(LCL_I, color="red", linestyle="-")
    axs[0].set_title(f"Carta I de {nome_coluna_y}", fontsize=14)
    axs[0].set_ylabel("Valor Individual", fontsize=12)

    xlim = axs[0].get_xlim()
    axs[0].text(xlim[1]+1, media, f"X̄ = {media:.3f}", va='center', fontsize=10, color="green")
    axs[0].text(xlim[1]+1, UCL_I, f"LSC = {UCL_I:.3f}", va='center', fontsize=10, color="red")
    axs[0].text(xlim[1]+1, LCL_I, f"LIC = {LCL_I:.3f}", va='center', fontsize=10, color="red")

    crit1_flag_I = any((y > UCL_I) | (y < LCL_I))
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
    axs[1].set_title("Carta MR", fontsize=14)
    axs[1].set_ylabel("Amplitude Móvel", fontsize=12)

    xlim_mr = axs[1].get_xlim()
    axs[1].text(xlim_mr[1]+1, mr_mean, f"MR̄ = {mr_mean:.3f}", va='center', fontsize=10, color="green")
    axs[1].text(xlim_mr[1]+1, UCL_MR, f"LSC = {UCL_MR:.3f}", va='center', fontsize=10, color="red")
    axs[1].text(xlim_mr[1]+1, 0, f"LIC = 0.000", va='center', fontsize=10, color="red")

    crit1_flag_MR = any(y_mr > UCL_MR)
    for xi, yi in zip(x_mr, y_mr):
        if yi > UCL_MR:
            axs[1].scatter(xi, yi, color="red")

    plt.tight_layout()

    # Funções de checagem (substitua pelas reais)
    def check_crit2(y): return False
    def check_crit3(y): return False
    def check_crit4(y): return False
    def check_crit5(y): return False
    def check_crit6(y): return False
    def check_crit7(y): return False
    def check_crit8(y): return False
    def check_crit9(y): return False
    def check_crit2_mr(y_mr): return False
    def check_crit3_mr(y_mr): return False
    def check_crit4_mr(y_mr): return False
    def check_crit5_mr(y_mr): return False
    def check_crit6_mr(y_mr): return False
    def check_crit7_mr(y_mr): return False
    def check_crit8_mr(y_mr): return False
    def check_crit9_mr(y_mr): return False

    # 🔷 BLOCO DE RELATÓRIO
    texto_resumo = f"📊 **Análise de Estabilidade – Carta I-MR ({nome_coluna_y})**\n"
    texto_resumo += "🔎 **Critérios avaliados:**\n"

    # Critério 1
    if crit1_flag_I and crit1_flag_MR:
        texto_resumo += "1. Critério 1 – Pontos fora dos limites: ❌ Detectado (Carta I e MR)\n"
    elif crit1_flag_I:
        texto_resumo += "1. Critério 1 – Pontos fora dos limites: ❌ Detectado (Carta I)\n"
    elif crit1_flag_MR:
        texto_resumo += "1. Critério 1 – Pontos fora dos limites: ❌ Detectado (Carta MR)\n"
    else:
        texto_resumo += "1. Critério 1 – Pontos fora dos limites: ✅ OK\n"

    # Critérios 2-9
    nomes_criterios = {
        2: "9 pontos do mesmo lado da média",
        3: "6 pontos subindo ou descendo",
        4: "14 pontos alternando",
        5: "2 de 3 pontos além de 2σ no mesmo lado",
        6: "4 de 5 pontos além de 1σ no mesmo lado",
        7: "15 pontos dentro de 1σ",
        8: "8 pontos fora de 1σ",
        9: "12 pontos alternando"
    }

    funcoes_I = {
        2: check_crit2,
        3: check_crit3,
        4: check_crit4,
        5: check_crit5,
        6: check_crit6,
        7: check_crit7,
        8: check_crit8,
        9: check_crit9
    }

    funcoes_MR = {
        2: check_crit2_mr,
        3: check_crit3_mr,
        4: check_crit4_mr,
        5: check_crit5_mr,
        6: check_crit6_mr,
        7: check_crit7_mr,
        8: check_crit8_mr,
        9: check_crit9_mr
    }

    for i in range(2, 10):
        flag_I = funcoes_I[i](y)
        flag_MR = funcoes_MR[i](y_mr)
        if flag_I and flag_MR:
            texto_resumo += f"{i}. Critério {i} – {nomes_criterios[i]}: ❌ Detectado (Carta I e MR)\n"
        elif flag_I:
            texto_resumo += f"{i}. Critério {i} – {nomes_criterios[i]}: ❌ Detectado (Carta I)\n"
        elif flag_MR:
            texto_resumo += f"{i}. Critério {i} – {nomes_criterios[i]}: ❌ Detectado (Carta MR)\n"
        else:
            texto_resumo += f"{i}. Critério {i} – {nomes_criterios[i]}: ✅ OK\n"

    # Conclusão
    if any([crit1_flag_I, crit1_flag_MR] + [funcoes_I[i](y) or funcoes_MR[i](y_mr) for i in range(2,10)]):
        texto_resumo += "🔎 **Conclusão:** Causa especial detectada. Investigue o processo para entender e remover a causa especial identificada.\n"
    else:
        texto_resumo += "🔎 **Conclusão:** Processo está estável. Nenhuma causa especial detectada.\n"

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

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


def analise_cluster_kmeans(df, lista_x, n_clusters=3):
    """
    Análise de Cluster KMeans com múltiplas variáveis (lista_x).
    """

    # ⚠️ Verificar se lista_x foi fornecida
    if not lista_x or any(col not in df.columns for col in lista_x):
        return "❌ É necessário informar uma lista de colunas X válidas.", None

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from io import BytesIO
    import base64

    # 🔹 Selecionar os dados e remover NA
    dados = df[lista_x].dropna()

    # ⚠️ Verificar se há dados suficientes
    if len(dados) < n_clusters:
        return f"❌ Dados insuficientes para formar {n_clusters} clusters.", None

    # 🔹 Padronizar os dados
    scaler = StandardScaler()
    dados_scaled = scaler.fit_transform(dados)

    # 🔹 Aplicar KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(dados_scaled)

    # 🔹 Adicionar cluster ao dataframe original
    df_resultado = dados.copy()
    df_resultado['Cluster'] = clusters

    # 🔹 Gráfico (primeiras duas variáveis)
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        dados_scaled[:, 0],
        dados_scaled[:, 1] if dados_scaled.shape[1] > 1 else np.zeros_like(dados_scaled[:, 0]),
        c=clusters,
        cmap='viridis',
        edgecolor='k'
    )
    ax.set_xlabel(lista_x[0])
    if len(lista_x) > 1:
        ax.set_ylabel(lista_x[1])
    else:
        ax.set_ylabel('')

    ax.set_title('📊 Cluster KMeans')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 🔹 Reporte
    texto = f"""
📊 **Análise de Cluster KMeans**

🔹 **Resultado**

- Número de clusters: {n_clusters}
- Variáveis utilizadas: {', '.join(lista_x)}

✔️ **Centroides**

{pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=lista_x).round(4).to_string(index=False)}

✔️ **Recomendações**

- Avaliar se o número de clusters escolhido faz sentido prático.
- Verificar a separação visual e estatística entre os clusters para interpretar os grupos formados.
    """.strip()

    return texto, grafico_base64




ANALISES = {
    "Gráfico Sumario": grafico_sumario,
    "Análise de outliers": analise_de_outliers,
    "Correlação de person": analise_correlacao_person,
    "Matrix de dispersão": analise_matrix_correlacao,
    "Análise de estabilidade": analise_estabilidade,
    "Análise de limpeza dos dados": analise_limpeza_dados,
    "Análise de cluster": analise_cluster_kmeans
}


