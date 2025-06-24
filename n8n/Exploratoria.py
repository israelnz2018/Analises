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

    serie_y = df[coluna_y].dropna()
    if serie_y.empty:
        return f"❌ A coluna Y '{coluna_y}' não contém dados válidos.", None

    linhas = []
    for nome_x in lista_x:
        if nome_x not in nomes_df:
            linhas.append(f"- {nome_x}: ❌ A coluna X não foi encontrada no arquivo.")
            continue

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

        linhas.append(f"- {nome_x}: Coeficiente de Pearson = {r:.2f}, p-valor = {p:.4f} → Correlação {forca}, {dependencia}.")

    resumo = f"""📊 **Análise de Correlação de Pearson**
Coluna Y: **{coluna_y}**
Resultados:
""" + "\n".join(linhas)

    return resumo, None


def analise_matrix_correlacao(df, coluna_y, lista_x):
    colunas = ([coluna_y] if coluna_y else []) + (lista_x or [])
    
    if len(colunas) < 2:
        return "❌ É necessário ao menos duas colunas para gerar a matriz de correlação.", None

    for col in colunas:
        if col not in df.columns:
            return f"❌ A coluna '{col}' não foi encontrada no arquivo.", None

    dados = df[colunas].dropna()
    if dados.empty:
        return "❌ Dados insuficientes após remoção de valores ausentes.", None

    matriz_cor = dados.corr(method='pearson')

    linhas_resumo = []
    for i in range(len(colunas)):
        for j in range(i + 1, len(colunas)):
            col1 = colunas[i]
            col2 = colunas[j]
            r = matriz_cor.loc[col1, col2]

            if abs(r) < 0.3:
                forca = "fraca"
            elif abs(r) < 0.7:
                forca = "moderada"
            else:
                forca = "forte"

            linhas_resumo.append(f"- {col1} vs {col2}: correlação {forca} (r = {r:.2f})")

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


def analise_estabilidade(df, coluna_y):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from io import BytesIO
    import base64

    aplicar_estilo_minitab()

    nome_coluna_y = coluna_y if isinstance(coluna_y, str) else (coluna_y[0] if coluna_y else None)
    if not nome_coluna_y or nome_coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

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

    # Gráficos
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(dados["Subgrupo"], dados[nome_coluna_y], marker="o", linestyle="-")
    axs[0].axhline(media, color="black", linestyle="--", label="Média")
    axs[0].axhline(UCL_I, color="red", linestyle="--", label="Limite Superior")
    axs[0].axhline(LCL_I, color="red", linestyle="--", label="Limite Inferior")
    axs[0].set_title("Carta Individual (I)")
    axs[0].set_ylabel("Valor")
    axs[0].legend()

    axs[1].plot(dados["Subgrupo"][1:], mr[1:], marker="o", linestyle="-")
    axs[1].axhline(mr_mean, color="black", linestyle="--", label="Média MR")
    axs[1].axhline(UCL_MR, color="red", linestyle="--", label="Limite Superior MR")
    axs[1].set_title("Carta Amplitude Móvel (MR)")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()

    texto_resumo = f"📊 **Análise de Estabilidade da coluna '{nome_coluna_y}'**\n"
    texto_resumo += "- Carta I-MR usada (dados individuais).\n"

    y = dados[nome_coluna_y].values

    # Critérios
    def check_crit1():
        return np.any((y > UCL_I) | (y < LCL_I))

    def check_crit2():
        seq = (y > media).astype(int)
        return any(sum(seq[i:i+9]) == 9 or sum(1 - seq[i:i+9]) == 9 for i in range(len(seq)-8))

    def check_crit3():
        diff = np.sign(np.diff(y))
        count = 1
        for i in range(1, len(diff)):
            if diff[i] == diff[i-1] and diff[i] != 0:
                count += 1
                if count >= 6:
                    return True
            else:
                count = 1
        return False

    def check_crit4():
        diff = np.sign(np.diff(y))
        return np.all(diff[:14] != 0) and np.all(diff[:14][::2] * diff[1:15:2] == -1) if len(y) >= 15 else False

    def check_crit5():
        z = (y - media) / sigma
        for i in range(len(z)-2):
            s = z[i:i+3]
            if (np.sum(np.abs(s) > 2) >= 2) and (np.all(s > 0) or np.all(s < 0)):
                return True
        return False

    def check_crit6():
        z = (y - media) / sigma
        for i in range(len(z)-4):
            s = z[i:i+5]
            if (np.sum(np.abs(s) > 1) >= 4) and (np.all(s > 0) or np.all(s < 0)):
                return True
        return False

    def check_crit7():
        z = (y - media) / sigma
        for i in range(len(z)-14):
            if np.all(np.abs(z[i:i+15]) < 1):
                return True
        return False

    def check_crit8():
        z = (y - media) / sigma
        for i in range(len(z)-7):
            if np.all(np.abs(z[i:i+8]) > 1):
                return True
        return False

    def check_crit9():
        diff = np.sign(np.diff(y))
        alt = 1
        for i in range(1, len(diff)):
            if diff[i] == -diff[i-1] and diff[i] != 0:
                alt += 1
                if alt >= 12:
                    return True
            else:
                alt = 1
        return False

    def check_crit10():
        return False  # Minitab original não define critério 10 padrão, deixamos reservado

    # Aplica critérios
    if check_crit1():
        texto_resumo += "- ⚠ Critério 1: Ponto fora dos limites de controle.\n"
    else:
        texto_resumo += "- ✅ Critério 1: Nenhum ponto fora dos limites.\n"

    if check_crit2():
        texto_resumo += "- ⚠ Critério 2: 9 pontos seguidos do mesmo lado da média.\n"
    else:
        texto_resumo += "- ✅ Critério 2: Nenhuma sequência longa de um lado.\n"

    if check_crit3():
        texto_resumo += "- ⚠ Critério 3: 6 pontos seguidos subindo ou descendo.\n"
    else:
        texto_resumo += "- ✅ Critério 3: Nenhuma tendência longa.\n"

    if check_crit4():
        texto_resumo += "- ⚠ Critério 4: 14 pontos alternando.\n"
    else:
        texto_resumo += "- ✅ Critério 4: Nenhuma alternância suspeita.\n"

    if check_crit5():
        texto_resumo += "- ⚠ Critério 5: 2 de 3 pontos consecutivos além de 2 sigma no mesmo lado.\n"
    else:
        texto_resumo += "- ✅ Critério 5: OK.\n"

    if check_crit6():
        texto_resumo += "- ⚠ Critério 6: 4 de 5 pontos além de 1 sigma no mesmo lado.\n"
    else:
        texto_resumo += "- ✅ Critério 6: OK.\n"

    if check_crit7():
        texto_resumo += "- ⚠ Critério 7: 15 pontos dentro de 1 sigma.\n"
    else:
        texto_resumo += "- ✅ Critério 7: OK.\n"

    if check_crit8():
        texto_resumo += "- ⚠ Critério 8: 8 pontos fora de 1 sigma.\n"
    else:
        texto_resumo += "- ✅ Critério 8: OK.\n"

    if check_crit9():
        texto_resumo += "- ⚠ Critério 9: 12 pontos alternando.\n"
    else:
        texto_resumo += "- ✅ Critério 9: OK.\n"

    # critério 10 reservado
    texto_resumo += "- ℹ Critério 10: Não implementado (padrão Minitab reservado).\n"

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return texto_resumo, img_base64




def analise_limpeza_dados(df):
    import numpy as np
    import pandas as pd

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

        # Linhas faltando
        if n_gaps > 0:
            idx_gaps = df[df[coluna].isnull()].index
            primeira_linha_gap = idx_gaps[0] + 2 if len(idx_gaps) > 0 else "?"
            resultado.append(
                f"Coluna <strong>{coluna}</strong>: {n_gaps} célula(s) vazia(s) (primeira ocorrência na linha {primeira_linha_gap})"
            )

        # Coluna totalmente vazia
        if n_valores == 0:
            resultado.append(f"Coluna <strong>{coluna}</strong>: ⚠ Totalmente vazia")

        # Coluna sem variação
        if df[coluna].nunique(dropna=True) <= 1:
            resultado.append(f"Coluna <strong>{coluna}</strong>: ⚠ Sem variação (todos os valores iguais)")

        # Dados não numéricos em colunas que parecem numéricas
        if df[coluna].dtype == object:
            try:
                pd.to_numeric(df[coluna].dropna())
            except:
                resultado.append(f"Coluna <strong>{coluna}</strong>: ⚠ Contém dados não numéricos suspeitos")

        # Outliers extremos
        if pd.api.types.is_numeric_dtype(df[coluna]):
            media = df[coluna].mean()
            sigma = df[coluna].std()
            if sigma > 0:
                extremos = df[(df[coluna] > media + 10 * sigma) | (df[coluna] < media - 10 * sigma)]
                if not extremos.empty:
                    resultado.append(f"Coluna <strong>{coluna}</strong>: ⚠ {len(extremos)} valor(es) extremamente fora do padrão")

    # Colunas duplicadas
    colunas_duplicadas = []
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            if df[df.columns[i]].equals(df[df.columns[j]]):
                colunas_duplicadas.append(f"{df.columns[i]} / {df.columns[j]}")
    if colunas_duplicadas:
        resultado.append(f"⚠ Colunas duplicadas detectadas: {colunas_duplicadas}")
    else:
        resultado.append("✅ Nenhuma coluna duplicada")

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


