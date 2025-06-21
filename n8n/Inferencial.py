from suporte import *

def analise_1_sample_t(df, colunas_usadas, field=None):
    if len(colunas_usadas) < 1 or not field:
        return "⚠ É obrigatório informar a coluna Y e o valor de referência.", []

    col_y = colunas_usadas[0]
    if col_y not in df.columns:
        return f"⚠ Coluna {col_y} não encontrada no arquivo.", []

    dados = df[col_y].dropna().astype(float)

    if dados.empty:
        return "⚠ Não há dados válidos na coluna Y.", []

    try:
        ref = float(field)
    except ValueError:
        return "⚠ O valor de referência informado não é numérico.", []

    n = len(dados)
    media = dados.mean()
    desvio = dados.std(ddof=1)
    erro_media = desvio / (n ** 0.5)
    ic_low, ic_up = stats.t.interval(0.95, n-1, loc=media, scale=erro_media)

    t_stat, p_value = stats.ttest_1samp(dados, ref)

    resultado = (
        f"**Estatísticas Descritivas**\n"
        f"N: {n}\n"
        f"Média: {media:.2f}\n"
        f"Desvio Padrão: {desvio:.2f}\n"
        f"Erro Padrão da Média: {erro_media:.2f}\n"
        f"IC 95% para μ: ({ic_low:.2f}, {ic_up:.2f})\n\n"
        f"**Teste t para uma amostra (1 Sample T)**\n"
        f"H₀: μ = {ref}\n"
        f"H₁: μ ≠ {ref}\n"
        f"T-Valor: {t_stat:.2f}\n"
        f"P-Valor: {p_value:.3f}\n"
    )

    if p_value < 0.05:
        resultado += "➡ Resultado: Rejeita H0 (diferença significativa)."
    else:
        resultado += "➡ Resultado: Não rejeita H0 (sem diferença significativa)."

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(dados, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))

    ax.plot(media, 1, 'kx', markersize=12)
    ax.hlines(1, ic_low, ic_up, colors='blue', lw=3)
    ax.plot(ref, 1, 'ro')
    ax.text(ref, 1.05, 'H0', color='red')

    ax.set_title(f"Boxplot de {col_y}\n(com H0 e intervalo de confiança de 95% para a média)")
    ax.set_xlabel(col_y)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return resultado, [imagem_base64]


from suporte import *

def analise_2_sample_t(df, colunas_usadas, **kwargs):
    if len(colunas_usadas) != 2:
        return "❌ É necessário selecionar exatamente duas colunas Y numéricas para o Teste 2 Sample T.", None

    col1, col2 = colunas_usadas
    serie1 = pd.to_numeric(df[col1], errors="coerce").dropna()
    serie2 = pd.to_numeric(df[col2], errors="coerce").dropna()

    if len(serie1) < 2 or len(serie2) < 2:
        return "❌ As colunas selecionadas não possuem dados suficientes para o teste.", None

    ad1 = anderson(serie1)
    ad2 = anderson(serie2)
    lim1 = ad1.critical_values[2]
    lim2 = ad2.critical_values[2]
    normal1 = ad1.statistic < lim1
    normal2 = ad2.statistic < lim2

    stat_f = np.var(serie1, ddof=1) / np.var(serie2, ddof=1)
    df1, df2 = len(serie1)-1, len(serie2)-1
    p_f = 2 * min(stats.f.cdf(stat_f, df1, df2), 1 - stats.f.cdf(stat_f, df1, df2))
    equal_var = p_f > 0.05

    t_stat, p_valor = stats.ttest_ind(serie1, serie2, equal_var=equal_var)

    texto = f"""📊 **Teste T para 2 Amostras Independentes**

🔹 Coluna 1: {col1}  
🔹 Coluna 2: {col2}  

🔹 Teste de normalidade (Anderson-Darling, 5%):  
- {col1}: {"✅ Normal" if normal1 else "❌ Não normal"} (estatística = {ad1.statistic:.4f}, limite crítico = {lim1:.4f})  
- {col2}: {"✅ Normal" if normal2 else "❌ Não normal"} (estatística = {ad2.statistic:.4f}, limite crítico = {lim2:.4f})  

🔹 Teste F para igualdade de variâncias:  
- Estatística F = {stat_f:.4f}, p = {p_f:.4f} → {"✅ Variâncias iguais" if equal_var else "❌ Variâncias diferentes"}

🔹 Resultado do Teste T:  
- Estatística t = {t_stat:.4f}, p = {p_valor:.4f}  
- {"✅ Não há diferença significativa" if p_valor > 0.05 else "❌ Diferença estatisticamente significativa entre as médias"}"""

    try:
        aplicar_estilo_minitab()
        fig, ax = plt.subplots(figsize=(6, 6))

        dados_plot = pd.DataFrame({
            'Valor': pd.concat([serie1, serie2]),
            'Grupo': [col1] * len(serie1) + [col2] * len(serie2)
        })

        sns.boxplot(x="Grupo", y="Valor", data=dados_plot, ax=ax, width=0.6, palette="pastel")
        medias = dados_plot.groupby("Grupo")["Valor"].mean()
        ax.plot(range(len(medias)), medias.values, marker="D", linestyle="None", color="black", markersize=6, label="Média")

        ax.set_title(f"Boxplot Comparativo 2 Sample T")
        ax.set_ylabel("Valores")
        ax.legend()

        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        texto += f"\n❌ Erro ao gerar o gráfico: {str(e)}"
        imagem_base64 = None

    return texto, imagem_base64


from suporte import *

def analise_teste_paired_t(df, colunas_usadas):
    if len(colunas_usadas) != 2:
        return "❌ O teste pareado requer exatamente duas colunas.", None

    col1, col2 = colunas_usadas
    serie1 = pd.to_numeric(df[col1], errors="coerce")
    serie2 = pd.to_numeric(df[col2], errors="coerce")

    diferencas = (serie1 - serie2).dropna()
    if len(diferencas) < 2:
        return "❌ Dados insuficientes para o teste pareado.", None

    stat, p_valor = stats.ttest_rel(serie1, serie2, nan_policy='omit')
    media = diferencas.mean()
    desvio = diferencas.std(ddof=1)
    n = len(diferencas)

    t_crit = stats.t.ppf(1 - 0.025, df=n - 1)
    erro = desvio / np.sqrt(n)
    ic = (media - t_crit * erro, media + t_crit * erro)

    interpretacao = f"""📊 **Teste T Pareado**  
🔹 Comparação entre: {col1} e {col2}  
🔹 Número de pares: {n}  
🔹 Média das diferenças: {media:.4f}  
🔹 Desvio padrão das diferenças: {desvio:.4f}  
🔹 Intervalo de confiança (95%): ({ic[0]:.4f}, {ic[1]:.4f})  
🔹 Valor-p: {p_valor:.4f}  

📌 **Conclusão**: {"❌ As médias são estatisticamente diferentes." if p_valor < 0.05 else "✅ Não há diferença estatística entre as médias."}"""

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(diferencas, vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue'))
    ax.set_title("Boxplot das Diferenças")
    ax.set_xlabel("Diferenças")
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.plot(media, 1, marker="x", color="red", markersize=10, label="Média")
    ax.hlines(1, ic[0], ic[1], color="black", linewidth=2, label="IC 95%")
    ax.legend()
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return interpretacao, imagem_base64


from suporte import *

def teste_variancias(df, colunas_usadas):
    if len(colunas_usadas) != 2:
        return "❌ Selecione exatamente duas colunas para comparar as variâncias.", None

    col1, col2 = colunas_usadas
    serie1 = pd.to_numeric(df[col1], errors="coerce").dropna()
    serie2 = pd.to_numeric(df[col2], errors="coerce").dropna()

    if len(serie1) < 3 or len(serie2) < 3:
        return "❌ É necessário pelo menos 3 dados em cada grupo para realizar o teste de variâncias.", None

    stat_f = np.var(serie1, ddof=1) / np.var(serie2, ddof=1)
    df1, df2 = len(serie1)-1, len(serie2)-1
    p_f = 2 * min(stats.f.cdf(stat_f, df1, df2), 1 - stats.f.cdf(stat_f, df1, df2))

    interpretacao = f"""📊 **Teste de Igualdade de Variâncias (F-Teste)**  
🔹 Grupos comparados: {col1} e {col2}  
🔹 Estatística F: {stat_f:.4f}  
🔹 Valor-p (bilateral): {p_f:.4f}  

{"✅ As variâncias são significativamente diferentes." if p_f < 0.05 else "➖ Não há evidência de diferença entre as variâncias."}
"""

    try:
        aplicar_estilo_minitab()
        df_plot = pd.DataFrame({
            "Valor": pd.concat([serie1, serie2]),
            "Grupo": [col1] * len(serie1) + [col2] * len(serie2)
        })
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="Valor", y="Grupo", data=df_plot, orient="h", palette="pastel", showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"})
        plt.title("Comparação das Variâncias (Boxplot)")

        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        print("Erro ao gerar gráfico:", str(e))
        imagem_base64 = None

    return interpretacao, imagem_base64


from suporte import *

def teste_anova(df, colunas_usadas):
    if len(colunas_usadas) < 2:
        return "❌ O Teste ANOVA exige no mínimo duas colunas com dados numéricos (grupos).", None

    dados_grupos = [df[coluna].dropna() for coluna in colunas_usadas]
    normalidade = []
    for i, grupo in enumerate(dados_grupos):
        stat, critico, _ = stats.anderson(grupo)
        if stat < critico[2]:  
            normalidade.append(f"✅ Grupo {colunas_usadas[i]}: distribuição normal (Anderson-Darling)")
        else:
            normalidade.append(f"⚠️ Grupo {colunas_usadas[i]}: não segue distribuição normal")

    try:
        f_stat, p_valor = stats.f_oneway(*dados_grupos)
    except Exception as e:
        return f"❌ Erro ao executar o teste ANOVA: {str(e)}", None

    interpretacao = f"""📊 **Teste ANOVA (Análise de Variância)**  
🔹 Grupos comparados: {", ".join(colunas_usadas)}  
🔹 Estatística F: {f_stat:.4f}  
🔹 Valor-p: {p_valor:.4f}  

📌 Este teste verifica se há diferença significativa entre as médias dos grupos.  
- Se **valor-p < 0.05**, rejeitamos H₀ e concluímos que **pelo menos um grupo tem média diferente**.
- Se **valor-p ≥ 0.05**, **não há evidências suficientes** para afirmar que as médias diferem.

🔍 **Verificação de normalidade (Anderson-Darling, 5%)**:
""" + "\n".join(normalidade)

    try:
        aplicar_estilo_minitab()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(dados_grupos, vert=False, patch_artist=True,
                   labels=colunas_usadas, boxprops=dict(facecolor="skyblue"))
        medias = [grupo.mean() for grupo in dados_grupos]
        for i, media in enumerate(medias, start=1):
            ax.plot(media, i, marker="o", color="red")
        ax.set_title("Boxplot por Grupo (ANOVA)")

        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        print("Erro ao gerar o gráfico:", str(e))
        imagem_base64 = None

    return interpretacao, imagem_base64

ANALISES = {
    "1 Sample T": analise_1_sample_t,
    "2 Sample T": analise_2_sample_t,
    "Paired Test": analise_teste_paired_t,
    "F/Levene Test": teste_variancias,
    "One way ANOVA": teste_anova
}


