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

def analise_paired_t(df: pd.DataFrame, colunas_usadas: list):
    if len(colunas_usadas) != 2:
        return "❌ O teste t pareado requer exatamente 2 colunas Y.", None

    col1, col2 = colunas_usadas
    dados1 = df[col1].dropna()
    dados2 = df[col2].dropna()

    # Garantir tamanhos iguais para o pareamento
    min_len = min(len(dados1), len(dados2))
    dados1 = dados1.iloc[:min_len]
    dados2 = dados2.iloc[:min_len]

    diferencas = dados1 - dados2

    # Teste de normalidade (Anderson-Darling)
    normalidade = stats.anderson(diferencas)
    ad_stat = normalidade.statistic
    ad_crit = normalidade.critical_values
    ad_sig = normalidade.significance_level
    ad_aprovado = ad_stat < ad_crit[list(ad_sig).index(5)] if 5 in ad_sig else False

    # Teste t pareado
    t_stat, p_valor = stats.ttest_rel(dados1, dados2, nan_policy='omit')
    media_diff = np.mean(diferencas)
    desvio_diff = np.std(diferencas, ddof=1)
    n = len(diferencas)
    se_diff = desvio_diff / np.sqrt(n)
    intervalo = stats.t.interval(0.95, n-1, loc=media_diff, scale=se_diff)

    # Conclusão
    conclusao = "✅ As diferenças seguem distribuição normal (Anderson-Darling)." if ad_aprovado else "⚠ As diferenças podem não ser normais (Anderson-Darling)."
    if p_valor < 0.05:
        conclusao += f" ✅ Rejeitamos H0 (p = {p_valor:.4f}). Existe diferença significativa entre as médias."
    else:
        conclusao += f" ⚠ Não rejeitamos H0 (p = {p_valor:.4f}). Não há diferença significativa entre as médias."

    # Gráfico
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(diferencas, bins=8, color='skyblue', edgecolor='black')
    ax.set_title("Histogram of Differences\n(with Ho and 95% t-confidence interval for the mean)")
    ax.set_xlabel("Differences")
    ax.set_ylabel("Frequency")

    # Adiciona H0 (linha no 0)
    ax.axvline(0, color='red', linestyle='--', label='H0')

    # Adiciona intervalo de confiança
    ax.hlines(-0.5, intervalo[0], intervalo[1], color='blue', lw=4)
    ax.text(media_diff, -1, "X̄", color='blue', ha='center')

    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Texto da análise
    texto = f"""
**Teste t Pareado**
- Média das diferenças: {media_diff:.4f}
- Desvio padrão das diferenças: {desvio_diff:.4f}
- N: {n}
- Intervalo 95%: [{intervalo[0]:.4f}, {intervalo[1]:.4f}]
- Estatística t: {t_stat:.4f}
- p-valor: {p_valor:.4f}
- Anderson-Darling: estatística={ad_stat:.4f}, normalidade={'Aprovada' if ad_aprovado else 'Reprovada'}

**Conclusão**
{conclusao}
"""

    return texto.strip(), grafico_base64

def analise_one_way_anova(df: pd.DataFrame, colunas_usadas: list):
    ys = [c for c in colunas_usadas if c != ""]
    x = None
    if "X" in colunas_usadas:
        x = colunas_usadas[-1]

    if len(ys) == 0:
        return "❌ O One way ANOVA requer pelo menos 1 coluna Y.", None

    if x and x in df.columns:
        # ANOVA com Y consolidado e X como grupo
        y_col = ys[0]
        df_valid = df[[y_col, x]].dropna()
        y = df_valid[y_col]
        grupos = [grupo[1].values for grupo in df_valid.groupby(x)[y_col]]

        if len(grupos) < 2:
            return "❌ O One way ANOVA requer pelo menos 2 grupos distintos na coluna X.", None

    else:
        # ANOVA com colunas Ys como grupos
        grupos = []
        for y_col in ys:
            grupo = df[y_col].dropna().values
            if len(grupo) > 0:
                grupos.append(grupo)

        if len(grupos) < 2:
            return "❌ O One way ANOVA requer pelo menos 2 colunas Y com dados.", None

    f_stat, p_valor = stats.f_oneway(*grupos)

    # Normalidade dos resíduos
    concatenado = np.concatenate(grupos)
    residuos = concatenado - np.mean(concatenado)
    normalidade = stats.anderson(residuos)
    ad_stat = normalidade.statistic
    ad_crit = normalidade.critical_values
    ad_sig = list(normalidade.significance_level)
    if 5 in ad_sig:
        idx = ad_sig.index(5)
        ad_aprovado = ad_stat < ad_crit[idx]
    else:
        ad_aprovado = False

    # Conclusão
    conclusao = "✅ Resíduos seguem distribuição normal (Anderson-Darling)." if ad_aprovado else "⚠ Resíduos podem não ser normais (Anderson-Darling)."
    if p_valor < 0.05:
        conclusao += f" ✅ Rejeitamos H0 (p = {p_valor:.4f}). Existem diferenças significativas entre as médias dos grupos."
    else:
        conclusao += f" ⚠ Não rejeitamos H0 (p = {p_valor:.4f}). Não há diferenças significativas entre as médias dos grupos."

    # Gráfico
    fig, ax = plt.subplots(figsize=(6, 4))
    if x and x in df.columns:
        df_valid.boxplot(column=y_col, by=x, ax=ax, grid=False)
        medias = df_valid.groupby(x)[y_col].mean()
        ax.plot(range(1, len(medias) + 1), medias.values, color='blue', marker='o', linestyle='-', label='Médias')
    else:
        ax.boxplot(grupos, labels=ys)
        medias = [np.mean(g) for g in grupos]
        ax.plot(range(1, len(medias) + 1), medias, color='blue', marker='o', linestyle='-', label='Médias')

    ax.set_title("One Way ANOVA - Boxplot por Grupo")
    ax.set_xlabel("Grupo")
    ax.set_ylabel("Valor")
    plt.suptitle("")
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Texto
    texto = f"""
**One Way ANOVA**
- Estatística F: {f_stat:.4f}
- p-valor: {p_valor:.4f}
- Anderson-Darling (resíduos): estatística={ad_stat:.4f}, normalidade={'Aprovada' if ad_aprovado else 'Reprovada'}

**Conclusão**
{conclusao}
"""

    return texto.strip(), grafico_base64


def analise_1_wilcoxon(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) != 1:
        return "❌ O teste 1 Wilcoxon requer exatamente 1 coluna Y.", None

    y_col = colunas_usadas[0]
    y = df[y_col].dropna()

    if len(y) < 5:
        return "❌ O teste requer ao menos 5 valores não nulos.", None

    # Valor de referência
    try:
        valor_ref = float(field) if field is not None else 0
    except:
        return "❌ Valor de referência inválido. Informe um número válido no campo Field.", None

    # Testes de normalidade
    ad = stats.anderson(y)
    sw_stat, sw_p = stats.shapiro(y)
    dp_stat, dp_p = stats.normaltest(y)

    ad_stat = ad.statistic
    ad_crit = ad.critical_values
    ad_sig = list(ad.significance_level)
    if 5 in ad_sig:
        idx = ad_sig.index(5)
        ad_normal = ad_stat < ad_crit[idx]
    else:
        ad_normal = False

    sw_normal = sw_p > 0.05
    dp_normal = dp_p > 0.05

    recomendacao = ""
    if ad_normal or sw_normal or dp_normal:
        recomendacao = "⚠ Pelo menos um teste indicou normalidade dos dados. Considere realizar o teste 1 Sample T em vez do Wilcoxon."

    # Wilcoxon Signed-Rank Test (comparando com valor_ref)
    w_stat, p_valor = stats.wilcoxon(y - valor_ref, zero_method='wilcox', correction=False)

    mediana_amostra = np.median(y)

    # Gráfico
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(y, vert=False)
    ax.axvline(valor_ref, color='red', linestyle='--', label=f'Mediana H0 ({valor_ref})')
    ax.axvline(mediana_amostra, color='blue', linestyle='-', label=f'Mediana amostra: {mediana_amostra:.2f}')
    ax.set_title("1 Wilcoxon - Boxplot com Mediana H0")
    ax.set_xlabel(y_col)
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
**1 Wilcoxon - Teste de Mediana**
- Mediana amostra: {mediana_amostra:.4f}
- Valor de referência (H0): {valor_ref}
- Estatística W: {w_stat:.4f}
- p-valor: {p_valor:.4f}

**Normalidade dos dados**
- Anderson-Darling: estatística={ad_stat:.4f}, normalidade={'Aprovada' if ad_normal else 'Reprovada'}
- Shapiro-Wilk: p-valor={sw_p:.4f}, normalidade={'Aprovada' if sw_normal else 'Reprovada'}
- D’Agostino-Pearson: p-valor={dp_p:.4f}, normalidade={'Aprovada' if dp_normal else 'Reprovada'}

{recomendacao}

**Conclusão**
{"✅ Rejeitamos H0: mediana diferente do valor de referência." if p_valor < 0.05 else "⚠ Não rejeitamos H0: mediana não difere significativamente do valor de referência."}
"""

    return texto.strip(), grafico_base64

def analise_2_mann_whitney(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) != 2:
        return "❌ O teste 2 Mann-Whitney requer exatamente 2 colunas Y.", None

    col1, col2 = colunas_usadas
    dados1 = df[col1].dropna()
    dados2 = df[col2].dropna()

    if len(dados1) < 5 or len(dados2) < 5:
        return "❌ O teste requer ao menos 5 valores não nulos em cada grupo.", None

    # Teste de normalidade dos dois grupos
    # Anderson-Darling
    ad1 = stats.anderson(dados1)
    ad2 = stats.anderson(dados2)
    ad1_stat = ad1.statistic
    ad2_stat = ad2.statistic
    ad1_crit = ad1.critical_values
    ad2_crit = ad2.critical_values
    ad_sig = list(ad1.significance_level)
    if 5 in ad_sig:
        idx = ad_sig.index(5)
        ad1_normal = ad1_stat < ad1_crit[idx]
        ad2_normal = ad2_stat < ad2_crit[idx]
    else:
        ad1_normal = ad2_normal = False

    # Shapiro
    sw1_stat, sw1_p = stats.shapiro(dados1)
    sw2_stat, sw2_p = stats.shapiro(dados2)
    sw1_normal = sw1_p > 0.05
    sw2_normal = sw2_p > 0.05

    # D'Agostino-Pearson
    dp1_stat, dp1_p = stats.normaltest(dados1)
    dp2_stat, dp2_p = stats.normaltest(dados2)
    dp1_normal = dp1_p > 0.05
    dp2_normal = dp2_p > 0.05

    recomendacao = ""
    if ad1_normal or sw1_normal or dp1_normal or ad2_normal or sw2_normal or dp2_normal:
        recomendacao = "⚠ Pelo menos um grupo apresentou indícios de normalidade. Considere realizar o teste 2 Sample T."

    # Mann-Whitney U
    u_stat, p_valor = stats.mannwhitneyu(dados1, dados2, alternative='two-sided')

    mediana1 = np.median(dados1)
    mediana2 = np.median(dados2)

    # Gráfico
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([dados1, dados2], labels=[col1, col2])
    ax.set_title("2 Mann-Whitney - Boxplot por Grupo")
    ax.set_ylabel("Valores")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
**2 Mann-Whitney - Wilcoxon Rank-Sum**
- Mediana {col1}: {mediana1:.4f}
- Mediana {col2}: {mediana2:.4f}
- Estatística U: {u_stat:.4f}
- p-valor: {p_valor:.4f}

**Normalidade dos dados**
{col1}:
- Anderson-Darling: estatística={ad1_stat:.4f}, normalidade={'Aprovada' if ad1_normal else 'Reprovada'}
- Shapiro-Wilk: p-valor={sw1_p:.4f}, normalidade={'Aprovada' if sw1_normal else 'Reprovada'}
- D’Agostino-Pearson: p-valor={dp1_p:.4f}, normalidade={'Aprovada' if dp1_normal else 'Reprovada'}

{col2}:
- Anderson-Darling: estatística={ad2_stat:.4f}, normalidade={'Aprovada' if ad2_normal else 'Reprovada'}
- Shapiro-Wilk: p-valor={sw2_p:.4f}, normalidade={'Aprovada' if sw2_normal else 'Reprovada'}
- D’Agostino-Pearson: p-valor={dp2_p:.4f}, normalidade={'Aprovada' if dp2_normal else 'Reprovada'}

{recomendacao}

**Conclusão**
{"✅ Rejeitamos H0: as distribuições são diferentes." if p_valor < 0.05 else "⚠ Não rejeitamos H0: não há diferença significativa entre as distribuições."}
"""

    return texto.strip(), grafico_base64



ANALISES = {
    "1 Sample T": analise_1_sample_t,
    "2 Sample T": analise_2_sample_t,
    "2 Paired Test": analise_paired_t,
    "One way ANOVA": analise_one_way_anova,
    "1 Wilcoxon": analise_1_wilcoxon,
    "2 Mann-Whitney": analise_2_mann_whitney

    

}


