from suporte import *

def analise_1_sample_t(df, coluna_y, field, field_conf=None):
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    from scipy import stats
    from suporte import aplicar_estilo_minitab

    if not coluna_y:
        return "⚠ É obrigatório informar a coluna Y.", []

    if coluna_y not in df.columns:
        return f"⚠ Coluna {coluna_y} não encontrada no arquivo.", []

    dados = df[coluna_y].dropna().astype(float)
    if dados.empty:
        return "⚠ Não há dados válidos na coluna Y.", []

    try:
        ref = float(field)
    except (ValueError, TypeError):
        return "⚠ O valor de referência informado não é numérico.", []

    try:
        confidence = float(field_conf) if field_conf else 95.0
        if confidence <= 1:
            confidence *= 100
    except (ValueError, TypeError):
        confidence = 95.0

    alpha = 1 - (confidence / 100)

    n = len(dados)
    media = dados.mean()
    desvio = dados.std(ddof=1)
    ic_low, ic_up = stats.t.interval(1 - alpha, n - 1, loc=media, scale=desvio / (n ** 0.5))

    t_stat, p_value = stats.ttest_1samp(dados, ref)

    # 🔷 Ajuste para vírgula no padrão BR
    def br(value):
        return str(round(value, 2)).replace('.', ',')

    resultado = (
        f"📊 **Análise – Teste T para uma Amostra ({coluna_y})**\n\n"
        f"🔎 **Estatísticas Descritivas:**\n"
        f"- **N:** {n}\n"
        f"- **Média:** {br(media)}\n"
        f"- **Desvio Padrão:** {br(desvio)}\n"
        f"- **IC {confidence:.0f}% para μ:** ({br(ic_low)} ; {br(ic_up)})\n\n"
        f"🔎 **Teste T (1 Sample T):**\n"
        f"- **Hipótese Conservadora (H₀):** μ = {br(ref)}\n"
        f"- **Hipótese Alternativa (H₁):** μ ≠ {br(ref)}\n"
        f"- **T-Valor:** {br(t_stat)}\n"
        f"- **P-Valor:** {br(p_value)}\n\n"
    )

    if p_value < alpha:
        resultado += f"🔎 **Conclusão:**\nCom {confidence:.0f}% de confiança, podemos rejeitar a hipótese conservadora. Logo, há diferença estatisticamente significativa entre a média amostral ({br(media)}) e o valor {br(ref)}."
    else:
        resultado += f"🔎 **Conclusão:**\nCom {confidence:.0f}% de confiança, não podemos rejeitar a hipótese conservadora. Logo, não há diferença estatisticamente significativa entre a média amostral ({br(media)}) e o valor {br(ref)}."

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(8, 4))

    # Boxplot cinza neutro COM outliers
    ax.boxplot(dados, vert=False, patch_artist=True,
               boxprops=dict(facecolor='lightgrey', color='black'),
               medianprops=dict(color='black'))

    # 🔴 Linha da média no próprio boxplot (vermelha pontilhada) desde topo do boxplot até IC
    box_y = 1  # posição y do boxplot
    ax.vlines(media, box_y, 0.85, color='red', linestyle='--', lw=2)
    ax.text(media, box_y + 0.05, 'x̄', ha='center', fontsize=10, color='red')

    # 🔵 Desenha IC abaixo do boxplot com barras verticais nas extremidades
    ax.hlines(0.85, ic_low, ic_up, colors='blue', lw=2)
    ax.vlines(ic_low, 0.82, 0.88, colors='blue', lw=2)
    ax.vlines(ic_up, 0.82, 0.88, colors='blue', lw=2)
    ax.text(media, 0.80, f"IC {confidence:.0f}%", ha='center', fontsize=9, color='blue')

    # 🔴 Marca H0 como bolinha vermelha NO MESMO NÍVEL DA LINHA AZUL DO IC
    ax.plot(ref, 0.85, 'ro')
    ax.text(ref, 0.75, 'H0', ha='center', fontsize=9, color='red')

    # Ajusta limites y para mostrar tudo
    ax.set_ylim(0.65, 1.3)

    # Ajusta título no padrão Minitab
    ax.set_title(f"Boxplot de {coluna_y}\n(com H0 e intervalo de confiança de {confidence:.0f}% para a média)", fontsize=11)
    ax.set_xlabel(coluna_y, fontsize=10)
    ax.set_yticks([])

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")



    return resultado, [imagem_base64]









from suporte import *

def analise_2_sample_t(df, lista_y, field_conf=None):
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.stats import anderson, shapiro, kstest, norm
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    if len(lista_y) != 2:
        return "❌ É necessário selecionar exatamente duas colunas Y numéricas para o Teste 2 Sample T.", None

    col1, col2 = lista_y
    serie1 = pd.to_numeric(df[col1], errors="coerce").dropna()
    serie2 = pd.to_numeric(df[col2], errors="coerce").dropna()

    if len(serie1) < 2 or len(serie2) < 2:
        return "❌ As colunas selecionadas não possuem dados suficientes para o teste.", None

    try:
        confidence = float(field_conf) if field_conf else 95.0
        if confidence <= 1:
            confidence *= 100
    except (ValueError, TypeError):
        confidence = 95.0

    alpha = 1 - (confidence / 100)

    # Função para normalidade consolidada
    def normalidade_consolidada(serie):
        ad = anderson(serie)
        normal_ad = ad.statistic < ad.critical_values[2]

        stat_shapiro, p_shapiro = shapiro(serie)
        normal_shapiro = p_shapiro > 0.05

        stat_ks, p_ks = kstest(serie, 'norm', args=(serie.mean(), serie.std(ddof=1)))
        normal_ks = p_ks > 0.05

        return normal_ad or normal_shapiro or normal_ks

    normal1 = normalidade_consolidada(serie1)
    normal2 = normalidade_consolidada(serie2)

    # Teste F
    stat_f = np.var(serie1, ddof=1) / np.var(serie2, ddof=1)
    df1, df2 = len(serie1)-1, len(serie2)-1
    p_f = 2 * min(stats.f.cdf(stat_f, df1, df2), 1 - stats.f.cdf(stat_f, df1, df2))
    equal_var = p_f > alpha

    # Teste T
    t_stat, p_valor = stats.ttest_ind(serie1, serie2, equal_var=equal_var)

    # 🔷 Ajuste para vírgula no padrão BR
    def br(value):
        return str(round(value, 2)).replace('.', ',')

    texto = f"""📊 **Análise – Teste T para 2 Amostras Independentes**

🔹 **Hipóteses:**
- **H₀:** As médias de {col1} e {col2} são iguais
- **H₁:** As médias de {col1} e {col2} são diferentes

🔎 **Estatísticas Descritivas:**

**{col1}:**
N = {len(serie1)}
Média = {br(serie1.mean())}
Desvio Padrão = {br(serie1.std(ddof=1))}

**{col2}:**
N = {len(serie2)}
Média = {br(serie2.mean())}
Desvio Padrão = {br(serie2.std(ddof=1))}

🔎 **Testes de Normalidade (Anderson-Darling, Ryan-Joiner, Kolmogorov-Smirnov):**
- {col1}: {"✅ Os dados podem ser considerados normais" if normal1 else "❌ Os dados não são normais"}
- {col2}: {"✅ Os dados podem ser considerados normais" if normal2 else "❌ Os dados não são normais"}

🔎 **Teste F para igualdade de variâncias:**
Estatística F = {br(stat_f)}
p-valor = {br(p_f)}
{"✅ Variâncias iguais, será usado o Teste T padrão" if equal_var else "❌ Variâncias diferentes, será usado o Teste T de Welch"}

🔎 **Teste T para 2 Amostras:**
Estatística t = {br(t_stat)}
p-valor = {br(p_valor)}

🔎 **Conclusão:**
Com {confidence:.0f}% de confiança, {"podemos rejeitar a hipótese conservadora. Logo, há diferença estatisticamente significativa entre as médias" if p_valor < alpha else "não podemos rejeitar a hipótese conservadora. Logo, não há diferença estatisticamente significativa entre as médias"} de {col1} ({br(serie1.mean())}) e {col2} ({br(serie2.mean())}).
"""

    if not normal1 or not normal2:
        texto += "\n⚠️ Como pelo menos um dos conjuntos de dados não é normal, recomenda-se coletar mais dados e/ou verificar a estabilidade do processo."

    # Gráfico
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

def analise_paired_t(df: pd.DataFrame, lista_y: list, field_conf=None):
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.stats import anderson, shapiro, kstest, norm
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    if len(lista_y) != 2:
        return "❌ O teste t pareado requer exatamente 2 colunas Y.", None

    col1, col2 = lista_y
    dados1 = df[col1].dropna()
    dados2 = df[col2].dropna()

    # Garantir tamanhos iguais para o pareamento
    min_len = min(len(dados1), len(dados2))
    dados1 = dados1.iloc[:min_len]
    dados2 = dados2.iloc[:min_len]
    diferencas = dados1 - dados2

    # Nível de confiança
    try:
        confidence = float(field_conf) if field_conf else 95.0
        if confidence <= 1:
            confidence *= 100
    except (ValueError, TypeError):
        confidence = 95.0
    alpha = 1 - (confidence / 100)

    n = len(diferencas)
    media_diff = np.mean(diferencas)
    desvio_diff = np.std(diferencas, ddof=1)
    se_diff = desvio_diff / np.sqrt(n)
    intervalo = stats.t.interval(1 - alpha, n-1, loc=media_diff, scale=se_diff)
    t_stat, p_valor = stats.ttest_rel(dados1, dados2, nan_policy='omit')

    # Testes de normalidade consolidados
    ad = anderson(diferencas)
    normal_ad = ad.statistic < ad.critical_values[2]

    stat_shapiro, p_shapiro = shapiro(diferencas)
    normal_shapiro = p_shapiro > 0.05

    stat_ks, p_ks = kstest(diferencas, 'norm', args=(media_diff, desvio_diff))
    normal_ks = p_ks > 0.05

    normal_final = normal_ad or normal_shapiro or normal_ks

    # 🔷 Ajuste BR
    def br(value):
        return str(round(value, 2)).replace('.', ',')

    # Reporte
    texto = f"""
📊 **Análise – Teste T Pareado**

🔹 **Hipóteses:**
- **H₀:** A média das diferenças entre os pares é igual a zero
- **H₁:** A média das diferenças entre os pares é diferente de zero

🔎 **Estatísticas Descritivas das Diferenças:**
- N = {n}
- Média = {br(media_diff)}
- Desvio Padrão = {br(desvio_diff)}
- Intervalo de Confiança ({confidence:.0f}%) = [{br(intervalo[0])} ; {br(intervalo[1])}]

🔎 **Testes de Normalidade (Anderson-Darling, Shapiro-Wilk, Kolmogorov-Smirnov):**
- Diferenças: {"✅ Os dados podem ser considerados normais" if normal_final else "❌ Os dados não são normais"}

🔎 **Teste T Pareado:**
- Estatística t = {br(t_stat)}
- p-valor = {br(p_valor)}

🔎 **Conclusão:**
Com {confidence:.0f}% de confiança, {"podemos rejeitar a hipótese conservadora. Logo, há diferença estatisticamente significativa entre os pares avaliados." if p_valor < alpha else "não podemos rejeitar a hipótese conservadora. Logo, não há diferença estatisticamente significativa entre os pares avaliados."}
"""

    if not normal_final:
        texto += "\n⚠️ Como as diferenças não são normais, recomenda-se coletar mais dados e/ou verificar a estabilidade do processo."

    # Gráfico
    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(diferencas, bins=8, color='skyblue', edgecolor='black')
    ax.set_title(f"Histograma das Diferenças\n(com H0 e intervalo t {confidence:.0f}% para a média)")
    ax.set_xlabel("Diferenças")
    ax.set_ylabel("Frequência")

    # Linha H0
    ax.axvline(0, color='red', linestyle='--', label='H0')

    # Intervalo de confiança
    ax.hlines(-0.5, intervalo[0], intervalo[1], color='blue', lw=4, label=f"IC {confidence:.0f}%")
    ax.text(media_diff, -1, "X̄", color='blue', ha='center')

    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return texto.strip(), grafico_base64




def analise_one_way_anova(df: pd.DataFrame, lista_y: list, subgrupo=None, field_conf=None):
    import pandas as pd
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    ys = [c for c in lista_y if c not in ["", "Subgrupo"]]
    x = subgrupo if subgrupo and subgrupo in df.columns else None

    if len(ys) == 0:
        return "❌ O One way ANOVA requer pelo menos 1 coluna Y.", None

    try:
        confidence = float(field_conf) if field_conf else 95.0
        if confidence <= 1:
            confidence *= 100
    except (TypeError, ValueError):
        confidence = 95.0

    alpha = 1 - (confidence / 100)

    if x:
        y_col = ys[0]
        df_valid = df[[y_col, x]].dropna()
        if df_valid[x].nunique() < 2:
            return "❌ O One way ANOVA requer pelo menos 2 grupos distintos na coluna Subgrupo.", None
        grupos = [grupo[1].values for grupo in df_valid.groupby(x)[y_col]]
    else:
        grupos = []
        for y_col in ys:
            grupo = df[y_col].dropna().values
            if len(grupo) > 0:
                grupos.append(grupo)
        if len(grupos) < 2:
            return "❌ O One way ANOVA requer pelo menos 2 colunas Y com dados.", None

    # 🔎 Estatísticas ANOVA
    f_stat, p_valor = stats.f_oneway(*grupos)

    # 🔎 Testes de normalidade (Anderson-Darling, Shapiro-Wilk, Kolmogorov-Smirnov)
    concatenado = np.concatenate(grupos)
    residuos = concatenado - np.mean(concatenado)

    ad_stat, ad_crit, ad_sig = stats.anderson(residuos)
    ad_pass = ad_stat < ad_crit[list(ad_sig).index(5)] if 5 in ad_sig else False

    sw_stat, sw_p = stats.shapiro(residuos)
    sw_pass = sw_p > alpha

    ks_stat, ks_p = stats.kstest(residuos, 'norm', args=(np.mean(residuos), np.std(residuos)))
    ks_pass = ks_p > alpha

    normalidade_geral = ad_pass or sw_pass or ks_pass

    # 🔎 Conclusão
    if p_valor < alpha:
        conclusao = f"Com {confidence:.1f}% de confiança, podemos rejeitar a hipótese conservadora. Logo, há diferenças estatisticamente significativas entre as médias dos grupos."
    else:
        conclusao = f"Com {confidence:.1f}% de confiança, não podemos rejeitar a hipótese conservadora. Logo, não há diferenças estatisticamente significativas entre as médias dos grupos."

    # 📊 Texto Final
    texto = f"""
📊 **Análise – One Way ANOVA**

🔹 **Hipóteses:**
- H₀: As médias dos grupos são iguais
- H₁: Pelo menos uma média de grupo é diferente

🔎 **Estatísticas ANOVA:**
- Estatística F = {f_stat:.4f}
- p-valor = {p_valor:.4f}
- Nível de confiança = {confidence:.1f}%

🔎 **Testes de Normalidade (Anderson-Darling, Shapiro-Wilk, Kolmogorov-Smirnov):**
- Resíduos: {"✅ Podem ser considerados normais" if normalidade_geral else "❌ Podem não ser normais"}

🔎 **Conclusão:**
{conclusao}
""".strip()

    # 🔷 Gráfico
    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    if x:
        df_valid.boxplot(column=y_col, by=x, ax=ax, grid=False)
        medias = df_valid.groupby(x)[y_col].mean()
        ax.plot(range(1, len(medias) + 1), medias.values, color='blue', marker='o', linestyle='-', label='Médias')
    else:
        ax.boxplot(grupos, labels=ys)
        medias = [np.mean(g) for g in grupos]
        ax.plot(range(1, len(medias) + 1), medias, color='blue', marker='o', linestyle='-', label='Médias')

    ax.set_title(f"One Way ANOVA – Boxplot por Grupo (IC {confidence:.1f}%)", fontsize=10)
    ax.set_xlabel("Grupo")
    ax.set_ylabel("Valor")
    plt.suptitle("")
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return texto, grafico_base64



def analise_1_intervalo_confianca(df: pd.DataFrame, coluna_y, field_conf=None):
    if not coluna_y:
        return "❌ O intervalo de confiança requer exatamente 1 coluna Y.", None

    if coluna_y not in df.columns:
        return f"❌ A coluna {coluna_y} não foi encontrada no arquivo.", None

    y = df[coluna_y].dropna()

    if len(y) < 5:
        return "❌ O teste requer ao menos 5 valores não nulos.", None

    # Nível de confiança informado
    try:
        nivel_conf = float(field_conf) if field_conf else 95.0
        if nivel_conf <= 1:  # Permite entrada como 0.95
            nivel_conf *= 100
        if not (50 <= nivel_conf < 100):
            return "❌ O nível de confiança deve ser entre 50 e 99.9.", None
    except:
        return "❌ Valor do nível de confiança inválido. Informe um número (ex.: 95).", None

    alpha = 1 - (nivel_conf / 100)

    media = np.mean(y)
    desvio = np.std(y, ddof=1)
    n = len(y)
    se = desvio / np.sqrt(n)
    intervalo = stats.t.interval(1 - alpha, n-1, loc=media, scale=se)

    # Normalidade dos dados
    ad = stats.anderson(y)
    sw_stat, sw_p = stats.shapiro(y)
    dp_stat, dp_p = stats.normaltest(y)

    ad_crit = ad.critical_values
    ad_sig = list(ad.significance_level)
    if 5 in ad_sig:
        idx = ad_sig.index(5)
        ad_normal = ad.statistic < ad_crit[idx]
    else:
        ad_normal = False
    sw_normal = sw_p > 0.05
    dp_normal = dp_p > 0.05

    # Conclusão baseada em normalidade
    if ad_normal or sw_normal or dp_normal:
        normalidade_texto = "✅ Os dados podem ser considerados normais."
        conclusao = f"Com {nivel_conf:.1f}% de confiança, estima-se que a média populacional esteja entre {intervalo[0]:.2f} e {intervalo[1]:.2f}. Como os dados podem ser considerados normais, este intervalo é confiável para inferências estatísticas."
    else:
        normalidade_texto = "⚠ Os dados podem não ser normais."
        conclusao = f"Com {nivel_conf:.1f}% de confiança, estima-se que a média populacional esteja entre {intervalo[0]:.2f} e {intervalo[1]:.2f}. Contudo, como os dados não seguem distribuição normal, a estimativa do intervalo de confiança pode não ser confiável.\n\nRecomenda-se coletar mais dados para avaliar novamente a distribuição, verificar a estabilidade do processo ou utilizar métodos não paramétricos (ex: análise de medianas)."

    # Gráfico
    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.errorbar(media, 0, xerr=[[media - intervalo[0]], [intervalo[1] - media]], fmt='o', color='blue', ecolor='black', capsize=5)
    ax.axvline(media, color='blue', linestyle='-', label=f'Média: {media:.2f}')
    ax.set_yticks([])
    ax.set_title(f"Intervalo de Confiança {nivel_conf:.1f}%")
    ax.set_xlabel(coluna_y)
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
📊 **Análise – Intervalo de Confiança**

🔹 **Hipóteses (usadas de forma geral):**
- **H₀:** A média populacional é igual à média amostral
- **H₁:** A média populacional é diferente da média amostral

🔎 **Estatísticas Descritivas:**
- **N:** {n}
- **Média:** {media:.2f}
- **Desvio Padrão:** {desvio:.2f}
- **Nível de Confiança:** {nivel_conf:.1f}%
- **Intervalo de Confiança:** [{intervalo[0]:.2f} ; {intervalo[1]:.2f}]

🔎 **Testes de Normalidade (Anderson-Darling, Shapiro-Wilk, D’Agostino-Pearson):**
{normalidade_texto}

🔎 **Conclusão:**
{conclusao}
"""

    return texto.strip(), grafico_base64



def analise_1_wilcoxon(df: pd.DataFrame, coluna_y, field, field_conf=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    from scipy import stats
    from suporte import aplicar_estilo_minitab

    if not coluna_y:
        return "❌ O teste 1 Wilcoxon requer exatamente 1 coluna Y.", None

    if coluna_y not in df.columns:
        return f"❌ A coluna {coluna_y} não foi encontrada no arquivo.", None

    y = df[coluna_y].dropna()

    if len(y) < 5:
        return "❌ O teste requer ao menos 5 valores não nulos.", None

    # Valor de referência
    try:
        valor_ref = float(field) if field is not None else 0
    except:
        return "❌ Valor de referência inválido. Informe um número válido no campo Field.", None

    # Nível de confiança
    try:
        confidence = float(field_conf) if field_conf else 95.0
        if confidence <= 1:
            confidence *= 100
    except (ValueError, TypeError):
        confidence = 95.0

    alpha = 1 - (confidence / 100)

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
    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    box = ax.boxplot(y, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))

    # Linha azul apenas dentro do boxplot
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    ax.vlines(mediana_amostra, 0.85, 1.15, color='blue', linestyle='-', label=f'Mediana amostra: {mediana_amostra:.2f}')

    # Linha vermelha pontilhada para mediana H0
    ax.axvline(valor_ref, color='red', linestyle='--', label=f'Mediana H0 ({valor_ref})')

    ax.set_title(f"1 Wilcoxon - Boxplot com Mediana H0 (IC {confidence:.1f}%)", fontsize=10)
    ax.set_xlabel(coluna_y)
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
📊 **Análise – Teste de Mediana (Wilcoxon)**

🔹 **Hipóteses:**
- H₀: A mediana populacional é igual a {valor_ref}
- H₁: A mediana populacional é diferente de {valor_ref}

🔎 **Resultados:**
- Mediana amostra: {mediana_amostra:.2f}
- Estatística W: {w_stat:.4f}
- p-valor: {p_valor:.4f}
- Nível de confiança: {confidence:.1f}%

🔎 **Testes de Normalidade (Anderson-Darling, Shapiro-Wilk, D’Agostino-Pearson):**
{"✅ Os dados podem ser considerados normais" if (ad_normal or sw_normal or dp_normal) else "⚠ Os dados podem não seguir distribuição normal"}

{recomendacao}

🔎 **Conclusão:**
{"✅ Rejeitamos H0. A mediana populacional difere significativamente de " + str(valor_ref) + "." if p_valor < alpha else "⚠ Não rejeitamos H0. A mediana populacional não difere significativamente de " + str(valor_ref) + "."}
"""

    return texto.strip(), grafico_base64




def analise_2_mann_whitney(df: pd.DataFrame, lista_y: list, field_conf=None):
    if len(lista_y) != 2:
        return "❌ O teste 2 Mann-Whitney requer exatamente 2 colunas Y.", None

    col1, col2 = lista_y
    dados1 = df[col1].dropna()
    dados2 = df[col2].dropna()

    if len(dados1) < 5 or len(dados2) < 5:
        return "❌ O teste requer ao menos 5 valores não nulos em cada grupo.", None

    try:
        confidence = float(field_conf) if field_conf else 95.0
        if confidence <= 1:
            confidence *= 100
    except (ValueError, TypeError):
        confidence = 95.0

    alpha = 1 - (confidence / 100)

    # Testes de normalidade
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

    sw1_stat, sw1_p = stats.shapiro(dados1)
    sw2_stat, sw2_p = stats.shapiro(dados2)
    sw1_normal = sw1_p > 0.05
    sw2_normal = sw2_p > 0.05

    dp1_stat, dp1_p = stats.normaltest(dados1)
    dp2_stat, dp2_p = stats.normaltest(dados2)
    dp1_normal = dp1_p > 0.05
    dp2_normal = dp2_p > 0.05

    # ✅ Observação apenas se ambos forem normais
    recomendacao = ""
    if (ad1_normal or sw1_normal or dp1_normal) and (ad2_normal or sw2_normal or dp2_normal):
        recomendacao = "⚠ Observação: Os dois grupos apresentaram indícios de normalidade. Logo, o teste paramétrico 2 Sample T seria mais apropriado."

    u_stat, p_valor = stats.mannwhitneyu(dados1, dados2, alternative='two-sided')

    mediana1 = np.median(dados1)
    mediana2 = np.median(dados2)

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([dados1, dados2], labels=[col1, col2], patch_artist=True,
               boxprops=dict(facecolor='lightblue'))
    ax.set_title(f"2 Mann-Whitney - Boxplot por Grupo (IC {confidence:.1f}%)", fontsize=10)
    ax.set_ylabel("Valores")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
📊 **Análise – Teste Mann-Whitney (Wilcoxon Rank-Sum)**

🔹 **Hipóteses:**
- H₀: As medianas dos dois grupos são iguais
- H₁: As medianas dos dois grupos são diferentes

🔎 **Estatísticas Descritivas:**

**{col1}:**
- Mediana = {mediana1:.2f}

**{col2}:**
- Mediana = {mediana2:.2f}

🔎 **Resultados do Teste Mann-Whitney:**
- Estatística U = {u_stat:.4f}
- p-valor = {p_valor:.4f}
- Nível de confiança = {confidence:.1f}%

🔎 **Testes de Normalidade (Anderson-Darling, Shapiro-Wilk, D’Agostino-Pearson):**

**{col1}:** {'✅ Os dados podem ser considerados normais' if (ad1_normal or sw1_normal or dp1_normal) else '⚠ Os dados podem não ser normais'}

**{col2}:** {'✅ Os dados podem ser considerados normais' if (ad2_normal or sw2_normal or dp2_normal) else '⚠ Os dados podem não ser normais'}

{recomendacao}

🔎 **Conclusão:**
{"✅ Rejeitamos H0: há diferença estatisticamente significativa entre as medianas dos grupos." if p_valor < alpha else "⚠ Não rejeitamos H0: não há diferença estatisticamente significativa entre as medianas dos grupos."}
"""

    return texto.strip(), grafico_base64

def analise_2_wilcoxon_paired(df: pd.DataFrame, lista_y: list, field_conf=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    from scipy import stats
    from suporte import aplicar_estilo_minitab

    if len(lista_y) != 2:
        return "❌ O teste 2 Wilcoxon Pareado requer exatamente 2 colunas Y.", None

    col1, col2 = lista_y
    dados1 = pd.to_numeric(df[col1], errors="coerce").dropna()
    dados2 = pd.to_numeric(df[col2], errors="coerce").dropna()

    if len(dados1) < 5 or len(dados2) < 5:
        return "❌ O teste requer ao menos 5 valores não nulos em cada grupo.", None

    try:
        confidence = float(field_conf) if field_conf else 95.0
        if confidence <= 1:
            confidence *= 100
    except (ValueError, TypeError):
        confidence = 95.0

    alpha = 1 - (confidence / 100)

    # Garantir mesmo tamanho para pareamento
    n = min(len(dados1), len(dados2))
    dados1 = dados1.iloc[:n]
    dados2 = dados2.iloc[:n]

    # Diferenças
    diferencas = dados1 - dados2
    mediana_diff = np.median(diferencas)

    # Testes de normalidade
    ad = stats.anderson(diferencas)
    sw_stat, sw_p = stats.shapiro(diferencas)
    dp_stat, dp_p = stats.normaltest(diferencas)

    ad_crit = ad.critical_values
    ad_sig = list(ad.significance_level)
    if 5 in ad_sig:
        idx = ad_sig.index(5)
        ad_normal = ad.statistic < ad_crit[idx]
    else:
        ad_normal = False

    sw_normal = sw_p > 0.05
    dp_normal = dp_p > 0.05

    # Resultado consolidado dos testes
    if ad_normal or sw_normal or dp_normal:
        normalidade = "✅ A diferença dos dados pode ser considerada normal."
    else:
        normalidade = "⚠ A diferença dos dados não segue distribuição normal."

    # Teste Wilcoxon pareado
    w_stat, p_valor = stats.wilcoxon(diferencas)

    # Conclusão
    if p_valor < alpha:
        conclusao = f"Com {confidence:.1f}% de confiança, podemos rejeitar a hipótese conservadora. Logo, há diferença estatisticamente significativa entre as medianas dos pares avaliados."
    else:
        conclusao = f"Com {confidence:.1f}% de confiança, não podemos rejeitar a hipótese conservadora. Logo, não há diferença estatisticamente significativa entre as medianas dos pares avaliados."

    # Gráfico
    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(diferencas, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))

    # Linha vertical da mediana (dentro do boxplot)
    q1 = np.percentile(diferencas, 25)
    q3 = np.percentile(diferencas, 75)
    ax.vlines(mediana_diff, 0.85, 1.15, color='blue', linestyle='-', label=f'Mediana: {mediana_diff:.2f}')

    # Linha H0 no zero
    ax.axvline(0, color='red', linestyle='--', label='H0 (0)')

    ax.set_title(f"2 Wilcoxon Paired – Diferenças Pareadas", fontsize=10)
    ax.set_xlabel(f"Diferença ({col1} - {col2})")
    ax.set_yticks([])

    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Reporte final
    texto = f"""
📊 **Análise – 2 Wilcoxon Paired**

🔹 **Estatísticas Descritivas das Diferenças:**
- N = {n}
- Mediana = {mediana_diff:.2f}

🔎 **Testes de Normalidade (Anderson-Darling, Shapiro-Wilk, D’Agostino-Pearson):**
- {normalidade}

🔎 **Teste Wilcoxon Pareado:**
- Estatística W = {w_stat:.4f}
- p-valor = {p_valor:.4f}

🔎 **Conclusão:**
{conclusao}
"""

    return texto.strip(), grafico_base64




def analise_kruskal_wallis(df: pd.DataFrame, lista_y: list, subgrupo=None, field_conf=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    # Validação básica
    if not lista_y:
        return "❌ É necessário selecionar pelo menos 1 coluna Y.", None

    if len(lista_y) == 1 and not subgrupo:
        return "❌ Se apenas 1 coluna Y for selecionada, é obrigatório informar uma coluna Subgrupo.", None

    # Definição dos grupos
    if len(lista_y) > 1:
        # Cenário 1: várias colunas Y (cada coluna é um grupo)
        grupos = []
        for col in lista_y:
            if col not in df.columns:
                return f"❌ A coluna {col} não foi encontrada no DataFrame.", None
            grupo = df[col].dropna().values
            if len(grupo) < 2:
                return f"❌ O grupo {col} não possui dados suficientes.", None
            grupos.append(grupo)
        nomes_grupos = lista_y

    else:
        # Cenário 2: 1 coluna Y + subgrupo X
        y_col = lista_y[0]
        if subgrupo not in df.columns:
            return f"❌ A coluna Subgrupo ({subgrupo}) não foi encontrada no DataFrame.", None
        df_valid = df[[y_col, subgrupo]].dropna()
        if df_valid[subgrupo].nunique() < 2:
            return "❌ O Subgrupo deve ter pelo menos 2 categorias diferentes.", None
        grupos = [g[1].values for g in df_valid.groupby(subgrupo)[y_col]]
        nomes_grupos = df_valid[subgrupo].unique()

    # Teste Kruskal-Wallis
    h_stat, p_valor = stats.kruskal(*grupos)

    # Teste de normalidade em cada grupo
    normal_flag = True
    for g in grupos:
        if len(g) >= 5:
            ad = stats.anderson(g)
            sw_stat, sw_p = stats.shapiro(g)
            dp_stat, dp_p = stats.normaltest(g)
            ad_crit = ad.critical_values
            ad_sig = list(ad.significance_level)
            if 5 in ad_sig:
                idx = ad_sig.index(5)
                ad_normal = ad.statistic < ad_crit[idx]
            else:
                ad_normal = False
            sw_normal = sw_p > 0.05
            dp_normal = dp_p > 0.05
            if not (ad_normal or sw_normal or dp_normal):
                normal_flag = False

    # Recomendação ANOVA
    recomendacao = ""
    if normal_flag:
        recomendacao = "⚠ Observação: Como todos os grupos apresentaram indícios de normalidade, recomenda-se considerar o uso do teste paramétrico One Way ANOVA, caso os pressupostos sejam atendidos."

        # Gráfico
    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(lista_y) > 1:
        ax.boxplot(grupos, labels=nomes_grupos)
        medianas = [np.median(g) for g in grupos]
        ax.plot(range(1, len(medianas)+1), medianas, color='blue', marker='o', linestyle='-', label='Medianas')
    else:
        df_valid.boxplot(column=y_col, by=subgrupo, ax=ax, grid=False)
        medianas = df_valid.groupby(subgrupo)[y_col].median()
        ax.plot(range(1, len(medianas)+1), medianas.values, color='blue', marker='o', linestyle='-', label='Medianas')

    ax.set_title("Kruskal-Wallis - Boxplot por Grupo")
    ax.set_xlabel("Grupo")
    ax.set_ylabel("Valor")
    plt.suptitle("")
    ax.legend()
    plt.tight_layout()


    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Texto do relatório
    texto = f"""
📊 Análise – Kruskal-Wallis

🔹 Hipóteses:
- H₀: As medianas dos grupos são iguais
- H₁: Pelo menos uma mediana de grupo é diferente

🔎 Estatísticas Kruskal-Wallis:
- Estatística H = {h_stat:.4f}
- p-valor = {p_valor:.4f}

{recomendacao}

🔎 Conclusão:
{"✅ Rejeitamos H0: há diferença estatisticamente significativa entre as medianas dos grupos." if p_valor < 0.05 else "⚠ Não rejeitamos H0: não há diferença estatisticamente significativa entre as medianas dos grupos."}
"""

    return texto.strip(), grafico_base64




def analise_friedman_pareado(df: pd.DataFrame, lista_y: list, subgrupo=None, field_conf=None):
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    from suporte import aplicar_estilo_minitab

    ys = [c for c in lista_y if c != ""]
    x = subgrupo if subgrupo and subgrupo in df.columns else None

    if len(ys) < 2 and not x:
        return "❌ O Friedman requer pelo menos 2 colunas Y ou 1 Y + Subgrupo com 2 ou mais categorias.", None

    try:
        confidence = float(field_conf) if field_conf else 95.0
        if confidence <= 1:
            confidence *= 100
    except (ValueError, TypeError):
        confidence = 95.0

    alpha = 1 - (confidence / 100)

    # Preparar dados
    if x:
        y_col = ys[0]
        df_valid = df[[y_col, x]].dropna()

        if df_valid[x].nunique() < 2:
            return "❌ O Friedman requer pelo menos 2 grupos distintos na coluna Subgrupo.", None

        # Pivotar em formato wide, apenas IDs completos
        df_valid['ID_temp'] = df_valid.groupby(x).cumcount()
        pivot = df_valid.pivot(index='ID_temp', columns=x, values=y_col).dropna()

        if pivot.shape[1] < 2:
            return "❌ O Friedman requer pelo menos 2 grupos completos para análise pareada.", None

        grupos = [pivot[col].values for col in pivot.columns]
        labels = pivot.columns.tolist()
    else:
        df_valid = df[ys].dropna()
        if df_valid.shape[1] < 2:
            return "❌ O Friedman requer pelo menos 2 colunas Y com dados.", None

        grupos = [df_valid[col].values for col in df_valid.columns]
        labels = ys

    # Garantir pareamento
    min_len = min(len(g) for g in grupos)
    grupos = [g[:min_len] for g in grupos]

    if min_len < 2:
        return "❌ O teste Friedman Pareado requer pelo menos 2 observações pareadas.", None

    # Executa teste
    try:
        stat, p_valor = stats.friedmanchisquare(*grupos)
    except Exception as e:
        return f"❌ Erro ao executar o teste Friedman: {str(e)}", None

    # Normalidade dos resíduos
    dados_matrix = np.array(grupos).T
    residuos = dados_matrix - np.mean(dados_matrix, axis=1, keepdims=True)
    residuos_flat = residuos.flatten()

    ad = stats.anderson(residuos_flat)
    sw_stat, sw_p = stats.shapiro(residuos_flat)
    dp_stat, dp_p = stats.normaltest(residuos_flat)

    ad_crit = ad.critical_values
    ad_sig = list(ad.significance_level)
    if 5 in ad_sig:
        idx = ad_sig.index(5)
        ad_normal = ad.statistic < ad_crit[idx]
    else:
        ad_normal = False
    sw_normal = sw_p > 0.05
    dp_normal = dp_p > 0.05

    normalidade_residuos = "✅ Os resíduos podem ser considerados normais." if ad_normal or sw_normal or dp_normal else "⚠ Os resíduos não seguem distribuição normal."

    # Recomendação
    recomendacao = ""
    if ad_normal and sw_normal and dp_normal:
        recomendacao = "⚠ Observação: Como os resíduos apresentaram indícios de normalidade, recomenda-se considerar o uso do teste paramétrico ANOVA Repeated Measures, caso os pressupostos sejam atendidos."

    # Gráfico
    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(grupos, labels=labels, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    medianas = [np.median(g) for g in grupos]
    ax.plot(range(1, len(medianas) + 1), medianas, color='blue', marker='o', linestyle='-', label='Medianas')

    ax.set_title("Friedman Pareado - Boxplot por Grupo", fontsize=10)
    ax.set_xlabel("Grupo")
    ax.set_ylabel("Valores")
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Reporte padronizado
    texto = f"""
📊 **Análise – Friedman Pareado**

🔹 **Hipóteses:**
- H₀: As medianas dos grupos são iguais
- H₁: Pelo menos uma mediana de grupo é diferente

🔎 **Estatísticas Friedman:**
- Estatística = {stat:.4f}
- p-valor = {p_valor:.4f}
- Nível de confiança = {confidence:.1f}%

🔎 **Normalidade dos resíduos (Anderson-Darling, Shapiro-Wilk, D’Agostino-Pearson):**
- {normalidade_residuos}

{recomendacao}

🔎 **Conclusão:**
{"✅ Com {:.1f}% de confiança, podemos rejeitar a hipótese conservadora. Logo, há diferenças estatisticamente significativas entre as medianas dos grupos.".format(confidence) if p_valor < alpha else "⚠ Com {:.1f}% de confiança, não podemos rejeitar a hipótese conservadora. Logo, não há diferenças estatisticamente significativas entre as medianas dos grupos.".format(confidence)}
"""


    return texto.strip(), grafico_base64

def analise_1_intervalo_interquartilico(df: pd.DataFrame, coluna_y, field_conf=None):
    if not coluna_y:
        return "❌ O intervalo interquartílico requer exatamente 1 coluna Y.", None

    if coluna_y not in df.columns:
        return f"❌ A coluna {coluna_y} não foi encontrada no arquivo.", None

    y = df[coluna_y].dropna()

    if len(y) < 5:
        return "❌ O teste requer ao menos 5 valores não nulos.", None

    mediana = np.median(y)
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    minimo = np.min(y)
    maximo = np.max(y)
    n = len(y)

    ad = stats.anderson(y)
    sw_stat, sw_p = stats.shapiro(y)
    dp_stat, dp_p = stats.normaltest(y)

    ad_crit = ad.critical_values
    ad_sig = list(ad.significance_level)
    if 5 in ad_sig:
        idx = ad_sig.index(5)
        ad_normal = ad.statistic < ad_crit[idx]
    else:
        ad_normal = False
    sw_normal = sw_p > 0.05
    dp_normal = dp_p > 0.05

    recomendacao = ""
    if ad_normal or sw_normal or dp_normal:
        recomendacao = "⚠ Os dados podem ser normais. Considere também o cálculo do intervalo de confiança da média."

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 3))

    # Boxplot tradicional
    bplot = ax.boxplot(y, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))

    # Linha azul da mediana apenas dentro do boxplot
    mediana_x = mediana
    mediana_ymin = 0.95
    mediana_ymax = 1.05
    ax.vlines(mediana_x, mediana_ymin, mediana_ymax, color='blue', linestyle='-', linewidth=1)

    # Label “Mediana” acima do boxplot
    ax.text(mediana, 1.08, f'Mediana: {mediana:.2f}', color='blue', ha='center', fontsize=8)

    # Linha preta do IQR abaixo do boxplot, mais próxima
    ax.hlines(0.85, q1, q3, color='black', lw=4)
    # Valor do IQR logo abaixo da linha preta
    ax.text((q1+q3)/2, 0.80, f'IQR: {iqr:.2f}', ha='center', fontsize=8)

    ax.set_title("1 Intervalo Interquartílico - Boxplot", fontsize=10)
    ax.set_xlabel(coluna_y)
    ax.set_yticks([])

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
📊 **Análise – Intervalo Interquartílico**

🔎 **Estatísticas Descritivas:**
- Mediana: {mediana:.4f}
- Q1 (25%): {q1:.4f}
- Q3 (75%): {q3:.4f}
- Intervalo Interquartílico (IQR): {iqr:.4f}
- Mínimo: {minimo:.4f}
- Máximo: {maximo:.4f}
- N: {n}

🔎 **Testes de Normalidade (Anderson-Darling, Shapiro-Wilk, D’Agostino-Pearson):**
- {'✅ Os dados podem ser considerados normais.' if ad_normal or sw_normal or dp_normal else '⚠ Os dados não seguem distribuição normal.'}

🔎 **Conclusão:**
O IQR indica que os 50% centrais dos dados estão distribuídos em um intervalo de {iqr:.2f} unidades. {recomendacao}
"""

    return texto.strip(), grafico_base64









def analise_2_variancas(df: pd.DataFrame, lista_y: list, field_conf=None):
    if len(lista_y) != 2:
        return "❌ O teste 2 Variâncias requer exatamente 2 colunas Y.", None

    col1, col2 = lista_y
    dados1 = df[col1].dropna()
    dados2 = df[col2].dropna()

    if len(dados1) < 5 or len(dados2) < 5:
        return "❌ O teste requer ao menos 5 valores não nulos em cada grupo.", None

    try:
        confidence = float(field_conf) if field_conf else 95.0
        if confidence <= 1:
            confidence *= 100
    except (ValueError, TypeError):
        confidence = 95.0

    alpha = 1 - (confidence / 100)

    var1 = np.var(dados1, ddof=1)
    var2 = np.var(dados2, ddof=1)

    f_stat = var1 / var2 if var1 > var2 else var2 / var1
    dfn = len(dados1) - 1
    dfd = len(dados2) - 1
    p_valor = 2 * min(stats.f.cdf(f_stat, dfn, dfd), 1 - stats.f.cdf(f_stat, dfn, dfd))

    def normalidade(dados):
        ad = stats.anderson(dados)
        sw_stat, sw_p = stats.shapiro(dados)
        dp_stat, dp_p = stats.normaltest(dados)
        ad_crit = ad.critical_values
        ad_sig = list(ad.significance_level)
        if 5 in ad_sig:
            idx = ad_sig.index(5)
            ad_normal = ad.statistic < ad_crit[idx]
        else:
            ad_normal = False
        sw_normal = sw_p > 0.05
        dp_normal = dp_p > 0.05
        algum_normal = ad_normal or sw_normal or dp_normal
        return algum_normal

    normal1 = normalidade(dados1)
    normal2 = normalidade(dados2)

    recomendacao = ""
    if not (normal1 and normal2):
        recomendacao = "⚠️ Como pelo menos um dos conjuntos de dados não é normal, recomenda-se utilizar o teste de Levene ou Brown-Forsythe para confirmação."

    texto = f"""
📊 **Análise** – Teste F para Igualdade de Variâncias

🔹 Hipóteses:
- H₀: As variâncias de {col1} e {col2} são iguais
- H₁: As variâncias de {col1} e {col2} são diferentes

🔎 **Estatísticas Descritivas**:

{col1}:
Variância = {var1:.2f}

{col2}:
Variância = {var2:.2f}

🔎 **Teste F para Igualdade de Variâncias**:
Estatística F = {f_stat:.2f}
p-valor = {p_valor:.2f}

🔎 **Testes de Normalidade**:
{col1}: {"✅ Os dados parecem ser normais." if normal1 else "❌ Os dados não são normais."}
{col2}: {"✅ Os dados parecem ser normais." if normal2 else "❌ Os dados não são normais."}

🔎 **Conclusão**:
{"Com " + str(int(confidence)) + "% de confiança, rejeitamos a hipótese conservadora. Logo, há diferença estatisticamente significativa entre as variâncias." if p_valor < alpha else "Com " + str(int(confidence)) + "% de confiança, não rejeitamos H0. Não há diferença significativa entre as variâncias."}

{recomendacao}
"""

    return texto.strip(), None



def analise_2_variancas_brown_forsythe(df: pd.DataFrame, lista_y, field_conf=None):
    if len(lista_y) != 2:
        return "❌ O teste 2 Variâncias Brown-Forsythe requer exatamente 2 colunas Y.", None

    col1, col2 = lista_y
    dados1 = df[col1].dropna()
    dados2 = df[col2].dropna()

    if len(dados1) < 5 or len(dados2) < 5:
        return "❌ O teste requer ao menos 5 valores não nulos em cada grupo.", None

    try:
        confidence = float(field_conf) if field_conf else 95.0
        if confidence <= 1:
            confidence *= 100
    except (ValueError, TypeError):
        confidence = 95.0

    alpha = 1 - (confidence / 100)

    var1 = np.var(dados1, ddof=1)
    var2 = np.var(dados2, ddof=1)

    stat, p_valor = stats.levene(dados1, dados2, center='median')

    def normalidade(dados):
        ad = stats.anderson(dados)
        sw_stat, sw_p = stats.shapiro(dados)
        dp_stat, dp_p = stats.normaltest(dados)
        ad_crit = ad.critical_values
        ad_sig = list(ad.significance_level)
        if 5 in ad_sig:
            idx = ad_sig.index(5)
            ad_normal = ad.statistic < ad_crit[idx]
        else:
            ad_normal = False
        sw_normal = sw_p > 0.05
        dp_normal = dp_p > 0.05
        algum_normal = ad_normal or sw_normal or dp_normal
        return algum_normal

    normal1 = normalidade(dados1)
    normal2 = normalidade(dados2)

    recomendacao = ""
    if normal1 and normal2:
        recomendacao = "Os dados são normais. Recomenda-se utilizar o teste paramétrico equivalente para maior precisão."

    # mantém o gráfico exatamente como estava
    aplicar_estilo_minitab()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].boxplot([dados1, dados2], labels=[col1, col2], patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
    axes[0].set_title("Boxplot por Grupo")
    axes[0].set_ylabel("Valores")

    axes[1].bar([col1, col2], [var1, var2], color=['skyblue', 'lightgreen'])
    axes[1].set_title("Comparação das Variâncias")
    axes[1].set_ylabel("Variância")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
📊 **Análise – Teste Brown-Forsythe para Igualdade de Variâncias**

🔹 **Hipóteses:**
- H₀: As variâncias de {col1} e {col2} são iguais
- H₁: As variâncias de {col1} e {col2} são diferentes

🔎 **Estatísticas Descritivas:**

{col1}:
Variância = {var1:.2f}

{col2}:
Variância = {var2:.2f}

🔎 **Testes de Normalidade:**
{col1}: {"✅ Os dados parecem ser normais." if normal1 else "❌ Os dados não são normais."}
{col2}: {"✅ Os dados parecem ser normais." if normal2 else "❌ Os dados não são normais."}

🔎 **Teste Brown-Forsythe:**
Estatística = {stat:.2f}
p-valor = {p_valor:.2f}

🔎 **Conclusão:**
{"Com " + str(int(confidence)) + "% de confiança, rejeitamos a hipótese conservadora. Logo, há diferença estatisticamente significativa entre as variâncias." if p_valor < alpha else "Com " + str(int(confidence)) + "% de confiança, não rejeitamos H0. Não há diferença significativa entre as variâncias."}

{f"⚠️ {recomendacao}" if recomendacao else ""}
"""

    return texto.strip(), grafico_base64


def analise_bartlett(df: pd.DataFrame, lista_y: list, subgrupo=None, field_conf=None):
    grupos = []
    variancias = {}
    normalidades = {}

    if subgrupo and subgrupo in df.columns and len(lista_y) == 1:
        y_col = lista_y[0]
        df_valid = df[[y_col, subgrupo]].dropna()
        if df_valid[subgrupo].nunique() < 2:
            return "❌ O Bartlett requer pelo menos 2 grupos distintos na coluna Subgrupo.", None
        for nome, g in df_valid.groupby(subgrupo)[y_col]:
            if len(g) < 5:
                return f"❌ O grupo {nome} requer ao menos 5 valores não nulos.", None
            grupos.append(g)
            variancias[str(nome)] = np.var(g, ddof=1)
            # normalidade
            ad = stats.anderson(g)
            sw_stat, sw_p = stats.shapiro(g)
            dp_stat, dp_p = stats.normaltest(g)
            ad_crit = ad.critical_values
            ad_sig = list(ad.significance_level)
            if 5 in ad_sig:
                idx = ad_sig.index(5)
                ad_normal = ad.statistic < ad_crit[idx]
            else:
                ad_normal = False
            sw_normal = sw_p > 0.05
            dp_normal = dp_p > 0.05
            algum_normal = ad_normal or sw_normal or dp_normal
            normalidades[str(nome)] = algum_normal
        labels = list(df_valid[subgrupo].unique())
    else:
        if len(lista_y) < 2:
            return "❌ O Bartlett requer pelo menos 2 colunas Y (grupos).", None
        for col in lista_y:
            if col not in df.columns:
                return f"❌ A coluna {col} não foi encontrada no arquivo.", None
            dados = df[col].dropna()
            if len(dados) < 5:
                return f"❌ O grupo {col} requer ao menos 5 valores não nulos.", None
            grupos.append(dados)
            variancias[col] = np.var(dados, ddof=1)
            # normalidade
            ad = stats.anderson(dados)
            sw_stat, sw_p = stats.shapiro(dados)
            dp_stat, dp_p = stats.normaltest(dados)
            ad_crit = ad.critical_values
            ad_sig = list(ad.significance_level)
            if 5 in ad_sig:
                idx = ad_sig.index(5)
                ad_normal = ad.statistic < ad_crit[idx]
            else:
                ad_normal = False
            sw_normal = sw_p > 0.05
            dp_normal = dp_p > 0.05
            algum_normal = ad_normal or sw_normal or dp_normal
            normalidades[col] = algum_normal
        labels = lista_y

    try:
        confidence = float(field_conf) if field_conf else 95.0
        if confidence <= 1:
            confidence *= 100
    except (ValueError, TypeError):
        confidence = 95.0

    alpha = 1 - (confidence / 100)

    stat, p_valor = stats.bartlett(*grupos)

    # recomendação se todos normais
    if all(normalidades.values()):
        recomendacao = "⚠️ Todos os grupos parecem ser normais. Recomenda-se utilizar o teste paramétrico equivalente para maior precisão."
    else:
        recomendacao = ""

    aplicar_estilo_minitab()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].boxplot(grupos, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
    axes[0].set_title("Boxplot por Grupo")
    axes[0].set_ylabel("Valores")

    axes[1].bar(labels, [variancias[label] for label in labels], color='skyblue')
    axes[1].set_title("Comparação das Variâncias")
    axes[1].set_ylabel("Variância")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    variancia_texto = "\n".join([f"{label}: Variância = {variancias[label]:.2f}, Normalidade = {'✅' if normalidades[label] else '❌'}" for label in labels])

    texto = f"""
📊 **Análise – Teste de Bartlett para Igualdade de Variâncias**

🔹 **Hipóteses:**
- H₀: As variâncias dos grupos são iguais
- H₁: As variâncias dos grupos são diferentes

🔎 **Estatísticas Descritivas e Normalidade:**
{variancia_texto}

🔎 **Teste de Bartlett:**
Estatística = {stat:.2f}
p-valor = {p_valor:.2f}

🔎 **Conclusão:**
{"Com " + str(int(confidence)) + "% de confiança, rejeitamos a hipótese conservadora. Logo, há diferença estatisticamente significativa entre as variâncias dos grupos." if p_valor < alpha else "Com " + str(int(confidence)) + "% de confiança, não rejeitamos H0. Não há diferença significativa entre as variâncias dos grupos."}

{recomendacao}
"""

    return texto.strip(), grafico_base64



def analise_brown_forsythe(df: pd.DataFrame, lista_y: list, subgrupo=None, field_conf=None):
    grupos = []
    variancias = {}
    normalidades = {}

    if subgrupo and subgrupo in df.columns and len(lista_y) == 1:
        y_col = lista_y[0]
        df_valid = df[[y_col, subgrupo]].dropna()
        if df_valid[subgrupo].nunique() < 2:
            return "❌ O Brown-Forsythe requer pelo menos 2 grupos distintos na coluna Subgrupo.", None
        for nome, g in df_valid.groupby(subgrupo)[y_col]:
            if len(g) < 5:
                return f"❌ O grupo {nome} requer ao menos 5 valores não nulos.", None
            grupos.append(g)
            variancias[str(nome)] = np.var(g, ddof=1)
            # normalidade
            ad = stats.anderson(g)
            sw_stat, sw_p = stats.shapiro(g)
            dp_stat, dp_p = stats.normaltest(g)
            ad_crit = ad.critical_values
            ad_sig = list(ad.significance_level)
            if 5 in ad_sig:
                idx = ad_sig.index(5)
                ad_normal = ad.statistic < ad_crit[idx]
            else:
                ad_normal = False
            sw_normal = sw_p > 0.05
            dp_normal = dp_p > 0.05
            algum_normal = ad_normal or sw_normal or dp_normal
            normalidades[str(nome)] = algum_normal
        labels = list(df_valid[subgrupo].unique())
    else:
        if len(lista_y) < 2:
            return "❌ O Brown-Forsythe requer pelo menos 2 colunas Y (grupos).", None
        for col in lista_y:
            if col not in df.columns:
                return f"❌ A coluna {col} não foi encontrada no arquivo.", None
            dados = df[col].dropna()
            if len(dados) < 5:
                return f"❌ O grupo {col} requer ao menos 5 valores não nulos.", None
            grupos.append(dados)
            variancias[col] = np.var(dados, ddof=1)
            # normalidade
            ad = stats.anderson(dados)
            sw_stat, sw_p = stats.shapiro(dados)
            dp_stat, dp_p = stats.normaltest(dados)
            ad_crit = ad.critical_values
            ad_sig = list(ad.significance_level)
            if 5 in ad_sig:
                idx = ad_sig.index(5)
                ad_normal = ad.statistic < ad_crit[idx]
            else:
                ad_normal = False
            sw_normal = sw_p > 0.05
            dp_normal = dp_p > 0.05
            algum_normal = ad_normal or sw_normal or dp_normal
            normalidades[col] = algum_normal
        labels = lista_y

    stat, p_valor = stats.levene(*grupos, center='median')

    # recomendação se todos normais
    if all(normalidades.values()):
        recomendacao = "⚠️ Todos os grupos parecem ser normais. Recomenda-se utilizar o teste paramétrico equivalente para maior precisão."
    else:
        recomendacao = ""

    aplicar_estilo_minitab()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].boxplot(grupos, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
    axes[0].set_title("Boxplot por Grupo")
    axes[0].set_ylabel("Valores")

    axes[1].bar(labels, [variancias[label] for label in labels], color='skyblue')
    axes[1].set_title("Comparação das Variâncias")
    axes[1].set_ylabel("Variância")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    variancia_texto = "\n".join([f"{label}: Variância = {variancias[label]:.2f}, Normalidade = {'✅' if normalidades[label] else '❌'}" for label in labels])

    texto = f"""
📊 **Análise – Teste Brown-Forsythe para Igualdade de Variâncias**

🔹 **Hipóteses:**
- H₀: As variâncias dos grupos são iguais
- H₁: As variâncias dos grupos são diferentes

🔎 **Estatísticas Descritivas e Normalidade:**
{variancia_texto}

🔎 **Teste Brown-Forsythe:**
Estatística = {stat:.2f}
p-valor = {p_valor:.2f}

🔎 **Conclusão:**
{"Com 95% de confiança, rejeitamos a hipótese conservadora. Logo, há diferença estatisticamente significativa entre as variâncias dos grupos." if p_valor < 0.05 else "Com 95% de confiança, não rejeitamos H0. Não há diferença significativa entre as variâncias dos grupos."}

{recomendacao}
"""

    return texto.strip(), grafico_base64



def analise_1_intervalo_confianca_variancia(df: pd.DataFrame, coluna_y, field_conf=None):
    if not coluna_y:
        return "❌ O intervalo de confiança da variância requer exatamente 1 coluna Y.", None

    if coluna_y not in df.columns:
        return f"❌ A coluna {coluna_y} não foi encontrada no arquivo.", None

    y = df[coluna_y].dropna()

    if len(y) < 5:
        return "❌ O teste requer ao menos 5 valores não nulos.", None

    try:
        nivel_conf = float(field_conf) if field_conf else 95.0
        if nivel_conf <= 1:
            nivel_conf *= 100
        if not (50 <= nivel_conf < 100):
            return "❌ O nível de confiança deve ser entre 50 e 99.9.", None
    except:
        return "❌ Valor do nível de confiança inválido. Informe um número (ex.: 95).", None

    alpha = 1 - (nivel_conf / 100)
    n = len(y)
    s2 = np.var(y, ddof=1)

    chi2_lower = stats.chi2.ppf(alpha / 2, df=n-1)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=n-1)

    ic_lower = (n - 1) * s2 / chi2_upper
    ic_upper = (n - 1) * s2 / chi2_lower

    # normalidade
    ad = stats.anderson(y)
    sw_stat, sw_p = stats.shapiro(y)
    dp_stat, dp_p = stats.normaltest(y)

    ad_crit = ad.critical_values
    ad_sig = list(ad.significance_level)
    if 5 in ad_sig:
        idx = ad_sig.index(5)
        ad_normal = ad.statistic < ad_crit[idx]
    else:
        ad_normal = False
    sw_normal = sw_p > 0.05
    dp_normal = dp_p > 0.05
    algum_normal = ad_normal or sw_normal or dp_normal

    recomendacao = ""
    if not algum_normal:
        recomendacao = "⚠️ Os dados não são normais. O intervalo de confiança da variância pode não ser confiável."

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(0, s2, color='skyblue', width=0.4, label='Variância amostra')
    ax.errorbar(0, s2, yerr=[[s2 - ic_lower], [ic_upper - s2]], fmt='o', color='black', capsize=5, label=f'IC {nivel_conf:.1f}%')
    ax.set_xticks([0])
    ax.set_xticklabels([coluna_y])
    ax.set_ylabel("Variância")
    ax.set_title(f"Intervalo de Confiança da Variância ({nivel_conf:.1f}%)")
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
📊 **Análise – Intervalo de Confiança da Variância**

🔹 **Descrição:**
Coluna = {coluna_y}
N = {n}
Variância amostral = {s2:.2f}

🔎 **Intervalo de Confiança ({nivel_conf:.1f}%):**
[{ic_lower:.2f}, {ic_upper:.2f}]

🔎 **Testes de Normalidade:**
{coluna_y}: {"✅ Os dados parecem ser normais." if algum_normal else "❌ Os dados não são normais."}

🔎 **Conclusão:**
O intervalo de confiança da variância foi calculado assumindo normalidade dos dados.

{recomendacao}
"""

    return texto.strip(), grafico_base64


def analise_1_proporcao(df: pd.DataFrame, coluna_x, field_conf=None):
    if not coluna_x:
        return "❌ O teste 1 Proporção requer exatamente 1 coluna X.", None

    if coluna_x not in df.columns:
        return f"❌ A coluna {coluna_x} não foi encontrada no arquivo.", None

    x = df[coluna_x].dropna()

    if len(x) < 5:
        return "❌ O teste requer ao menos 5 valores não nulos.", None

    try:
        p0 = float(field_conf) if field_conf else 0.5
        if not (0 < p0 < 1):
            return "❌ A proporção de referência deve estar entre 0 e 1 (ex.: 0.5).", None
    except:
        return "❌ Valor de referência inválido. Informe um número como 0.5 no campo Field.", None

    n = len(x)
    nivel_conf = 95.0
    alpha = 1 - (nivel_conf / 100)

    categorias = x.value_counts()
    resultados = []

    # identifica a categoria com maior proporção
    categoria_top = categorias.idxmax()
    sucesso_top = categorias.max()
    p_hat_top = sucesso_top / n

    z = stats.norm.ppf(1 - alpha / 2)  # define z fora do loop para consistência

    for categoria, sucesso in categorias.items():
        p_hat = sucesso / n

        # intervalo de confiança
        se = np.sqrt(p_hat * (1 - p_hat) / n)
        ic_lower = max(0, p_hat - z * se)
        ic_upper = min(1, p_hat + z * se)

        # teste de proporção
        se_h0 = np.sqrt(p0 * (1 - p0) / n)
        z_stat = (p_hat - p0) / se_h0
        p_valor = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        if p_valor < alpha:
            conclusao = f"✅ Com {nivel_conf:.1f}% de confiança, rejeitamos H0. Proporção observada ({p_hat:.2f}) é estatisticamente diferente da referência ({p0:.2f})."
        else:
            conclusao = f"⚠️ Com {nivel_conf:.1f}% de confiança, não rejeitamos H0. Proporção observada ({p_hat:.2f}) não difere significativamente da referência ({p0:.2f})."

        resultados.append(f"""
🔹 **Categoria: {categoria}**
- N = {n}
- Sucessos = {sucesso}
- Proporção amostral = {p_hat:.2f}
- Intervalo {nivel_conf:.1f}% = [{ic_lower:.2f}, {ic_upper:.2f}]
- Estatística Z = {z_stat:.2f}
- p-valor = {p_valor:.2f}

🔎 **Conclusão:**
{conclusao}
""")

    # gera gráfico apenas para a categoria com maior proporção
    se_top = np.sqrt(p_hat_top * (1 - p_hat_top) / n)
    ic_lower_top = max(0, p_hat_top - z * se_top)
    ic_upper_top = min(1, p_hat_top + z * se_top)

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([0], [p_hat_top], color='skyblue', width=0.4, label=f'Proporção {categoria_top}')
    ax.errorbar([0], [p_hat_top], yerr=[[p_hat_top - ic_lower_top], [ic_upper_top - p_hat_top]], fmt='o', color='black', capsize=5, label=f'IC {nivel_conf:.1f}%')
    ax.axhline(p0, color='red', linestyle='--', label=f'Referência: {p0:.2f}')
    ax.set_xticks([0])
    ax.set_xticklabels([categoria_top])
    ax.set_ylim(0, max(ic_upper_top * 1.2, p_hat_top * 1.2, p0 * 1.2))
    ax.set_ylabel('Proporção')
    ax.legend()
    ax.set_title(f'Proporção {categoria_top} com IC {nivel_conf:.1f}%')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
📊 **Análise – Teste de 1 Proporção**
{''.join(resultados)}
"""

    return texto.strip(), grafico_base64





  def analise_2_proporcoes(df: pd.DataFrame, coluna_x, coluna_y=None, field_conf=None):
    if not coluna_x:
        return "❌ O teste 2 Proporções requer pelo menos a coluna X.", None

    if coluna_x not in df.columns:
        return f"❌ A coluna {coluna_x} não foi encontrada no arquivo.", None

    x = df[coluna_x].dropna()

    try:
        nivel_conf = float(field_conf) if field_conf else 95.0
        if nivel_conf <= 1:
            nivel_conf *= 100
        if not (50 <= nivel_conf < 100):
            return "❌ O nível de confiança deve ser entre 50 e 99.9.", None
    except:
        return "❌ Valor do nível de confiança inválido. Informe um número como 95 no campo Field.", None

    alpha = 1 - (nivel_conf / 100)
    z = stats.norm.ppf(1 - alpha / 2)

    if coluna_y and coluna_y in df.columns:
        # fluxo tabela: coluna_x = grupo, coluna_y = sucesso/fracasso
        df_valid = df[[coluna_x, coluna_y]].dropna()
        grupos = df_valid[coluna_x].unique()
        if len(grupos) != 2:
            return "❌ O teste 2 Proporções com tabela requer exatamente 2 grupos na coluna X.", None

        prop = {}
        n = {}
        for g in grupos:
            sub = df_valid[df_valid[coluna_x] == g][coluna_y]
            n[g] = len(sub)
            if n[g] < 5:
                return f"❌ O grupo {g} requer ao menos 5 valores não nulos.", None
            prop[g] = np.mean(sub == sub.unique()[0])  # considera sucesso como a primeira categoria encontrada

        g1, g2 = grupos
        p1, p2 = prop[g1], prop[g2]
        n1, n2 = n[g1], n[g2]

    else:
        # fluxo sem coluna_y: coluna_x possui duas categorias
        categorias = x.value_counts()
        if len(categorias) != 2:
            return "❌ O teste 2 Proporções requer exatamente 2 categorias na coluna X.", None

        g1, g2 = categorias.index
        n1, n2 = categorias[g1], categorias[g2]
        total = n1 + n2
        p1, p2 = n1 / total, n2 / total

    # cálculo estatístico
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z_stat = (p1 - p2) / se
    p_valor = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # IC da diferença
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    ic_lower = diff - z * se_diff
    ic_upper = diff + z * se_diff

    # gráfico
    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([0, 1], [p1, p2], color=['skyblue', 'lightgreen'], width=0.4)
    ax.errorbar([0, 1], [p1, p2],
                yerr=[[p1 - max(0, p1 - z * np.sqrt(p1 * (1 - p1) / n1)),
                       p2 - max(0, p2 - z * np.sqrt(p2 * (1 - p2) / n2))],
                      [min(1, p1 + z * np.sqrt(p1 * (1 - p1) / n1)) - p1,
                       min(1, p2 + z * np.sqrt(p2 * (1 - p2) / n2)) - p2]],
                fmt='o', color='black', capsize=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g1, g2])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proporção")
    ax.set_title(f"2 Proporções - IC {nivel_conf:.1f}% e Teste H0: p1 = p2")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
📊 **Análise – Teste de 2 Proporções**

🔹 **Hipóteses:**
- H₀: As proporções de {g1} e {g2} são iguais
- H₁: As proporções de {g1} e {g2} são diferentes

🔎 **Estatísticas Descritivas:**
- {g1}: proporção = {p1:.2f}, N = {n1}
- {g2}: proporção = {p2:.2f}, N = {n2}

🔎 **Teste de 2 Proporções:**
- Diferença = {diff:.2f}
- Intervalo {nivel_conf:.1f}% = [{ic_lower:.2f}, {ic_upper:.2f}]
- Estatística Z = {z_stat:.2f}
- p-valor = {p_valor:.2f}

🔎 **Conclusão:**
{"✅ Com {nivel_conf:.1f}% de confiança, rejeitamos H0. As proporções são estatisticamente diferentes." if p_valor < alpha else f"⚠️ Com {nivel_conf:.1f}% de confiança, não rejeitamos H0. Não há diferença significativa entre as proporções."}
"""

    return texto.strip(), grafico_base64


def analise_k_proporcoes(df: pd.DataFrame, lista_y, field_conf=None):
    if len(lista_y) < 2:
        return "❌ O teste K Proporções requer pelo menos 2 colunas Y (grupos).", None

    sucessos = []
    totais = []
    proporcoes = {}
    for col in lista_y:
        if col not in df.columns:
            return f"❌ A coluna {col} não foi encontrada no arquivo.", None
        dados = df[col].dropna()
        if len(dados) < 5:
            return f"❌ O grupo {col} requer ao menos 5 valores não nulos.", None
        sucesso = np.sum(dados)
        total = len(dados)
        sucessos.append(sucesso)
        totais.append(total)
        proporcoes[col] = sucesso / total

    table = np.array([sucessos, [t - s for s, t in zip(sucessos, totais)]])
    stat, p_valor, dof, expected = stats.chi2_contingency(table.T)

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(lista_y)), [proporcoes[c] for c in lista_y], color='skyblue')
    ax.set_xticks(range(len(lista_y)))
    ax.set_xticklabels(lista_y)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proporção")
    ax.set_title("K Proporções - Proporção por Grupo")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    proporcao_texto = "\n".join([f"- {col}: {proporcoes[col]:.4f}" for col in lista_y])

    texto = f"""
**K Proporções - Teste de Homogeneidade**
{proporcao_texto}
- Estatística Qui-quadrado: {stat:.4f}
- p-valor: {p_valor:.4f}
- Graus de liberdade: {dof}

**Conclusão**
{"✅ Rejeitamos H0: as proporções não são todas iguais." if p_valor < 0.05 else "⚠ Não rejeitamos H0: não há diferença significativa entre as proporções dos grupos."}
"""

    return texto.strip(), grafico_base64

def analise_associacao(df: pd.DataFrame, coluna_y, lista_x, subgrupo):
    if not coluna_y or not lista_x or len(lista_x) != 1:
        return "❌ A análise de associação requer exatamente 1 coluna Y e 1 coluna X.", None

    y_col = coluna_y
    x_col = lista_x[0]

    if y_col not in df.columns or x_col not in df.columns:
        return "❌ As colunas informadas não foram encontradas no arquivo.", None

    df_valid = df[[y_col, x_col]].dropna()
    if df_valid.empty:
        return "❌ Não há dados suficientes após remoção de valores nulos.", None

    table = pd.crosstab(df_valid[y_col], df_valid[x_col])
    y_labels = table.index.tolist()
    x_labels = table.columns.tolist()
    table_values = table.to_numpy()

    stat, p_valor, dof, expected = stats.chi2_contingency(table_values)

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    proporcoes = table_values / table_values.sum()
    im = ax.imshow(proporcoes, cmap='Blues', aspect='auto')

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title("Mapa de calor das proporções")

    for i in range(proporcoes.shape[0]):
        for j in range(proporcoes.shape[1]):
            ax.text(j, i, f"{proporcoes[i, j]:.2f}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
**Análise de Associação (Qui-quadrado)**
- Estatística Qui-quadrado: {stat:.4f}
- p-valor: {p_valor:.4f}
- Graus de liberdade: {dof}

**Conclusão**
{"✅ Rejeitamos H0: existe associação entre as variáveis." if p_valor < 0.05 else "⚠ Não rejeitamos H0: não há evidência de associação entre as variáveis."}
"""

    return texto.strip(), grafico_base64


ANALISES = {
    "1 Sample T": analise_1_sample_t,
    "2 Sample T": analise_2_sample_t,
    "2 Paired Test": analise_paired_t,
    "One way ANOVA": analise_one_way_anova,
    "1 Wilcoxon": analise_1_wilcoxon,
    "2 Mann-Whitney": analise_2_mann_whitney,
    "2 Wilcoxon Paired": analise_2_wilcoxon_paired,
    "Kruskal-Wallis": analise_kruskal_wallis,
    "Friedman Pareado": analise_friedman_pareado,
    "1 Intervalo de Confianca": analise_1_intervalo_confianca,
    "1 Intervalo Interquartilico": analise_1_intervalo_interquartilico,
    "2 Varianças": analise_2_variancas,
    "2 Variancas Brown-Forsythe": analise_2_variancas_brown_forsythe,
    "Bartlett": analise_bartlett,
    "Brown-Forsythe": analise_brown_forsythe,
    "1 Intervalo de Confianca Variancia": analise_1_intervalo_confianca_variancia,
    "1 Proporcao": analise_1_proporcao,
    "2 Proporcoes": analise_2_proporcoes,
    "K Proporcoes": analise_k_proporcoes,
    "Qui-quadrado": analise_associacao
  

}


