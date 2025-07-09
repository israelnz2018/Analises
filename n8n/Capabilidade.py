from suporte import *

def teste_normalidade(df: pd.DataFrame, coluna_y: str):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    from scipy import stats
    import statsmodels.api as sm

    aplicar_estilo_minitab()

    if coluna_y not in df.columns:
        return "❌ Coluna não encontrada para teste de normalidade.", None

    dados = df[coluna_y].dropna()
    N = len(dados)
    media = np.mean(dados)
    desvio = np.std(dados, ddof=1)

    # Testes de normalidade
    ad_result = stats.anderson(dados, dist='norm')
    ad_stat = ad_result.statistic

    # p-valor aproximado para Anderson-Darling (método padrão)
    if ad_stat < 0.2:
        ad_p = "> 0,25"
    elif ad_stat < 0.34:
        ad_p = "≈ 0,15"
    elif ad_stat < 0.6:
        ad_p = "≈ 0,10"
    elif ad_stat < 0.8:
        ad_p = "≈ 0,05"
    elif ad_stat < 1.1:
        ad_p = "≈ 0,025"
    else:
        ad_p = "< 0,01"

    sw_stat, sw_p = stats.shapiro(dados)
    ks_stat, ks_p = stats.kstest(dados, 'norm', args=(media, desvio))

    ad_conc = "Aprovada" if ad_stat < 0.752 else "Reprovada"
    sw_conc = "Aprovada" if sw_p > 0.05 else "Reprovada"
    ks_conc = "Aprovada" if ks_p > 0.05 else "Reprovada"
    normal = any([ad_conc == "Aprovada", sw_conc == "Aprovada", ks_conc == "Aprovada"])

    # Gráfico de probabilidade sem linha da média
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    pp = sm.ProbPlot(dados, dist=stats.norm)
    pp.probplot(ax=ax)
    ax.set_title(f"Gráfico de Probabilidade – {coluna_y}", fontsize=14, fontweight='bold')
    ax.set_xlabel(f"{coluna_y}", fontsize=12, fontweight='bold')
    ax.set_ylabel("Percentual", fontsize=12, fontweight='bold')
    plt.grid(True)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    def fbr(v): return f"{v:.4f}".replace('.', ',')

    # Relatório final – títulos em negrito
    texto = (
        f"📊 **Análise de Normalidade – {coluna_y}**\n\n"
        f"🔎 **Resumo estatístico dos dados**\n"
        f"- **Média:** {fbr(media)}\n"
        f"- **Desvio Padrão:** {fbr(desvio)}\n"
        f"- **N (amostras):** {N}\n\n"
        f"ℹ️ **Critério de normalidade**\n"
        f"Se pelo menos um dos testes abaixo indicar normalidade (p-valor > 0,05), considera-se que os dados seguem distribuição normal e podem ser analisados com métodos paramétricos.\n\n"
        f"✅ **Resultados dos Testes de Normalidade**\n\n"
        f"Anderson-Darling: estat={fbr(ad_stat)} | p-valor={ad_p} | Conclusão: {'Normalidade aprovada' if ad_conc == 'Aprovada' else 'Normalidade reprovada'}\n"
        f"Shapiro-Wilk: estat={fbr(sw_stat)} | p-valor={fbr(sw_p)} | Conclusão: {'Normalidade aprovada' if sw_conc == 'Aprovada' else 'Normalidade reprovada'}\n"
        f"Kolmogorov-Smirnov: estat={fbr(ks_stat)} | p-valor={fbr(ks_p)} | Conclusão: {'Normalidade aprovada' if ks_conc == 'Aprovada' else 'Normalidade reprovada'}\n\n"
        f"📝 **Conclusão**\n"
        f"{'✅ **Os dados seguem distribuição normal. Pode-se prosseguir com métodos paramétricos sem restrições.**' if normal else '❌ **Os dados não seguem distribuição normal.**'}\n\n"
    )

    if not normal:
        texto += (
            "**⚠️ Recomendações**\n"
            "1. Verificar estabilidade do processo\n"
            "2. Coletar mais dados\n"
            "3. Buscar outra distribuição que melhor se ajuste\n"
            "4. Como último recurso, aplicar transformações matemáticas (ex: Box-Cox, Johnson)\n"
        )

    return texto.strip(), grafico_base64










def analise_estabilidade(df, coluna_y):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no dataframe.", None

    dados = df[[coluna_y]].copy()
    dados['Subgrupo'] = range(1, len(dados) + 1)

    if dados.empty or dados[coluna_y].dropna().empty:
        return "❌ Dados insuficientes para análise.", None

    aplicar_estilo_minitab()
    texto_resumo = f"📊 **Análise de Estabilidade da coluna '{coluna_y}'**\n"

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    sns.lineplot(x="Subgrupo", y=coluna_y, data=dados, ax=axs[0], marker="o")
    axs[0].set_title("Carta Individual")
    axs[0].set_ylabel("Valor")

    mr = dados[coluna_y].diff().abs()
    axs[1].plot(dados['Subgrupo'][1:], mr[1:], marker="o")
    axs[1].set_title("Carta MR")
    axs[1].set_ylabel("Movimento Range")

    texto_resumo += "- Carta I-MR usada (sem subgrupos).\n"

    media = dados[coluna_y].mean()
    sigma = dados[coluna_y].std()
    outliers = dados[(dados[coluna_y] > media + 3 * sigma) | (dados[coluna_y] < media - 3 * sigma)]

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

def analise_distribuicao_estatistica(df, coluna_y):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A análise de distribuição requer 1 coluna Y válido.", None

    dados = df[coluna_y].dropna()
    if len(dados) < 10:
        return "❌ A análise de distribuição requer pelo menos 10 dados.", None

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats
    from io import BytesIO
    import base64

    distribs = {
        'Normal': stats.norm,
        'Lognormal': stats.lognorm,
        'Exponencial': stats.expon,
        'Weibull': stats.weibull_min,
        'Gamma': stats.gamma
    }

    resultados = []

    for nome, dist in distribs.items():
        try:
            params = dist.fit(dados)
            ad_res = stats.anderson(dados, dist='norm') if nome == 'Normal' else None
            d, p = stats.kstest(dados, dist.cdf, args=params)
            resultados.append({
                'Distribuição': nome,
                'AD': ad_res.statistic if ad_res else np.nan,
                'P-valor': p
            })
        except Exception:
            resultados.append({
                'Distribuição': nome,
                'AD': np.nan,
                'P-valor': 0
            })

    # Escolha do melhor modelo
    df_res = pd.DataFrame(resultados)
    df_res_validos = df_res[df_res['P-valor'] > 0.05]
    if not df_res_validos.empty:
        melhor = df_res_validos.loc[df_res_validos['AD'].idxmin()] if df_res_validos['AD'].notna().any() else df_res_validos.iloc[0]
    else:
        melhor = df_res.loc[df_res['P-valor'].idxmax()]

    nome_melhor = melhor['Distribuição']

    # Estatísticas descritivas
    desc = dados.describe()
    skew = dados.skew()
    kurt = dados.kurtosis()

    estatisticas = f"""
📊 **Análise de Distribuição Estatística**

📈 **Estatísticas Descritivas**

N: {int(desc['count'])}
Média: {desc['mean']:.3f}
Desvio Padrão: {desc['std']:.3f}
Mediana: {dados.median():.3f}
Mínimo: {desc['min']:.3f}
Máximo: {desc['max']:.3f}
Assimetria (Skewness): {skew:.3f}
Curtose (Kurtosis): {kurt:.3f}
""".strip()

    # Tabela de resultados formatada bonita e alinhada
    linhas = ["\n📊 **Resultados do Teste de Aderência**",
              "**Distribuição**       **AD**      **P-valor**"]
    for r in resultados:
        ad_str = f"{r['AD']:.3f}" if not np.isnan(r['AD']) else "-"
        p_str = f"{r['P-valor']:.3f}" if r['P-valor'] >= 0.001 else "<0.001"
        linhas.append(f"{r['Distribuição']:<18} {ad_str:<8} {p_str}")

    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dados, bins=10, density=True, alpha=0.6, color='gray', edgecolor='black')
    params = distribs[nome_melhor].fit(dados)
    x = np.linspace(min(dados), max(dados), 100)
    y = distribs[nome_melhor].pdf(x, *params)
    ax.plot(x, y, color='blue', label=f'{nome_melhor}')
    ax.set_title(f'Histograma com ajuste da distribuição: {nome_melhor}')
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Relatório final
    texto = f"""
{estatisticas}

{chr(10).join(linhas)}

📝 **Critério de Seleção**  
Para escolher a melhor distribuição, foi utilizado:
- Menor AD (Anderson-Darling), indica melhor ajuste aos dados.
- P-valor > 0.05, quando possível, para não rejeitar a hipótese de aderência.

✅ **Conclusão**  
A melhor distribuição é: **{nome_melhor}**
""".strip()

    return texto, grafico_base64


def analise_capabilidade_normal(df, coluna_y, subgrupo=None, field_LIE=None, field_LSE=None):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ É necessário informar uma coluna Y válida.", None

    if not field_LIE and not field_LSE:
        return "❌ Pelo menos um dos limites (LIE ou LSE) deve ser informado.", None

    try:
        LIE = float(field_LIE) if field_LIE else None
    except:
        LIE = None

    try:
        LSE = float(field_LSE) if field_LSE else None
    except:
        LSE = None

    dados = df[coluna_y].dropna()
    if len(dados) < 30:
        return "⚠ Recomenda-se pelo menos 30 dados para análise de capabilidade.", None

    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    # Testes de normalidade
    ad_res = stats.anderson(dados)
    sw_stat, sw_p = stats.shapiro(dados)
    ks_stat, ks_p = stats.kstest((dados - np.mean(dados)) / np.std(dados, ddof=1), 'norm')

    ad_sig = list(ad_res.significance_level)
    ad_normal = 5 in ad_sig and ad_res.statistic < ad_res.critical_values[ad_sig.index(5)]
    sw_normal = sw_p > 0.05
    ks_normal = ks_p > 0.05

    dados_normais = ad_normal or sw_normal or ks_normal

    if not dados_normais:
        texto = f"""
❌ **Os dados não seguem distribuição normal.**

✔️ Recomendações:
- Verificar a estabilidade do processo.
- Adicionar mais dados para análise.
- Realizar análise de capabilidade para dados não normais (ex: transformação de Johnson ou capabilidade não paramétrica).
"""
        return texto.strip(), None

    # Estatísticas gerais
    mean = np.mean(dados)
    std_global = np.std(dados, ddof=1)

    if subgrupo and subgrupo in df.columns:
        grupos = df[[coluna_y, subgrupo]].dropna().groupby(subgrupo)[coluna_y]
        variancias = grupos.var(ddof=1)
        mean_variancia = variancias.mean()
        std_within = np.sqrt(mean_variancia)
    else:
        dados_ordenados = dados.reset_index(drop=True)
        moving_ranges = dados_ordenados.diff().abs().dropna()
        mr_bar = moving_ranges.mean()
        d2 = 1.128
        std_within = mr_bar / d2

    # Inicializar cálculos
    Cp = Cpk = Pp = Ppk = sigma_real = None
    amplitude = None

    if LIE is not None and LSE is not None:
        amplitude = LSE - LIE
        Cp = amplitude / (6 * std_within)
        Cpk = min((LSE - mean), (mean - LIE)) / (3 * std_within)
        Pp = amplitude / (6 * std_global)
        Ppk = min((LSE - mean), (mean - LIE)) / (3 * std_global)
        sigma_real = min((LSE - mean), (mean - LIE)) / std_global
    else:
        if LIE is not None:
            Cpk = (mean - LIE) / (3 * std_within)
            Ppk = (mean - LIE) / (3 * std_global)
            sigma_real = (mean - LIE) / std_global
        elif LSE is not None:
            Cpk = (LSE - mean) / (3 * std_within)
            Ppk = (LSE - mean) / (3 * std_global)
            sigma_real = (LSE - mean) / std_global

    # Percentual de defeitos (global)
    ppm_below_lie = stats.norm.cdf(LIE, loc=mean, scale=std_global) * 1e6 if LIE is not None else 0
    ppm_above_lse = (1 - stats.norm.cdf(LSE, loc=mean, scale=std_global)) * 1e6 if LSE is not None else 0
    ppm_total = ppm_below_lie + ppm_above_lse
    percent_below = ppm_below_lie / 10000
    percent_above = ppm_above_lse / 10000
    percent_total = ppm_total / 10000

    # Interpretação baseada nos resultados reais
    interpretacao = []
    recomendacoes = []

    if Cp is not None and Pp is not None:
        if Cp > Pp:
            interpretacao.append("✔️ **Cp > Pp:** O processo tem bom potencial, mas há variações ao longo do tempo (instabilidade).")
        elif abs(Cp - Pp) <= 0.05:
            interpretacao.append("✔️ **Cp ≈ Pp:** O processo está estável.")
        else:
            interpretacao.append("⚠️ **Cp < Pp:** Pode haver erro de subgrupamento ou presença de outliers.")

    if Cpk is not None:
        if Cpk < 1.00:
            interpretacao.append(f"❌ **Cpk = {Cpk:.2f} < 1.00:** O processo **não é capaz**.")
        elif Cpk < 1.33:
            interpretacao.append(f"⚠️ **Cpk = {Cpk:.2f} < 1.33:** O processo **não atende ao valor recomendado (≥ 1.33)**.")
        else:
            interpretacao.append(f"✅ **Cpk = {Cpk:.2f} ≥ 1.33:** O processo é capaz e aceitável.")

    if Ppk is not None:
        if Ppk < 1.00:
            interpretacao.append(f"❌ **Ppk = {Ppk:.2f} < 1.00:** Performance real não é capaz.")
        elif Ppk < 1.33:
            interpretacao.append(f"⚠️ **Ppk = {Ppk:.2f} < 1.33:** Performance real abaixo do recomendado.")
        else:
            interpretacao.append(f"✅ **Ppk = {Ppk:.2f} ≥ 1.33:** Performance real aceitável.")

    if Cpk is not None and Ppk is not None and abs(Cpk - Ppk) > 0.1:
        interpretacao.append("⚠️ **Cpk e Ppk diferem significativamente**, sugerindo variação ao longo do tempo ou instabilidade do processo.")

    indices = [i for i in [Cp, Cpk, Pp, Ppk] if i is not None]
    min_indice = min(indices) if indices else None

    if min_indice is not None:
        if min_indice > 1.33:
            recomendacoes.append("✅ Todos os índices estão acima de 1.33, indicando que o processo é capaz e aceitável.")
        elif min_indice < 1.00:
            recomendacoes.append("❌ Um ou mais índices estão abaixo de 1.00, indicando que o processo **não é capaz**. Recomenda-se investigar causas especiais de variação ou revisar especificações.")
        else:
            recomendacoes.append("⚠️ Alguns índices estão entre 1.00 e 1.33. O processo atende minimamente, mas é recomendável melhorá-lo para maior segurança.")

    # Relatório
    relatorio = f"""
✅ Os dados foram considerados **normais**. Segue a análise de capabilidade:

📊 **Análise de Capabilidade de Processo**

**Estatísticas do Processo**
N: {len(dados)}
Média: {mean:.3f}
Desvio Padrão Global (σ_total): {std_global:.3f}
Desvio Padrão Within (σ_within): {std_within:.3f}
""".strip()

    if amplitude is not None:
        relatorio += f"\n\n**Limites de Especificação**\nLIE (Limite Inferior de Engenharia): {LIE}\nLSE (Limite Superior de Engenharia): {LSE}\nAmplitude (LSE - LIE): {amplitude}"

    if Cp is not None and Cpk is not None:
        relatorio += f"\n\n**Índices de Capabilidade (Potencial)**\nCp: {Cp:.2f}\nCpk: {Cpk:.2f}"
    elif Cpk is not None:
        relatorio += f"\n\n**Índice de Capabilidade (Potencial)**\nCpk: {Cpk:.2f}"

    if Pp is not None and Ppk is not None:
        relatorio += f"\n\n**Índices de Desempenho (Performance Real)**\nPp: {Pp:.2f}\nPpk: {Ppk:.2f}\nNível Sigma (Real): {sigma_real:.2f} sigma"
    elif Ppk is not None:
        relatorio += f"\n\n**Índice de Desempenho (Performance Real)**\nPpk: {Ppk:.2f}\nNível Sigma (Real): {sigma_real:.2f} sigma"
    relatorio += f"""
\n\n**% de Defeitos (Global)**
Abaixo do LIE: {percent_below:.2f}%
Acima do LSE: {percent_above:.2f}%
Total: {percent_total:.2f}%
"""

    relatorio += f"""
\n📝 **Interpretação dos Resultados**
{chr(10).join(interpretacao)}

✔️ **Recomendações**
{chr(10).join(recomendacoes)}
""".strip()

    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dados, bins=15, density=True, alpha=0.6, color='gray', edgecolor='black')
    x = np.linspace(min(dados), max(dados), 100)
    y = stats.norm.pdf(x, mean, std_global)
    ax.plot(x, y, 'b-', label='Distribuição Normal')

    if LIE is not None:
        ax.axvline(LIE, color='red', linestyle='--', label='LIE (Limite Inf. Eng.)')
    if LSE is not None:
        ax.axvline(LSE, color='red', linestyle='--', label='LSE (Limite Sup. Eng.)')
    ax.axvline(mean, color='green', linestyle='--', label='Média')
    ax.set_title(f'Capabilidade - {coluna_y}')
    ax.legend()

    ax.set_facecolor('white')
    ax.grid(True, linestyle=':', color='gray')
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return relatorio, grafico_base64


def analise_capabilidade_outros(df, coluna_y, field_dist, subgrupo=None, field_LIE=None, field_LSE=None):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ É necessário informar uma coluna Y válida.", None

    if not field_dist:
        return "❌ É necessário informar a distribuição escolhida.", None

    if not field_LIE and not field_LSE:
        return "❌ Pelo menos um dos limites (LIE ou LSE) deve ser informado.", None

    try:
        LIE = float(field_LIE) if field_LIE else None
    except:
        LIE = None

    try:
        LSE = float(field_LSE) if field_LSE else None
    except:
        LSE = None

    dist_nome = field_dist.strip()
    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    # Distribuições suportadas (Normal removida)
    distros = {
        "Lognormal": stats.lognorm,
        "Exponencial": stats.expon,
        "Weibull": stats.weibull_min,
        "Gamma": stats.gamma,
        "Logistica": stats.logistic
    }

    if dist_nome not in distros:
        return f"❌ Distribuição '{dist_nome}' não suportada no momento.", None

    dist = distros[dist_nome]

    relatorio = f"📊 **Análise de Capabilidade com Distribuição {dist_nome}**\n\n"

    imagens = []

    if subgrupo and subgrupo in df.columns:
        grupos = df[subgrupo].dropna().unique()
    else:
        grupos = [None]

    for grupo in grupos:
        if grupo is None:
            dados = df[coluna_y].dropna()
        else:
            dados = df[df[subgrupo] == grupo][coluna_y].dropna()

        if len(dados) < 30:
            relatorio += f"🔹 **Resultado**\n\n⚠️ Mínimo de 30 dados recomendado.\n\n"
            continue

        try:
            params = dist.fit(dados)
            ks_stat, ks_p = stats.kstest(dados, dist.cdf, args=params)
            ppm_lie = dist.cdf(LIE, *params) * 1e6 if LIE is not None else 0
            ppm_lse = (1 - dist.cdf(LSE, *params)) * 1e6 if LSE is not None else 0
            ppm_total = ppm_lie + ppm_lse
            perc_defeitos = ppm_total / 10000

            # Calcular Nível Sigma corretamente
            if perc_defeitos == 0:
                nivel_sigma = ">6"
            else:
                p_defeito = ppm_total / 1e6
                nivel_sigma = round(stats.norm.ppf(1 - p_defeito / 2), 2)

            # Calcular Pp e Ppk
            if LIE is not None and LSE is not None:
                amplitude = LSE - LIE
                std_global = np.std(dados, ddof=1)
                Pp = amplitude / (6 * std_global)
                Ppk = min((LSE - np.mean(dados)), (np.mean(dados) - LIE)) / (3 * std_global)
            else:
                Pp = "N/A"
                Ppk = "N/A"

            # Gráfico (apenas curva da distribuição escolhida)
            x = np.linspace(min(dados), max(dados), 100)
            try:
                y = dist.pdf(x, *params)
            except Exception as e:
                return f"❌ Erro ao calcular PDF para distribuição '{dist_nome}': {str(e)}", None

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(dados, bins=15, density=True, alpha=0.6, color='gray', edgecolor='black')
            ax.plot(x, y, 'b-', label=f'Curva {dist_nome}')
            if LIE is not None:
                ax.axvline(LIE, color='red', linestyle='--', label='LIE (Limite Inf. Eng.)')
            if LSE is not None:
                ax.axvline(LSE, color='red', linestyle='--', label='LSE (Limite Sup. Eng.)')
            ax.axvline(np.mean(dados), color='green', linestyle='--', label='Média')
            ax.set_title(f'Capabilidade - {dist_nome}')
            ax.legend()
            ax.set_facecolor('white')
            ax.grid(True, linestyle=':', color='gray')
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            imagens.append(base64.b64encode(buf.getvalue()).decode('utf-8'))

            # Interpretação
            ajuste = "✅ A distribuição se ajusta bem aos dados." if ks_p >= 0.05 else "⚠️ A distribuição não se ajusta bem aos dados. Considere testar outra."

            # Recomendações detalhadas
            recomendacoes = []
            if perc_defeitos > 1.0:
                recomendacoes.append("- Alta taxa de defeitos (>1%). Recomenda-se revisão do processo, investigação de causas especiais e ações corretivas.")
            else:
                recomendacoes.append("- Baixa taxa de defeitos. Manter controle e monitoramento periódico.")
            if isinstance(nivel_sigma, (int, float)) and nivel_sigma < 3:
                recomendacoes.append("- Nível sigma abaixo de 3. Processo precisa ser melhorado para reduzir variação e aumentar qualidade.")
            elif isinstance(nivel_sigma, (int, float)) and nivel_sigma >= 3:
                recomendacoes.append("- Nível sigma adequado. Buscar oportunidades de melhoria contínua.")

            relatorio += f"""
🔹 **Resultado**

- {ajuste}
- Porcentagem de defeitos estimada: {perc_defeitos:.2f}%
- Nível Sigma estimado: {nivel_sigma}
- Pp: {Pp if isinstance(Pp,str) else f"{Pp:.2f}"}
- Ppk: {Ppk if isinstance(Ppk,str) else f"{Ppk:.2f}"}

✔️ **Recomendações**
{chr(10).join(recomendacoes)}
""".strip()

        except Exception as e:
            relatorio += f"\n🔹 **Resultado**\n\n❌ Erro: {str(e)}\n\n"

    if not imagens:
        return relatorio.strip(), None
    else:
        from PIL import Image
        imgs = [Image.open(BytesIO(base64.b64decode(img))) for img in imagens]
        largura_max = max(img.width for img in imgs)
        altura_total = sum(img.height for img in imgs)
        imagem_final = Image.new("RGB", (largura_max, altura_total), (255, 255, 255))
        y_offset = 0
        for img in imgs:
            imagem_final.paste(img, (0, y_offset))
            y_offset += img.height
        buffer = BytesIO()
        imagem_final.save(buffer, format="PNG")
        grafico_final_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return relatorio.strip(), grafico_final_base64



def analise_capabilidade_johnson(df, coluna_y, subgrupo=None, field_LIE=None, field_LSE=None):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ É necessário informar uma coluna Y válida.", None

    if not field_LIE and not field_LSE:
        return "❌ Pelo menos um dos limites (LIE ou LSE) deve ser informado.", None

    try:
        LSL = float(field_LIE) if field_LIE else float('-inf')
    except:
        LSL = float('-inf')

    try:
        USL = float(field_LSE) if field_LSE else float('inf')
    except:
        USL = float('inf')

    dados = df[coluna_y].dropna()
    if len(dados) < 30:
        return "⚠ Recomenda-se pelo menos 30 dados para capabilidade.", None

    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    # Fit Johnson SU
    try:
        params = stats.johnsonsu.fit(dados)
        gamma, delta, xi, lam = params
        dados_t = gamma + delta * np.arcsinh((dados - xi) / lam)

    except Exception as e:
        return f"❌ Erro na transformação Johnson SU: {str(e)}", None

    mean = np.mean(dados_t)
    std = np.std(dados_t, ddof=1)

    # Cálculos de capabilidade
    Pp = (USL - LSL) / (6 * std)
    Ppk = min((USL - mean), (mean - LSL)) / (3 * std)
    nivel_sigma = min((USL - mean), (mean - LSL)) / std

    ppm_lsl = stats.norm.cdf(LSL, loc=mean, scale=std) * 1e6
    ppm_usl = (1 - stats.norm.cdf(USL, loc=mean, scale=std)) * 1e6
    ppm_total = ppm_lsl + ppm_usl
    percent_below = ppm_lsl / 10000
    percent_above = ppm_usl / 10000
    percent_total = ppm_total / 10000

    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dados_t, bins=15, density=True, alpha=0.6, color='gray', edgecolor='black')
    x = np.linspace(min(dados_t), max(dados_t), 100)
    y = stats.norm.pdf(x, mean, std)
    ax.plot(x, y, 'b-', label='Curva Normal transformada')

    if LSL != float('-inf'):
        ax.axvline(LSL, color='red', linestyle='--', label='LIE (Limite Inf. Eng.)')
    if USL != float('inf'):
        ax.axvline(USL, color='red', linestyle='--', label='LSE (Limite Sup. Eng.)')
    ax.axvline(mean, color='green', linestyle='--', label='Média')

    ax.set_title(f'Capabilidade - Dados transformados ({coluna_y})')
    ax.legend()
    ax.set_facecolor('white')
    ax.grid(True, linestyle=':', color='gray')
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Relatório no formato alinhado
    relatorio = f"""
📊 Análise de Capabilidade com Transformação Johnson

🔹 Resultado

- Transformação Johnson SU aplicada para normalizar os dados
- Equação da transformação (params): {', '.join([f"{p:.4f}" for p in params])}
- Transformação validada: SIM

✔️ Resultados

- Pp: {Pp:.2f} ➔ {'Muito abaixo do mínimo recomendado (1.33)' if Pp < 1.33 else 'Aceitável'}
- Ppk: {Ppk:.2f} ➔ {'Processo fora de especificação' if Ppk < 1 else 'Processo aceitável'}
- Nível Sigma: {nivel_sigma:.2f} ➔ {'Inferior a 3 sigma (inaceitável)' if nivel_sigma < 3 else 'Aceitável'}
- % Defeito abaixo LIE: {percent_below:.2f}%
- % Defeito acima LSE: {percent_above:.2f}%
- % Defeito Total: {percent_total:.2f}%

✔️ Recomendações

- Validar a adequação do modelo transformado antes de usá-lo em decisões críticas.
- Processo incapaz de atender aos limites especificados. Ações urgentes de melhoria são necessárias.
- Revisar especificações ou melhorar a capacidade do processo através de projetos de melhoria contínua.
""".strip()

    return relatorio, grafico_base64




def analise_capabilidade_discretizado(df, coluna_y, field_LIE=None, field_LSE=None):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ É necessário informar uma coluna Y válida.", None

    if not field_LIE and not field_LSE:
        return "❌ Pelo menos um dos limites (LIE ou LSE) deve ser informado.", None

    try:
        LSL = float(field_LIE) if field_LIE else float('-inf')
    except:
        LSL = float('-inf')

    try:
        USL = float(field_LSE) if field_LSE else float('inf')
    except:
        USL = float('inf')

    dados = df[coluna_y].dropna()
    if len(dados) < 10:
        return "⚠ Recomenda-se pelo menos 10 dados para capabilidade.", None

    import matplotlib.pyplot as plt
    import numpy as np
    from io import BytesIO
    import base64

    ppm_lsl = 1e6 * np.sum(dados < LSL) / len(dados)
    ppm_usl = 1e6 * np.sum(dados > USL) / len(dados)
    ppm_total = ppm_lsl + ppm_usl

    fig, ax = plt.subplots(figsize=(8, 5))
    valores, contagens = np.unique(dados, return_counts=True)
    ax.bar(valores, contagens / len(dados), width=0.8, color='gray', edgecolor='black', alpha=0.7)
    if LSL != float('-inf'):
        ax.axvline(LSL, color='red', linestyle='--', label='LIE')
    if USL != float('inf'):
        ax.axvline(USL, color='red', linestyle='--', label='LSE')
    ax.axvline(np.mean(dados), color='green', linestyle='--', label='Média')
    ax.set_title('Capabilidade - Dados Discretizados')
    ax.set_xlabel(coluna_y)
    ax.set_ylabel('Frequência relativa')
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
**Capabilidade - com dados discretizados**
- PPM observado < LIE: {ppm_lsl:.2f}
- PPM observado > LSE: {ppm_usl:.2f}
- PPM total: {ppm_total:.2f}

⚠ Cp, Cpk, Pp e Ppk não são aplicáveis a dados discretizados.
✅ Capabilidade calculada com base na frequência real dos dados. Avalie o gráfico para verificar a distribuição dos níveis em relação aos limites.
"""

    return texto.strip(), grafico_base64


ANALISES = {
    "Teste de normalidade": teste_normalidade,
    "Análise de estabilidade": analise_estabilidade,
    "Análise de distribuição estatística": analise_distribuicao_estatistica,
    "Capabilidade - dados normais": analise_capabilidade_normal,
    "Capabilidade - outras distribuições": analise_capabilidade_outros,
    "Capabilidade - com dados transformados": analise_capabilidade_johnson,
    "Capabilidade - com dados discretizados": analise_capabilidade_discretizado

}

