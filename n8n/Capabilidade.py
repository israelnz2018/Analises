from suporte import *

def analise_teste_normalidade(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) != 1:
        return "❌ O teste de normalidade requer 1 coluna Y.", None

    y_col = colunas_usadas[0]
    dados = df[y_col].dropna()
    if len(dados) < 5:
        return "❌ É necessário pelo menos 5 dados para o teste de normalidade.", None

    from scipy import stats
    import matplotlib.pyplot as plt
    import numpy as np
    from io import BytesIO
    import base64

    # Anderson-Darling
    ad_res = stats.anderson(dados)
    ad_stat = ad_res.statistic
    ad_crit = ad_res.critical_values
    ad_sig = list(ad_res.significance_level)
    ad_pseudo_p = None
    if 5 in ad_sig:
        idx = ad_sig.index(5)
        ad_normal = ad_stat < ad_crit[idx]
    else:
        ad_normal = False

    # Shapiro-Wilk (representando Ryan-Joiner no estilo Minitab)
    sw_stat, sw_p = stats.shapiro(dados)
    sw_normal = sw_p > 0.05

    # Kolmogorov-Smirnov
    ks_stat, ks_p = stats.kstest((dados - np.mean(dados)) / np.std(dados, ddof=1), 'norm')
    ks_normal = ks_p > 0.05

    # Texto
    texto = f"""
**Teste de Normalidade**
- Anderson-Darling: estat={ad_stat:.4f}, normalidade={"Aprovada" if ad_normal else "Reprovada"}
- Shapiro-Wilk: estat={sw_stat:.4f}, p-valor={sw_p:.4f}, normalidade={"Aprovada" if sw_normal else "Reprovada"}
- Kolmogorov-Smirnov: estat={ks_stat:.4f}, p-valor={ks_p:.4f}, normalidade={"Aprovada" if ks_normal else "Reprovada"}
"""

    # Conclusão
    if ad_normal or sw_normal or ks_normal:
        normal_txt = "✅ Pelo menos um teste indica normalidade. Modelo pode prosseguir com métodos paramétricos."
    else:
        normal_txt = "⚠ Nenhum teste indicou normalidade. Recomenda-se coletar pelo menos 50 amostras e realizar teste de estabilidade."

    texto += f"\n**Conclusão**\n{normal_txt}"

    # Gráfico
    fig, ax = plt.subplots(figsize=(6,4))
    stats.probplot(dados, dist="norm", plot=ax)
    if ad_normal:
        ax.set_title("QQ-Plot (Anderson-Darling indicou normalidade)")
    elif sw_normal:
        ax.set_title("QQ-Plot (Shapiro-Wilk indicou normalidade)")
    elif ks_normal:
        ax.set_title("QQ-Plot (Kolmogorov-Smirnov indicou normalidade)")
    else:
        ax.set_title("QQ-Plot (Nenhum teste indicou normalidade)")

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return texto.strip(), grafico_base64

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
    
def analise_distribuicao_estatistica(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) != 1:
        return "❌ A análise de distribuição requer 1 coluna Y.", None

    y_col = colunas_usadas[0]
    dados = df[y_col].dropna()
    if len(dados) < 10:
        return "❌ A análise de distribuição requer pelo menos 10 dados.", None

    from scipy import stats
    import matplotlib.pyplot as plt
    import numpy as np
    from io import BytesIO
    import base64

    distribs = {
        'Normal': stats.norm,
        'Lognormal': stats.lognorm,
        'Exponencial': stats.expon,
        'Weibull': stats.weibull_min,
        'Gama': stats.gamma
    }

    resultados = []

    for nome, dist in distribs.items():
        try:
            params = dist.fit(dados)
            ad_res = stats.anderson(dados, dist='norm') if nome == 'Normal' else None
            # Para simplificação: usamos Anderson-Darling apenas no Normal
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

    # Melhor distribuição
    melhor = max(resultados, key=lambda r: r['P-valor'])
    nome_melhor = melhor['Distribuição']

    # Texto da tabela
    linhas = ["Distribuição                 AD        P-valor"]
    for r in resultados:
        ad_str = f"{r['AD']:.3f}" if not np.isnan(r['AD']) else "-"
        p_str = f"{r['P-valor']:.3f}" if r['P-valor'] >= 0.001 else "<0.001"
        linhas.append(f"{r['Distribuição']:<25} {ad_str:<8} {p_str}")

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

    texto = f"""
**Goodness of Fit Test (Teste de Aderência)**

{chr(10).join(linhas)}

**Conclusão**
✅ A melhor distribuição é: {nome_melhor} (maior P-valor).
"""

    return texto.strip(), grafico_base64

def analise_capabilidade_normal(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) < 1 or not field or not isinstance(field, dict):
        return "❌ É necessário Y, LSL e USL.", None

    y_col = colunas_usadas[0]
    dados = df[y_col].dropna()
    if len(dados) < 30:
        return "⚠ Recomenda-se pelo menos 30 dados para análise de capabilidade.", None

    LSL = float(field.get("Field_LSL", np.nan))
    USL = float(field.get("Field_USL", np.nan))
    if np.isnan(LSL) or np.isnan(USL):
        return "❌ Limites LSL e USL são obrigatórios.", None

    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    # Normalidade
    ad_res = stats.anderson(dados)
    sw_stat, sw_p = stats.shapiro(dados)
    ks_stat, ks_p = stats.kstest((dados - np.mean(dados)) / np.std(dados, ddof=1), 'norm')

    ad_sig = list(ad_res.significance_level)
    ad_normal = False
    if 5 in ad_sig:
        idx = ad_sig.index(5)
        ad_normal = ad_res.statistic < ad_res.critical_values[idx]

    sw_normal = sw_p > 0.05
    ks_normal = ks_p > 0.05

    if not (ad_normal or sw_normal or ks_normal):
        texto = f"""
**Capabilidade - dados normais**
⚠ Nenhum teste indicou normalidade. Recomenda-se realizar análise de distribuição estatística e usar capabilidade para dados não normais.
- Anderson-Darling: estat={ad_res.statistic:.4f}
- Shapiro-Wilk: p-valor={sw_p:.4f}
- Kolmogorov-Smirnov: p-valor={ks_p:.4f}
"""
        return texto.strip(), None

    # Estatísticas
    mean = np.mean(dados)
    std_within = np.std(dados, ddof=1)
    std_overall = np.std(dados, ddof=0)

    Cp = (USL - LSL) / (6 * std_within)
    Cpk = min((USL - mean), (mean - LSL)) / (3 * std_within)

    Pp = (USL - LSL) / (6 * std_overall)
    Ppk = min((USL - mean), (mean - LSL)) / (3 * std_overall)

    z_bench_within = min((USL - mean) / std_within, (mean - LSL) / std_within)
    z_bench_overall = min((USL - mean) / std_overall, (mean - LSL) / std_overall)

    ppm_within_lsl = 1e6 * stats.norm.cdf(LSL, loc=mean, scale=std_within)
    ppm_within_usl = 1e6 * (1 - stats.norm.cdf(USL, loc=mean, scale=std_within))
    ppm_within_total = ppm_within_lsl + ppm_within_usl

    ppm_overall_lsl = 1e6 * stats.norm.cdf(LSL, loc=mean, scale=std_overall)
    ppm_overall_usl = 1e6 * (1 - stats.norm.cdf(USL, loc=mean, scale=std_overall))
    ppm_overall_total = ppm_overall_lsl + ppm_overall_usl

    # Gráfico
    fig, ax = plt.subplots(figsize=(8,5))
    count, bins, ignored = ax.hist(dados, bins=15, density=True, alpha=0.6, color='gray', edgecolor='black')
    x = np.linspace(min(dados), max(dados), 100)
    y_within = stats.norm.pdf(x, mean, std_within)
    y_overall = stats.norm.pdf(x, mean, std_overall)

    ax.plot(x, y_within, 'r-', label='Curto prazo')
    ax.plot(x, y_overall, 'k--', label='Longo prazo')
    ax.axvline(LSL, color='red', linestyle='dashed', label='LSL')
    ax.axvline(USL, color='red', linestyle='dashed', label='USL')
    ax.axvline(mean, color='green', linestyle='dashed', label='Média')
    ax.set_title('Capabilidade - Dados Normais')
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
**Capabilidade - dados normais**
- Anderson-Darling: estat={ad_res.statistic:.4f}, normalidade={"Aprovada" if ad_normal else "Reprovada"}
- Shapiro-Wilk: p-valor={sw_p:.4f}, normalidade={"Aprovada" if sw_normal else "Reprovada"}
- Kolmogorov-Smirnov: p-valor={ks_p:.4f}, normalidade={"Aprovada" if ks_normal else "Reprovada"}

**Resultados**
- Cp = {Cp:.2f}
- Cpk = {Cpk:.2f}
- Pp = {Pp:.2f}
- Ppk = {Ppk:.2f}
- Z.bench curto prazo = {z_bench_within:.2f}
- Z.bench longo prazo = {z_bench_overall:.2f}

**PPM Curto Prazo**
- < LSL: {ppm_within_lsl:.2f}
- > USL: {ppm_within_usl:.2f}
- Total: {ppm_within_total:.2f}

**PPM Longo Prazo**
- < LSL: {ppm_overall_lsl:.2f}
- > USL: {ppm_overall_usl:.2f}
- Total: {ppm_overall_total:.2f}

**Conclusão**
✅ Capabilidade calculada com base nos dados normais. Avalie Cp/Cpk e ações de melhoria se Cpk < 1.33.
"""

    return texto.strip(), grafico_base64



def analise_capabilidade_nao_normal(df, colunas_usadas):
    nome_coluna_y = colunas_usadas[0]
    nome_coluna_x = colunas_usadas[1]

    dados = df[nome_coluna_y].dropna().astype(float)
    limites = df[nome_coluna_x].dropna().unique()

    if len(limites) < 1 or len(limites) > 2:
        raise ValueError("A coluna de limites deve conter um ou dois valores numéricos (LSL e/ou USL).")

    lsl = usl = None
    if len(limites) == 2:
        lsl, usl = sorted(limites)
    elif len(limites) == 1:
        if df[nome_coluna_x].iloc[1] != '':
            lsl = limites[0]
        else:
            usl = limites[0]

    media = np.mean(dados)
    desvio = np.std(dados, ddof=1)
    normal1 = stats.shapiro(dados)[1] > 0.05
    normal2 = stats.kstest(dados, 'norm', args=(media, desvio))[1] > 0.05
    normal3 = stats.anderson(dados).statistic < 0.6810

    if normal1 or normal2 or normal3:
        texto = "📊 **Análise de Capabilidade**\n\n✅ Os dados parecem seguir uma distribuição normal. Recomenda-se utilizar a análise de capabilidade normal."
        return texto, None

    distribuicoes = ['lognorm', 'weibull_min', 'gamma', 'expon', 'beta']
    resultados = []

    for dist_name in distribuicoes:
        dist = getattr(stats, dist_name)
        try:
            params = dist.fit(dados)
            stat, p = stats.kstest(dados, dist_name, args=params)
            resultados.append((dist_name, p, params))
        except Exception:
            continue

    resultados.sort(key=lambda x: x[1], reverse=True)

    texto = "📊 **Teste de Aderência às Distribuições**\n\n"
    for nome, pval, _ in resultados:
        texto += f"- {nome}: p = {pval:.4f}\n"

    if len(resultados) == 0:
        texto += "\n❌ Nenhuma distribuição pôde ser ajustada."
        texto += "\n\n🔁 Recomenda-se aplicar transformação matemática (ex: Yeo-Johnson)."
        return texto, None

    melhor_nome, melhor_p, melhor_params = resultados[0]

    if melhor_p > 0.05:
        texto += f"\n✅ **Melhor distribuição encontrada: {melhor_nome} (p = {melhor_p:.4f})**"

        x = np.linspace(min(dados), max(dados), 500)
        dist = getattr(stats, melhor_nome)

        try:
            y = dist.pdf(x, *melhor_params)
        except Exception:
            return texto + "\n\n❌ Erro ao gerar gráfico da distribuição.", None

        aplicar_estilo_minitab()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dados, bins=15, density=True, alpha=0.7, color="#A6CEE3", edgecolor='black')
        ax.plot(x, y, 'darkred', linewidth=2)

        if lsl is not None:
            ax.axvline(lsl, color='maroon', linestyle='--')
        if usl is not None:
            ax.axvline(usl, color='maroon', linestyle='--')
        ax.axvline(media, color='darkgreen', linestyle='-', linewidth=2)
        ax.text(media, max(y) * 1.05, "Média", ha='center', fontsize=10, color='darkgreen')

        ax.set_title(f'Capabilidade - {melhor_nome}')
        ax.set_xlabel(nome_coluna_y)
        ax.set_ylabel("Densidade")
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        return texto, imagem_base64

    else:
        texto += "\n\n❌ Nenhuma distribuição apresentou p > 0.05."
        texto += "\n\n🔁 Recomenda-se aplicar uma transformação matemática (ex: Yeo-Johnson)."
        return texto, None

def aplicar_transformacao_johnson(df, colunas_usadas):
    nome_coluna_y = colunas_usadas[0]
    nome_coluna_x = colunas_usadas[1]

    dados = df[nome_coluna_y].dropna().astype(float)
    limites = df[nome_coluna_x].dropna().unique()

    if len(limites) < 1 or len(limites) > 2:
        raise ValueError("A coluna de limites deve conter um ou dois valores numéricos (LSL e/ou USL).")

    lsl = usl = None
    if len(limites) == 2:
        lsl, usl = sorted(limites)
    elif len(limites) == 1:
        if df[nome_coluna_x].iloc[1] != '':
            lsl = limites[0]
        else:
            usl = limites[0]

    pt = PowerTransformer(method='yeo-johnson')
    dados_transformados = pt.fit_transform(dados.values.reshape(-1, 1)).flatten()

    stat_ad = stats.anderson(dados_transformados).statistic
    normal = stat_ad < 0.6810

    if normal:
        media = np.mean(dados_transformados)
        desvio = np.std(dados_transformados, ddof=1)

        ppl = ppu = pp = ppk = None
        if lsl is not None:
            ppl = (media - lsl) / (3 * desvio)
        if usl is not None:
            ppu = (usl - media) / (3 * desvio)

        if ppl is not None and ppu is not None:
            pp = (usl - lsl) / (6 * desvio)
            ppk = min(ppl, ppu)
        elif ppl is not None:
            ppk = ppl
        elif ppu is not None:
            ppk = ppu

        sigma_nivel = 3 * ppk if ppk is not None else None

        texto = "🔁 **Transformação Yeo-Johnson aplicada com sucesso**\n\n"
        texto += f"📌 Média transformada: {media:.4f}\n"
        texto += f"📌 Desvio padrão: {desvio:.4f}\n"
        if lsl is not None:
            texto += f"- LSL: {lsl:.4f}\n"
        if usl is not None:
            texto += f"- USL: {usl:.4f}\n"
        if pp is not None:
            texto += f"- Pp: {pp:.4f}\n"
        if ppk is not None:
            texto += f"- Ppk: {ppk:.4f}\n"
        if sigma_nivel is not None:
            texto += f"- Nível Sigma estimado: {sigma_nivel:.4f}"

        aplicar_estilo_minitab()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dados_transformados, bins=15, density=True, alpha=0.7, color="#A6CEE3", edgecolor='black')

        x = np.linspace(min(dados_transformados), max(dados_transformados), 500)
        y = stats.norm.pdf(x, media, desvio)
        ax.plot(x, y, 'darkred', linewidth=2)

        if lsl is not None:
            ax.axvline(lsl, color='maroon', linestyle='--')
        if usl is not None:
            ax.axvline(usl, color='maroon', linestyle='--')
        ax.axvline(media, color='darkgreen', linestyle='-', linewidth=2)
        ax.text(media, max(y) * 1.05, "Média", ha='center', fontsize=10, color='darkgreen')

        ax.set_title("Capabilidade (dados transformados)")
        ax.set_xlabel(nome_coluna_y + " (transformado)")
        ax.set_ylabel("Densidade")
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        return texto, imagem_base64

    else:
        total = len(dados)
        abaixo_lsl = len(dados[dados < lsl]) if lsl is not None else 0
        acima_usl = len(dados[dados > usl]) if usl is not None else 0
        fora = abaixo_lsl + acima_usl
        percentual_fora = fora / total

        try:
            sigma_estimado = stats.norm.ppf(1 - percentual_fora / 2)
        except:
            sigma_estimado = None

        texto = "🔁 **Transformação Yeo-Johnson falhou em normalizar os dados**\n"
        texto += f"\n❌ Dados continuam não normais após transformação."
        texto += f"\n📊 Estimando capabilidade com base em dados discretos:"
        texto += f"\n- Total de amostras: {total}"
        if lsl is not None:
            texto += f"\n- Abaixo do LSL: {abaixo_lsl}"
        if usl is not None:
            texto += f"\n- Acima do USL: {acima_usl}"
        texto += f"\n- % Fora dos limites: {percentual_fora:.2%}"
        if sigma_estimado:
            texto += f"\n- Nível Sigma estimado (aproximado): {sigma_estimado:.2f}"

        aplicar_estilo_minitab()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dados, bins=15, color="#A6CEE3", edgecolor='black', alpha=0.8)

        if lsl is not None:
            ax.axvline(lsl, color='maroon', linestyle='--', label='LSL')
        if usl is not None:
            ax.axvline(usl, color='maroon', linestyle='--', label='USL')
        ax.axvline(np.mean(dados), color='darkgreen', linestyle='-', linewidth=2, label='Média')
        ax.set_title("Capabilidade (dados discretos)")
        ax.set_xlabel(nome_coluna_y)
        ax.set_ylabel("Frequência")
        ax.legend()
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        return texto, imagem_base64, False


ANALISES = {
    "Teste de normalidade": analise_teste_normalidade,
    "Análise de estabilidade": analise_estabilidade,
    "Análise de distribuição estatística": analise_distribuicao_estatistica,
    "Capabilidade - dados normais": analise_capabilidade_normal



    "Capabilidade para outras distribuições": analise_capabilidade_nao_normal,
    "Capabilidade com dados transformados": aplicar_transformacao_johnson
}

