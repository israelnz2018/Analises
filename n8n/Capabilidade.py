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
    
def analise_capabilidade_normal(df, colunas_usadas):
    nome_coluna_y = colunas_usadas[0]
    nome_coluna_x = colunas_usadas[1]

    dados = df[nome_coluna_y].dropna().astype(float)
    limites = df[nome_coluna_x].dropna().astype(float).unique()

    if len(limites) < 1 or len(limites) > 2:
        raise ValueError("A coluna de limites deve conter um ou dois valores numéricos.")

    LSL, USL = None, None
    if len(limites) == 1:
        if not pd.isna(df[nome_coluna_x].iloc[1]):
            LSL = limites[0]
        else:
            USL = limites[0]
    else:
        LSL, USL = sorted(limites[:2])

    media = np.mean(dados)
    desvio_padrao = np.std(dados, ddof=1)
    desvio_padrao_pop = np.std(dados, ddof=0)

    sw_stat, sw_p = stats.shapiro(dados)
    ad_result = stats.anderson(dados)
    ad_stat = ad_result.statistic
    ad_critico = ad_result.critical_values[2]
    ks_stat, ks_p = stats.kstest(dados, 'norm', args=(media, desvio_padrao))

    normal = False
    if sw_p > 0.05 or ad_stat < ad_critico or ks_p > 0.05:
        normal = True

    if not normal:
        return {
            "analise": f"""❌ Os dados **não são normais** segundo os testes de normalidade.
- Shapiro-Wilk: p = {sw_p:.4f}
- Anderson-Darling: estatística = {ad_stat:.4f} | critério = {ad_critico:.4f}
- Kolmogorov-Smirnov: p = {ks_p:.4f}

Recomenda-se utilizar a **Análise de Capabilidade para Dados Não Normais**.""",
            "graficos": [],
            "colunas_utilizadas": colunas_usadas
        }

    cp = ((USL - LSL) / (6 * desvio_padrao)) if (USL and LSL) else None
    cpu = ((USL - media) / (3 * desvio_padrao)) if USL else None
    cpl = ((media - LSL) / (3 * desvio_padrao)) if LSL else None
    cpk = min(cpu or float('inf'), cpl or float('inf'))

    pp = ((USL - LSL) / (6 * desvio_padrao_pop)) if (USL and LSL) else None
    ppu = ((USL - media) / (3 * desvio_padrao_pop)) if USL else None
    ppl = ((media - LSL) / (3 * desvio_padrao_pop)) if LSL else None
    ppk = min(ppu or float('inf'), ppl or float('inf'))

    sigma_nivel = 3 * cpk

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(8, 4))
    counts, bins, patches = ax.hist(dados, bins=15, color="#A6CEE3", edgecolor='black', alpha=0.9, density=True)

    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 500)
    p = stats.norm.pdf(x, media, desvio_padrao)
    ax.plot(x, p, 'darkred', linewidth=2)

    if LSL: ax.axvline(LSL, color='maroon', linestyle='--', linewidth=1.5, label='LSL')
    if USL: ax.axvline(USL, color='maroon', linestyle='--', linewidth=1.5, label='USL')
    ax.axvline(media, color='darkgreen', linestyle='-', linewidth=2, label='Média')
    ax.text(media, max(p) * 1.05, "Alvo", ha='center', va='bottom', fontsize=10, color='darkgreen')

    ax.set_title('Capabilidade do Processo (Normal)', fontsize=14, weight='bold')
    ax.set_xlabel(nome_coluna_y)
    ax.set_ylabel('Densidade')
    ax.set_xlim(xmin, xmax)
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    texto = f"""📊 **Análise de Capabilidade (Distribuição Normal)**

- Média: {media:.4f}
- Desvio Padrão: {desvio_padrao:.4f}
"""

    if LSL: texto += f"- LSL: {LSL:.4f}\n"
    if USL: texto += f"- USL: {USL:.4f}\n"
    if cp: texto += f"- Cp: {cp:.4f}\n"
    if cpk: texto += f"- Cpk: {cpk:.4f}\n"
    if pp: texto += f"- Pp: {pp:.4f}\n"
    if ppk: texto += f"- Ppk: {ppk:.4f}\n"

    texto += f"- Nível Sigma estimado: {sigma_nivel:.4f}"

    return {
        "analise": texto,
        "graficos": [imagem_base64],
        "colunas_utilizadas": colunas_usadas
    }

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

    "Capabilidade para dados normais": analise_capabilidade_normal,
    "Capabilidade para outras distribuições": analise_capabilidade_nao_normal,
    "Capabilidade com dados transformados": aplicar_transformacao_johnson
}

