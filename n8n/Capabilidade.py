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


def analise_capabilidade_outros(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) < 1 or not field or not isinstance(field, dict):
        return "❌ É necessário Y, LSL, USL e a distribuição escolhida.", None

    y_col = colunas_usadas[0]
    dados = df[y_col].dropna()
    if len(dados) < 30:
        return "⚠ Recomenda-se pelo menos 30 dados para capabilidade.", None

    LSL = float(field.get("Field_LSL", np.nan))
    USL = float(field.get("Field_USL", np.nan))
    dist_nome = field.get("Field_Distribuicao", "").strip()
    if np.isnan(LSL) or np.isnan(USL) or dist_nome == "":
        return "❌ LSL, USL e distribuição são obrigatórios.", None

    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    # Mapear distribuição
    distros = {
        "Normal": stats.norm,
        "Lognormal": stats.lognorm,
        "Exponencial": stats.expon,
        "Weibull": stats.weibull_min,
        "Gamma": stats.gamma,
        "Loglogística": stats.fisk
    }

    if dist_nome not in distros:
        return f"❌ Distribuição '{dist_nome}' não suportada no momento.", None

    dist = distros[dist_nome]

    # Ajuste da distribuição
    try:
        params = dist.fit(dados)
    except Exception as e:
        return f"❌ Erro ao ajustar a distribuição: {str(e)}", None

    # Goodness of fit (Kolmogorov-Smirnov)
    ks_stat, ks_p = stats.kstest(dados, dist.cdf, args=params)

    # Capabilidade
    ppm_lsl = 1e6 * dist.cdf(LSL, *params)
    ppm_usl = 1e6 * (1 - dist.cdf(USL, *params))
    ppm_total = ppm_lsl + ppm_usl
    z_bench = min(
        (dist.ppf(0.99865, *params) - LSL) / (dist.ppf(0.99865, *params) - dist.ppf(0.00135, *params)),
        (USL - dist.ppf(0.00135, *params)) / (dist.ppf(0.99865, *params) - dist.ppf(0.00135, *params))
    ) * 6

    # Gráfico
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(dados, bins=15, density=True, alpha=0.6, color='gray', edgecolor='black')
    x = np.linspace(min(dados), max(dados), 100)
    y = dist.pdf(x, *params)
    ax.plot(x, y, 'b-', label=f'Curva {dist_nome}')
    ax.axvline(LSL, color='red', linestyle='--', label='LSL')
    ax.axvline(USL, color='red', linestyle='--', label='USL')
    ax.axvline(np.mean(dados), color='green', linestyle='--', label='Média')
    ax.set_title(f'Capabilidade - {dist_nome}')
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Texto
    alerta = ""
    if ks_p < 0.05:
        alerta = f"⚠ A distribuição {dist_nome} não se ajusta bem aos dados (p-valor={ks_p:.4f}). Não recomendado para capabilidade."
    else:
        alerta = f"✅ A distribuição {dist_nome} apresentou bom ajuste (p-valor={ks_p:.4f})."

    texto = f"""
**Capabilidade - outras distribuições**
- Distribuição escolhida: {dist_nome}
- Parâmetros estimados: {', '.join([f'{p:.4f}' for p in params])}
- Kolmogorov-Smirnov p-valor: {ks_p:.4f}

**Resultados**
- PPM < LSL: {ppm_lsl:.2f}
- PPM > USL: {ppm_usl:.2f}
- PPM Total: {ppm_total:.2f}
- Nível sigma estimado (Z.bench): {z_bench:.2f}

**Conclusão**
{alerta}
"""

    return texto.strip(), grafico_base64

def analise_capabilidade_outros(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) < 1 or not field or not isinstance(field, dict):
        return "❌ É necessário Y, LSL, USL e a distribuição escolhida.", None

    y_col = colunas_usadas[0]
    dados = df[y_col].dropna()
    if len(dados) < 30:
        return "⚠ Recomenda-se pelo menos 30 dados para capabilidade.", None

    LSL = float(field.get("Field_LSL", np.nan))
    USL = float(field.get("Field_USL", np.nan))
    dist_nome = field.get("Field_Distribuicao", "").strip()
    if np.isnan(LSL) or np.isnan(USL) or dist_nome == "":
        return "❌ LSL, USL e distribuição são obrigatórios.", None

    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    # Mapear distribuição
    distros = {
        "Normal": stats.norm,
        "Lognormal": stats.lognorm,
        "Exponencial": stats.expon,
        "Weibull": stats.weibull_min,
        "Gamma": stats.gamma,
        "Loglogística": stats.fisk
    }

    if dist_nome not in distros:
        return f"❌ Distribuição '{dist_nome}' não suportada no momento.", None

    dist = distros[dist_nome]

    # Ajuste da distribuição
    try:
        params = dist.fit(dados)
    except Exception as e:
        return f"❌ Erro ao ajustar a distribuição: {str(e)}", None

    # Goodness of fit (Kolmogorov-Smirnov)
    ks_stat, ks_p = stats.kstest(dados, dist.cdf, args=params)

    # Capabilidade
    ppm_lsl = 1e6 * dist.cdf(LSL, *params)
    ppm_usl = 1e6 * (1 - dist.cdf(USL, *params))
    ppm_total = ppm_lsl + ppm_usl
    z_bench = min(
        (dist.ppf(0.99865, *params) - LSL) / (dist.ppf(0.99865, *params) - dist.ppf(0.00135, *params)),
        (USL - dist.ppf(0.00135, *params)) / (dist.ppf(0.99865, *params) - dist.ppf(0.00135, *params))
    ) * 6

    # Gráfico
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(dados, bins=15, density=True, alpha=0.6, color='gray', edgecolor='black')
    x = np.linspace(min(dados), max(dados), 100)
    y = dist.pdf(x, *params)
    ax.plot(x, y, 'b-', label=f'Curva {dist_nome}')
    ax.axvline(LSL, color='red', linestyle='--', label='LSL')
    ax.axvline(USL, color='red', linestyle='--', label='USL')
    ax.axvline(np.mean(dados), color='green', linestyle='--', label='Média')
    ax.set_title(f'Capabilidade - {dist_nome}')
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Texto
    alerta = ""
    if ks_p < 0.05:
        alerta = f"⚠ A distribuição {dist_nome} não se ajusta bem aos dados (p-valor={ks_p:.4f}). Não recomendado para capabilidade."
    else:
        alerta = f"✅ A distribuição {dist_nome} apresentou bom ajuste (p-valor={ks_p:.4f})."

    texto = f"""
**Capabilidade - outras distribuições**
- Distribuição escolhida: {dist_nome}
- Parâmetros estimados: {', '.join([f'{p:.4f}' for p in params])}
- Kolmogorov-Smirnov p-valor: {ks_p:.4f}

**Resultados**
- PPM < LSL: {ppm_lsl:.2f}
- PPM > USL: {ppm_usl:.2f}
- PPM Total: {ppm_total:.2f}
- Nível sigma estimado (Z.bench): {z_bench:.2f}

**Conclusão**
{alerta}
"""

    return texto.strip(), grafico_base64

def analise_capabilidade_transformado(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) < 1 or not field or not isinstance(field, dict):
        return "❌ É necessário Y, LSL e USL.", None

    y_col = colunas_usadas[0]
    dados = df[y_col].dropna()
    if len(dados) < 30:
        return "⚠ Recomenda-se pelo menos 30 dados para capabilidade.", None

    LSL = float(field.get("Field_LSL", np.nan))
    USL = float(field.get("Field_USL", np.nan))
    if np.isnan(LSL) or np.isnan(USL):
        return "❌ LSL e USL são obrigatórios.", None

    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PowerTransformer
    from io import BytesIO
    import base64

    # Testa distribuições
    distribs = {
        'Normal': stats.norm,
        'Lognormal': stats.lognorm,
        'Exponencial': stats.expon,
        'Weibull': stats.weibull_min,
        'Gama': stats.gamma,
        'Loglogística': stats.fisk
    }

    resultados = []
    for nome, dist in distribs.items():
        try:
            params = dist.fit(dados)
            ks_stat, ks_p = stats.kstest(dados, dist.cdf, args=params)
            resultados.append((nome, ks_p, params))
        except Exception:
            resultados.append((nome, 0.0, None))

    melhor_dist = max(resultados, key=lambda x: x[1])
    nome_melhor, p_melhor, params_melhor = melhor_dist

    # Caso 1: encontrou boa distribuição
    if p_melhor >= 0.05:
        texto = f"""
**Capabilidade - dados transformados**
✅ A distribuição {nome_melhor} apresentou bom ajuste (p-valor={p_melhor:.4f}).
Recomenda-se usar a capabilidade para essa distribuição ao invés de transformação.
"""
        return texto.strip(), None

    # Caso 2: tentar transformação Johnson
    try:
        pt = PowerTransformer(method='yeo-johnson')
        dados_t = pt.fit_transform(dados.values.reshape(-1,1)).flatten()
        ad_res = stats.anderson(dados_t)
        ad_sig = list(ad_res.significance_level)
        if 5 in ad_sig:
            idx = ad_sig.index(5)
            ad_normal = ad_res.statistic < ad_res.critical_values[idx]
        else:
            ad_normal = False
    except Exception as e:
        return f"❌ Erro na transformação Johnson: {str(e)}", None

    if not ad_normal:
        texto = f"""
**Capabilidade - dados transformados**
⚠ Nenhuma distribuição apresentou bom ajuste. Nenhuma transformação Johnson obteve normalidade (Anderson-Darling estat={ad_res.statistic:.4f}).
Recomenda-se realizar capabilidade para dados discretizados ou usar métodos não paramétricos.
"""
        return texto.strip(), None

    # Capabilidade no transformado
    mean = np.mean(dados_t)
    std = np.std(dados_t, ddof=1)
    Cp = (USL - LSL) / (6 * std)
    Cpk = min((USL - mean), (mean - LSL)) / (3 * std)
    z_bench = min((USL - mean)/std, (mean - LSL)/std)

    ppm_lsl = 1e6 * stats.norm.cdf(LSL, loc=mean, scale=std)
    ppm_usl = 1e6 * (1 - stats.norm.cdf(USL, loc=mean, scale=std))
    ppm_total = ppm_lsl + ppm_usl

    # Gráfico
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(dados_t, bins=15, density=True, alpha=0.6, color='gray', edgecolor='black')
    x = np.linspace(min(dados_t), max(dados_t), 100)
    y = stats.norm.pdf(x, mean, std)
    ax.plot(x, y, 'b-', label='Curva Normal transformada')
    ax.axvline(LSL, color='red', linestyle='--', label='LSL')
    ax.axvline(USL, color='red', linestyle='--', label='USL')
    ax.axvline(mean, color='green', linestyle='--', label='Média')
    ax.set_title('Capabilidade - Johnson Transformado')
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
**Capabilidade - dados transformados**
⚠ Nenhuma distribuição apresentou bom ajuste. Procedeu-se com transformação Johnson.
- Anderson-Darling no transformado: estat={ad_res.statistic:.4f}, normalidade={'Aprovada' if ad_normal else 'Reprovada'}

**Resultados**
- Cp = {Cp:.2f}
- Cpk = {Cpk:.2f}
- Z.bench = {z_bench:.2f}
- PPM < LSL: {ppm_lsl:.2f}
- PPM > USL: {ppm_usl:.2f}
- PPM Total: {ppm_total:.2f}

✅ Transformação Johnson bem-sucedida. Capabilidade calculada no transformado.
"""

    return texto.strip(), grafico_base64
def analise_capabilidade_discretizado(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) < 1 or not field or not isinstance(field, dict):
        return "❌ É necessário Y, LSL e USL.", None

    y_col = colunas_usadas[0]
    dados = df[y_col].dropna()
    if len(dados) < 10:
        return "⚠ Recomenda-se pelo menos 10 dados para capabilidade.", None

    LSL = float(field.get("Field_LSL", np.nan))
    USL = float(field.get("Field_USL", np.nan))
    if np.isnan(LSL) or np.isnan(USL):
        return "❌ LSL e USL são obrigatórios.", None

    import matplotlib.pyplot as plt
    import numpy as np
    from io import BytesIO
    import base64

    # PPM observado
    ppm_lsl = 1e6 * np.sum(dados < LSL) / len(dados)
    ppm_usl = 1e6 * np.sum(dados > USL) / len(dados)
    ppm_total = ppm_lsl + ppm_usl

    # Gráfico
    fig, ax = plt.subplots(figsize=(8,5))
    valores, contagens = np.unique(dados, return_counts=True)
    ax.bar(valores, contagens / len(dados), width=0.8, color='gray', edgecolor='black', alpha=0.7)
    ax.axvline(LSL, color='red', linestyle='--', label='LSL')
    ax.axvline(USL, color='red', linestyle='--', label='USL')
    ax.axvline(np.mean(dados), color='green', linestyle='--', label='Média')
    ax.set_title('Capabilidade - Dados Discretizados')
    ax.set_xlabel(y_col)
    ax.set_ylabel('Frequência relativa')
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Texto
    texto = f"""
**Capabilidade - com dados discretizados**
- PPM observado < LSL: {ppm_lsl:.2f}
- PPM observado > USL: {ppm_usl:.2f}
- PPM total: {ppm_total:.2f}

⚠ Cp, Cpk, Pp e Ppk não são aplicáveis a dados discretizados.
✅ Capabilidade calculada com base na frequência real dos dados. Avalie o gráfico para verificar a distribuição dos níveis em relação aos limites.
"""

    return texto.strip(), grafico_base64

ANALISES = {
    "Teste de normalidade": analise_teste_normalidade,
    "Análise de estabilidade": analise_estabilidade,
    "Análise de distribuição estatística": analise_distribuicao_estatistica,
    "Capabilidade - dados normais": analise_capabilidade_normal,
    "Capabilidade - outras distribuições": analise_capabilidade_outros,
    "Capabilidade - com dados transformados": analise_capabilidade_transformado,
    "Capabilidade - com dados discretizados": analise_capabilidade_discretizado

}

