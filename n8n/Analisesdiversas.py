
from suporte import *

def analise_calculo_probabilidade(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) != 1:
        return "❌ O cálculo de probabilidade requer 1 coluna Y (variável).", None

    y_col = colunas_usadas[0]
    dados = df[y_col].dropna()
    if len(dados) < 5:
        return "❌ É necessário pelo menos 5 dados para o cálculo de probabilidade.", None

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from io import BytesIO
    import base64

    # Interpreta fields
    fields = field if field else []
    valor_menor = float(fields[0]) if len(fields) > 0 and fields[0] not in [None, ""] else None
    valor_maior = float(fields[1]) if len(fields) > 1 and fields[1] not in [None, ""] else None
    valor_entre_inf = float(fields[2]) if len(fields) > 2 and fields[2] not in [None, ""] else None
    valor_entre_sup = float(fields[3]) if len(fields) > 3 and fields[3] not in [None, ""] else None

    # Validação
    campos_preenchidos = sum([valor_menor is not None, valor_maior is not None, 
                              (valor_entre_inf is not None and valor_entre_sup is not None)])
    if campos_preenchidos != 1:
        return "❌ Preencha apenas um tipo de cálculo: menor que, maior que ou entre dois valores.", None

    # Testa distribuições
    distros = {
        "Normal": lambda x: stats.anderson(x, 'norm'),
        "Lognormal": lambda x: stats.anderson(np.log(x[x > 0]), 'norm') if np.all(x > 0) else None,
        "Exponencial": lambda x: stats.anderson(x, 'expon'),
        "Weibull": None,  # Scipy não tem Anderson para Weibull; pode implementar se quiser
    }

    resultados = {}
    for nome, func in distros.items():
        if func is None:
            continue
        try:
            res = func(dados)
            if res is None:
                continue
            stat = res.statistic
            crit = res.critical_values
            sig = list(res.significance_level)
            p_aprox = max([s for s, c in zip(sig, crit) if stat < c], default=0)
            resultados[nome] = p_aprox
        except:
            continue

    if not resultados:
        return "❌ Nenhuma distribuição pôde ser ajustada aos dados. Sugere-se coletar mais dados ou aplicar transformação (ex: Johnson).", None

    # Decide distribuição
    if "Normal" in resultados and resultados["Normal"] >= 5:
        escolhida = "Normal"
    else:
        escolhida = max(resultados, key=resultados.get)

    if resultados[escolhida] < 5:
        return "⚠ Nenhuma distribuição passou no teste (p ≥ 5%). Recomenda-se coletar mais dados ou aplicar transformação (ex: Johnson).", None

    # Ajuste parâmetros
    if escolhida == "Normal":
        mu, sigma = np.mean(dados), np.std(dados, ddof=1)
        dist = stats.norm(loc=mu, scale=sigma)
        parametros_txt = f"média={mu:.4f}, sigma={sigma:.4f}"
    elif escolhida == "Lognormal":
        log_dados = np.log(dados[dados > 0])
        mu, sigma = np.mean(log_dados), np.std(log_dados, ddof=1)
        dist = stats.lognorm(s=sigma, scale=np.exp(mu))
        parametros_txt = f"mu={mu:.4f}, sigma={sigma:.4f}"
    elif escolhida == "Exponencial":
        loc, scale = stats.expon.fit(dados)
        dist = stats.expon(loc=loc, scale=scale)
        parametros_txt = f"loc={loc:.4f}, scale={scale:.4f}"
    else:
        return "⚠ Distribuição escolhida não suportada no cálculo automático.", None

    # Calcula probabilidade
    if valor_menor is not None:
        prob = dist.cdf(valor_menor)
        texto_prob = f"P(X ≤ {valor_menor}) = {prob:.4%}"
    elif valor_maior is not None:
        prob = dist.sf(valor_maior)
        texto_prob = f"P(X ≥ {valor_maior}) = {prob:.4%}"
    else:
        prob = dist.cdf(valor_entre_sup) - dist.cdf(valor_entre_inf)
        texto_prob = f"P({valor_entre_inf} ≤ X ≤ {valor_entre_sup}) = {prob:.4%}"

    # Gera gráfico
    x = np.linspace(min(dados), max(dados), 500)
    y = dist.pdf(x)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, label=f'{escolhida} ajustada')
    ax.fill_between(
        x,
        y,
        0,
        where=(
            (valor_menor is not None and x <= valor_menor) |
            (valor_maior is not None and x >= valor_maior) |
            (valor_entre_inf is not None and (x >= valor_entre_inf) & (x <= valor_entre_sup))
        ),
        alpha=0.3,
        color='blue',
        label='Área da probabilidade'
    )
    ax.set_title(f"Cálculo de Probabilidade - {escolhida}")
    ax.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Texto final
    texto = f"""
**Cálculo de Probabilidade**
- Distribuição escolhida: {escolhida} ({parametros_txt})
- {texto_prob}
"""

    return texto.strip(), grafico_base64

