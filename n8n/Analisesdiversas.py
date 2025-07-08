
from suporte import *

def analise_probabilidade_baixo_X(df, coluna_y, field=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from fitter import Fitter
    from io import BytesIO
    import base64

    aplicar_estilo_minitab()

    # 🔷 Validação coluna
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    # 🔷 Validação do field como valor de referência
    if field is None:
        return "❌ O valor de referência X (field) deve ser informado.", None

    try:
        x_ref = float(field)
    except:
        return "❌ O valor de referência X deve ser numérico.", None

    dados = df[[coluna_y]].dropna()
    y = dados[coluna_y].astype(float).values

    if len(y) < 5:
        return "❌ Dados insuficientes para análise.", None

    # 🔷 3 testes de normalidade (Minitab padrão)
    ad_stat, ad_crit, _ = stats.anderson(y)
    sw_p = stats.shapiro(y)[1]
    ks_p = stats.kstest(y, 'norm', args=(np.mean(y), np.std(y, ddof=1)))[1]

    normal_flag = (sw_p > 0.05) and (ks_p > 0.05) and (ad_stat < ad_crit[2])  # Anderson critical value 5%

    # 🔷 Se normal, usa normal
    if normal_flag:
        dist_name = "Normal"
        mu = np.mean(y)
        sigma = np.std(y, ddof=1)
        prob = stats.norm.cdf(x_ref, mu, sigma)

        x_plot = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        y_plot = stats.norm.pdf(x_plot, mu, sigma)

    else:
        # 🔷 Ajusta distribuições alternativas
        f = Fitter(y, distributions=['lognorm','weibull_min','gamma','expon'])
        f.fit()
        best = f.get_best()
        dist_name = list(best.keys())[0]
        params = list(best.values())[0]

        dist = getattr(stats, dist_name)

        # 🔷 Validação de tipos numéricos e estrutura de params
        params = [float(p) for p in params]

        # 🔷 Separando shape, loc, scale
        if dist_name == 'lognorm':
            shape, loc, scale = params
            prob = dist.cdf(x_ref, shape, loc=loc, scale=scale)
            x_plot = np.linspace(min(y), max(y), 1000)
            y_plot = dist.pdf(x_plot, shape, loc=loc, scale=scale)

        elif dist_name == 'weibull_min':
            c, loc, scale = params
            prob = dist.cdf(x_ref, c, loc=loc, scale=scale)
            x_plot = np.linspace(min(y), max(y), 1000)
            y_plot = dist.pdf(x_plot, c, loc=loc, scale=scale)

        elif dist_name == 'gamma':
            a, loc, scale = params
            prob = dist.cdf(x_ref, a, loc=loc, scale=scale)
            x_plot = np.linspace(min(y), max(y), 1000)
            y_plot = dist.pdf(x_plot, a, loc=loc, scale=scale)

        elif dist_name == 'expon':
            loc, scale = params
            prob = dist.cdf(x_ref, loc=loc, scale=scale)
            x_plot = np.linspace(min(y), max(y), 1000)
            y_plot = dist.pdf(x_plot, loc=loc, scale=scale)

        else:
            return f"❌ Distribuição '{dist_name}' não suportada no momento.", None

    # 🔷 Gráfico
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(x_plot, y_plot, color='black')
    ax.fill_between(x_plot, y_plot, 0, where=(x_plot <= x_ref), color='red', alpha=0.3)
    ax.axvline(x_ref, color='red', linestyle='--', label=f'X = {x_ref}')
    ax.set_title(f"📊 Probabilidade de X ≤ {x_ref} ({dist_name})", fontsize=18, fontweight='bold')
    ax.set_ylabel("Densidade", fontsize=16, fontweight='bold')
    ax.set_xlabel(coluna_y, fontsize=16, fontweight='bold')
    ax.legend()
    plt.tight_layout()

    # 🔷 Converter imagem
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 🔷 Report
    texto = (
        f"📊 **Análise – Probabilidade abaixo de {x_ref}**\n"
        f"🔎 **Distribuição utilizada:** {dist_name}\n"
        f"🔎 **Valor X de referência:** {x_ref}\n"
        f"🔎 **Probabilidade acumulada (X ≤ {x_ref}):** {prob:.2%}\n"
    )

    if normal_flag:
        texto += "✅ Os dados foram considerados **normais** com base nos testes aplicados.\n"
    else:
        texto += "❌ Os dados **não foram normais**. A melhor distribuição foi ajustada automaticamente.\n"

    texto += (
        "🔎 **Conclusão:**\n"
        "➡️ O cálculo indica a probabilidade acumulada até o valor X. "
        "Use esta informação para avaliar riscos ou proporções esperadas no processo.\n"
    )

    return texto.strip(), img_base64





def analise_cluster_mista(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) < 2:
        return "❌ A análise de cluster (mista) requer pelo menos 2 variáveis (Xs e/ou Xs_discreto).", None

    dados = df[colunas_usadas].dropna()
    if dados.shape[0] < 5:
        return "❌ É necessário pelo menos 5 linhas para formar clusters.", None

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    # Separa numérico e categórico
    numericos = dados.select_dtypes(include=np.number)
    categoricos = dados.select_dtypes(exclude=np.number)

    # Normaliza numéricos
    if not numericos.empty:
        scaler = StandardScaler()
        numericos_scaled = scaler.fit_transform(numericos)
    else:
        numericos_scaled = np.empty((len(dados), 0))

    # One-hot para categóricos
    if not categoricos.empty:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        categoricos_encoded = encoder.fit_transform(categoricos)
    else:
        categoricos_encoded = np.empty((len(dados), 0))

    # Junta
    matriz = np.hstack([numericos_scaled, categoricos_encoded])

    # Cluster
    cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
    labels = cluster.fit_predict(matriz)

    # Relatório
    dados_cluster = dados.copy()
    dados_cluster['Cluster'] = labels
    resumo = []
    for c in np.unique(labels):
        grupo = dados_cluster[dados_cluster['Cluster'] == c]
        resumo.append(f"- Cluster {c}: {len(grupo)} amostras")
    texto = "**Análise de Cluster (mista)**\n" + "\n".join(resumo)

    # Gráfico 2D
    if matriz.shape[1] > 2:
        pca = PCA(n_components=2)
        matriz_plot = pca.fit_transform(matriz)
    else:
        matriz_plot = matriz

    fig, ax = plt.subplots(figsize=(8, 4))
    for c in np.unique(labels):
        pontos = matriz_plot[labels == c]
        ax.scatter(pontos[:, 0], pontos[:, 1], label=f'Cluster {c}', alpha=0.6)
    ax.set_title("Clusters (mista) - Redução PCA")
    ax.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return texto.strip(), grafico_base64


ANALISES = {
    "Cálculo de probabilidade": analise_probabilidade_baixo_X,
    "Análise de Cluster (mista)": analise_cluster_mista

}


