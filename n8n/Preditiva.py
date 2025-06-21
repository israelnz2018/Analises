from suporte import *

def analise_regressao_linear_simples(df, colunas):
    colunas = [interpretar_coluna(df, c) for c in colunas]
    X = df[colunas[0]].astype(str).str.strip().str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    Y = df[colunas[1]].astype(str).str.strip().str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    X = pd.to_numeric(X, errors="coerce")
    Y = pd.to_numeric(Y, errors="coerce")
    validos = ~(X.isna() | Y.isna())
    X = X[validos]
    Y = Y[validos]

    if len(X) < 2 or len(Y) < 2:
        raise ValueError("Não há dados numéricos suficientes para a regressão.")

    X_const = sm.add_constant(X)
    modelo = sm.OLS(Y, X_const).fit()

    a = modelo.params.iloc[0]
    b = modelo.params.iloc[1]
    p_valor = modelo.pvalues.iloc[1]
    r2 = modelo.rsquared
    r2_ajustado = modelo.rsquared_adj
    erro_padrao = np.sqrt(modelo.mse_resid)

    resumo = f"""
**Equação da reta:**  y = {a:.3f} + {b:.3f}·x  
**Valor-p da inclinação:**  {p_valor:.4f}  
**Coeficiente de determinação (R²):**  {r2:.4f}  
**R² ajustado:**  {r2_ajustado:.4f}  
**Erro padrão da estimativa:**  {erro_padrao:.4f}
""".strip()

    aplicar_estilo_minitab()
    plt.figure(figsize=(8, 6))
    sns.regplot(x=X, y=Y, ci=None, line_kws={"color": "red"})
    plt.xlabel(colunas[0])
    plt.ylabel(colunas[1])
    plt.title("Regressão Linear Simples")

    return resumo, salvar_grafico()


def analise_regressao_linear_multipla(df, colunas):
    colunas = [interpretar_coluna(df, c) for c in colunas]

    y_col = colunas[-1]
    x_cols = colunas[:-1]

    X = df[x_cols].apply(pd.to_numeric, errors='coerce')
    Y = pd.to_numeric(df[y_col], errors='coerce')

    dados = pd.concat([X, Y], axis=1).dropna()
    X = dados[x_cols]
    Y = dados[y_col]

    if len(dados) < 3:
        raise ValueError("Não há dados suficientes após remoção de NaNs para análise.")

    X_const = sm.add_constant(X)
    modelo = sm.OLS(Y, X_const).fit()

    eq_terms = [f"{coef:.2f}*{var}" for var, coef in modelo.params.items() if var != 'const']
    equacao = f"{modelo.params['const']:.2f} + " + " + ".join(eq_terms)

    r2 = modelo.rsquared
    r2_adj = modelo.rsquared_adj
    erro_padrao = modelo.mse_resid**0.5
    p_valor_modelo = modelo.f_pvalue

    vif_data = pd.DataFrame()
    vif_data["Variável"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

    residuos = modelo.resid
    residuos_padronizados = (residuos - residuos.mean()) / residuos.std()
    outliers = sum(abs(residuos_padronizados) > 3)

    stat_ad, crit_vals, sig_levels = stats.anderson(residuos, dist='norm')
    limiar_5 = crit_vals[sig_levels.tolist().index(5.0)]
    passou_normalidade = stat_ad < limiar_5

    dw = durbin_watson(residuos)

    texto = f"""📊 Regressão Linear Múltipla

🔹 Equação:
Y = {equacao}

🔹 Qualidade do modelo:
- R² = {r2:.3f}
- R² ajustado = {r2_adj:.3f}
- Erro padrão da estimativa = {erro_padrao:.3f}
- Valor-p do modelo = {p_valor_modelo:.4f}

🔹 VIF:
""" + "\n".join([f"  - {row['Variável']}: {row['VIF']:.2f}" for _, row in vif_data.iterrows() if row['Variável'] != 'const']) + f"""

🔹 Resíduos:
- Anderson-Darling (5%): {'✅' if passou_normalidade else '❌'} (estat={stat_ad:.4f}, lim={limiar_5:.4f})
- Durbin-Watson: {dw:.2f}
- Outliers (>|3|): {outliers}
"""

    aplicar_estilo_minitab()
    plt.figure(figsize=(6, 4))
    sns.histplot(residuos, kde=True, color="steelblue", edgecolor="black")
    plt.title("Histograma dos Resíduos")
    return texto.strip(), salvar_grafico()

def analise_regressao_logistica_binaria(df, colunas_usadas):
    if len(colunas_usadas) < 2:
        return "❌ É necessário selecionar uma coluna Y (resposta binária) e pelo menos uma coluna X (numérica).", None

    nome_coluna_y = colunas_usadas[0]
    nomes_colunas_x = colunas_usadas[1:]

    y_raw = df[nome_coluna_y].dropna()
    X_raw = df[nomes_colunas_x]
    df_modelo = pd.concat([y_raw, X_raw], axis=1).dropna()
    y = df_modelo[nome_coluna_y]
    X = df_modelo[nomes_colunas_x]

    if y.dtype == object or str(y.dtype).startswith('category'):
        y = pd.factorize(y)[0]

    X = sm.add_constant(X)
    modelo = sm.Logit(y, X)
    resultado = modelo.fit(disp=0)

    pseudo_r2 = resultado.prsquared
    resumo = resultado.summary2().as_text()

    interpretacao = f"""📊 **Regressão Logística Binária**  
🔹 Variável de resposta (Y): {nome_coluna_y}  
🔹 Variáveis preditoras (X): {", ".join(nomes_colunas_x)}  
🔹 Pseudo R²: {pseudo_r2:.4f}"""

    imagem_base64 = None
    if len(nomes_colunas_x) == 1:
        nome_x = nomes_colunas_x[0]
        x_plot = df_modelo[nome_x]
        y_plot = y

        x_ord = np.linspace(x_plot.min(), x_plot.max(), 100)
        X_pred = sm.add_constant(pd.DataFrame({nome_x: x_ord}))
        y_pred = resultado.predict(X_pred)

        aplicar_estilo_minitab()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x_plot, y_plot, alpha=0.7, color="black", label="Dados")
        ax.plot(x_ord, y_pred, color="red", linewidth=2, label="Curva Ajustada")
        ax.set_xlabel(nome_x)
        ax.set_ylabel(f"Probabilidade de {nome_coluna_y}")
        ax.set_title("Gráfico de Linha Ajustada - Regressão Logística")
        ax.legend()
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return interpretacao + "\n\n```\n" + resumo + "\n```", imagem_base64


def analise_regressao_logistica_nominal(df, colunas_usadas):
    if len(colunas_usadas) < 2:
        return "❌ É necessário selecionar uma coluna Y (nominal com mais de duas categorias) e pelo menos uma coluna X (numérica).", None

    nome_coluna_y = colunas_usadas[0]
    nomes_colunas_x = colunas_usadas[1:]

    y_raw = df[nome_coluna_y].dropna()
    X_raw = df[nomes_colunas_x]
    df_modelo = pd.concat([y_raw, X_raw], axis=1).dropna()
    y = df_modelo[nome_coluna_y]
    X = df_modelo[nomes_colunas_x]

    if y.dtype == object or str(y.dtype).startswith("category"):
        y, categorias = pd.factorize(y)

    X = sm.add_constant(X)
    modelo = sm.MNLogit(y, X)
    resultado = modelo.fit(disp=0)

    pseudo_r2 = 1 - resultado.llf / resultado.llnull
    resumo = resultado.summary().as_text()

    interpretar = f"""📊 **Regressão Logística Nominal**  
🔹 Y: {nome_coluna_y}  
🔹 X: {", ".join(nomes_colunas_x)}  
🔹 Pseudo R²: {pseudo_r2:.4f}"""

    imagem_base64 = None
    try:
        aplicar_estilo_minitab()
        fig, ax = plt.subplots(figsize=(6, 4))
        df_modelo[nome_coluna_y].value_counts().sort_index().plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title("Distribuição da variável resposta")
        ax.set_xlabel(nome_coluna_y)
        ax.set_ylabel("Frequência")
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    except:
        imagem_base64 = None

    return interpretar + "\n\n```\n" + resumo + "\n```", imagem_base64


def analise_regressao_logistica_ordinal(df, colunas_usadas):
    if len(colunas_usadas) < 2:
        return "❌ É necessário selecionar uma coluna Y (ordinal) e pelo menos uma coluna X.", None

    nome_coluna_y = colunas_usadas[0]
    nomes_colunas_x = colunas_usadas[1:]

    y_raw = df[nome_coluna_y].dropna()
    if not pd.api.types.is_categorical_dtype(y_raw) or not y_raw.cat.ordered:
        categorias_ordenadas = sorted(y_raw.dropna().unique())
        y = pd.Categorical(y_raw, categories=categorias_ordenadas, ordered=True)
    else:
        y = y_raw

    X_raw = df[nomes_colunas_x].apply(pd.to_numeric, errors="coerce")
    df_modelo = pd.concat([pd.Series(y, name=nome_coluna_y), X_raw], axis=1).dropna()
    y = df_modelo[nome_coluna_y]
    X = df_modelo[nomes_colunas_x]

    modelo = OrderedModel(y, X, distr="logit")
    resultado = modelo.fit(method="bfgs", disp=0)

    pseudo_r2 = 1 - resultado.llf / resultado.llnull
    resumo = resultado.summary().as_text()

    interpretar = f"""📊 **Regressão Logística Ordinal**  
🔹 Y: {nome_coluna_y}  
🔹 X: {", ".join(nomes_colunas_x)}  
🔹 Pseudo R²: {pseudo_r2:.4f}"""

    imagem_base64 = None
    try:
        aplicar_estilo_minitab()
        fig, ax = plt.subplots(figsize=(6, 4))
        df_modelo[nome_coluna_y].value_counts().sort_index().plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title("Distribuição da variável resposta")
        ax.set_xlabel(nome_coluna_y)
        ax.set_ylabel("Frequência")
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    except:
        imagem_base64 = None

    return interpretar + "\n\n```\n" + resumo + "\n```", imagem_base64


def analise_chi_quadrado(df, colunas_usadas):
    if len(colunas_usadas) < 2:
        raise ValueError("O teste qui-quadrado requer pelo menos duas colunas: uma Y e uma X.")

    col_y = colunas_usadas[0]
    col_x = colunas_usadas[1]
    col_freq = colunas_usadas[2] if len(colunas_usadas) >= 3 else None

    if col_freq and col_freq in df.columns:
        tabela = df.pivot_table(index=col_x, columns=col_y, values=col_freq, aggfunc="sum", fill_value=0)
    else:
        tabela = pd.crosstab(df[col_x], df[col_y])

    chi2, p, dof, expected = chi2_contingency(tabela)

    resumo = f"""🔎 **Qui-Quadrado**
{tabela.to_string()}
Estatística: {chi2:.4f}
GL: {dof}
P-valor: {p:.4f}
"""

    conclusao = "❗ Existe associação." if p < 0.05 else "✅ Não há associação."

    aplicar_estilo_minitab()
    tabela.plot(kind='bar')
    plt.title("Distribuição das Categorias")
    plt.xlabel(col_x)
    plt.ylabel("Frequência")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return resumo + conclusao, imagem_base64


ANALISES = {
    "Regressão linear simples": analise_regressao_linear_simples,
    "Regressão linear múltipla": analise_regressao_linear_multipla,
    "Regressão logística binária": analise_regressao_logistica_binaria,
    "Regressão logística nominal": analise_regressao_logistica_nominal,
    "Regressão logística ordinal": analise_regressao_logistica_ordinal,
    "Qui- quadrado": analise_chi_quadrado
}
