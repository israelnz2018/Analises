from suporte import *

def analise_tipo_modelo_regressao(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) != 2:
        return "❌ O Tipo de modelo de regressão requer 1 Y e 1 X.", None

    y_col, x_col = colunas_usadas
    df_valid = df[[x_col, y_col]].dropna()
    if len(df_valid) < 5:
        return "❌ O teste requer ao menos 5 valores não nulos.", None

    X_raw = df_valid[x_col].values
    Y = df_valid[y_col].values

    resultados = {}

    # Define os modelos
    modelos = {
        "Linear": np.polyfit(X_raw, Y, 1),
        "Quadrático": np.polyfit(X_raw, Y, 2),
        "Cúbico": np.polyfit(X_raw, Y, 3)
    }

    # Logarítmico
    X_log = np.log(X_raw[X_raw > 0])
    Y_log = Y[X_raw > 0]
    if len(X_log) >= 5:
        modelos["Logarítmico"] = np.polyfit(X_log, Y_log, 1)

    # Exponencial (lineariza com log Y)
    Y_exp = Y[Y > 0]
    X_exp = X_raw[Y > 0]
    if len(Y_exp) >= 5:
        modelos["Exponencial"] = np.polyfit(X_exp, np.log(Y_exp), 1)

    # Calcula métricas
    for nome, coef in modelos.items():
        if nome == "Logarítmico":
            y_pred = np.polyval(coef, np.log(X_raw))
        elif nome == "Exponencial":
            y_pred = np.exp(np.polyval(coef, X_raw))
        else:
            y_pred = np.polyval(coef, X_raw)

        ss_res = np.sum((Y - y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        adj = 1 - (1 - r2) * (len(Y) - 1) / (len(Y) - len(coef) - 1)

        # R2 preditivo leave-one-out
        erros = []
        for i in range(len(Y)):
            X_train = np.delete(X_raw, i)
            Y_train = np.delete(Y, i)
            if nome == "Logarítmico":
                X_train_t = np.log(X_train[X_train > 0])
                Y_train_t = Y_train[X_train > 0]
                if len(X_train_t) < 2:
                    continue
                coef_lo = np.polyfit(X_train_t, Y_train_t, 1)
                y_pred_lo = np.polyval(coef_lo, np.log(X_raw[i]))
            elif nome == "Exponencial":
                Y_train_t = Y_train[Y_train > 0]
                X_train_t = X_train[Y_train > 0]
                if len(Y_train_t) < 2:
                    continue
                coef_lo = np.polyfit(X_train_t, np.log(Y_train_t), 1)
                y_pred_lo = np.exp(np.polyval(coef_lo, X_raw[i]))
            else:
                coef_lo = np.polyfit(X_train, Y_train, len(coef) - 1)
                y_pred_lo = np.polyval(coef_lo, X_raw[i])
            erros.append((Y[i] - y_pred_lo) ** 2)

        if len(erros) > 0:
            ss_pred = np.sum(erros)
            r2_pred = 1 - ss_pred / ss_tot
        else:
            r2_pred = float('nan')

        resultados[nome] = {
            "coef": coef,
            "r2": r2,
            "r2_adj": adj,
            "r2_pred": r2_pred
        }

    # Escolher o modelo
    melhor = max(resultados.items(), key=lambda x: x[1]["r2_pred"] if not np.isnan(x[1]["r2_pred"]) else -np.inf)
    nome_vencedor, res_vencedor = melhor

    # Verificar simplicidade
    simples = ["Linear", "Quadrático", "Cúbico", "Logarítmico", "Exponencial"]
    mais_simples = [m for m in simples if m in resultados][:simples.index(nome_vencedor)+1]
    modelo_recomendado = nome_vencedor
    for m in mais_simples:
        if (resultados[nome_vencedor]["r2_pred"] - resultados[m]["r2_pred"]) < 0.01:
            modelo_recomendado = m
            break

    # Gráfico
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(X_raw, Y, color='black', label='Dados')
    if modelo_recomendado == "Logarítmico":
        X_plot = X_raw[X_raw > 0]
        y_fit = np.polyval(resultados["Logarítmico"]["coef"], np.log(X_plot))
    elif modelo_recomendado == "Exponencial":
        y_fit = np.exp(np.polyval(resultados["Exponencial"]["coef"], X_raw))
    else:
        y_fit = np.polyval(resultados[modelo_recomendado]["coef"], X_raw)

    ax.plot(X_raw, y_fit, color='blue', label=f'Modelo: {modelo_recomendado}')
    ax.set_title("Tipo de Modelo de Regressão")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Texto
    linhas = []
    for nome, r in resultados.items():
        coef_txt = " + ".join([f"{c:.4f}*X^{i}" for i, c in enumerate(r["coef"][::-1])])
        linhas.append(f"- {nome}: R²={r['r2']:.4f}, R² ajustado={r['r2_adj']:.4f}, R² preditivo={r['r2_pred']:.4f}\n  Equação: {coef_txt}")

    texto = f"""
**Tipo de Modelo de Regressão**
{chr(10).join(linhas)}

**Conclusão**
Modelo recomendado: {modelo_recomendado}
"""

    return texto.strip(), grafico_base64

def analise_regressao_linear_simples(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) != 2:
        return "❌ A regressão linear simples requer 1 Y e 1 X.", None

    y_col, x_col = colunas_usadas
    df_valid = df[[x_col, y_col]].dropna()

    if len(df_valid) < 5:
        return "❌ O modelo requer ao menos 5 observações válidas.", None

    X = df_valid[x_col].values
    Y = df_valid[y_col].values

    # Ajuste do modelo
    coef = np.polyfit(X, Y, 1)
    y_pred = np.polyval(coef, X)
    intercepto, angular = coef[1], coef[0]

    ss_res = np.sum((Y - y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (len(Y) - 1) / (len(Y) - 2)

    # p-valor do coeficiente
    X_mat = np.vstack([np.ones_like(X), X]).T
    beta, residuals, rank, s = np.linalg.lstsq(X_mat, Y, rcond=None)
    mse = ss_res / (len(Y) - 2)
    var_beta = mse * np.linalg.inv(X_mat.T @ X_mat)
    se_beta1 = np.sqrt(var_beta[1,1])
    t_stat = beta[1] / se_beta1
    p_valor_beta1 = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(Y)-2))

    # R2 preditivo leave-one-out
    erros = []
    for i in range(len(Y)):
        X_train = np.delete(X, i)
        Y_train = np.delete(Y, i)
        coef_lo = np.polyfit(X_train, Y_train, 1)
        y_pred_lo = np.polyval(coef_lo, X[i])
        erros.append((Y[i] - y_pred_lo) ** 2)
    ss_pred = np.sum(erros)
    r2_pred = 1 - ss_pred / ss_tot

    # Normalidade dos resíduos
    residuos = Y - y_pred
    ad = stats.anderson(residuos)
    ad_crit = ad.critical_values
    ad_sig = list(ad.significance_level)
    if 5 in ad_sig:
        idx = ad_sig.index(5)
        ad_normal = ad.statistic < ad_crit[idx]
    else:
        ad_normal = False

    # Conclusão
    conclusao = []
    conclusao.append("✅ Resíduos seguem distribuição normal (Anderson-Darling)." if ad_normal else "⚠ Resíduos podem não ser normais (Anderson-Darling).")
    conclusao.append(f"✅ Coeficiente angular significativo (p = {p_valor_beta1:.4f})." if p_valor_beta1 < 0.05 else f"⚠ Coeficiente angular não significativo (p = {p_valor_beta1:.4f}).")
    if ad_normal and p_valor_beta1 < 0.05 and r2_pred > 0.5:
        conclusao.append("✅ Modelo validado.")
    else:
        conclusao.append("⚠ Modelo pode não ser adequado. Verifique os indicadores acima.")

    # Gráfico
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(X, Y, color='black', label='Dados')
    ax.plot(X, y_pred, color='blue', label='Reta ajustada')
    ax.set_title("Regressão Linear Simples")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Texto
    texto = f"""
**Regressão Linear Simples**
- Equação: Y = {intercepto:.4f} + {angular:.4f} * X
- R²: {r2:.4f}
- R² ajustado: {r2_adj:.4f}
- R² preditivo: {r2_pred:.4f}
- p-valor do coeficiente angular: {p_valor_beta1:.4f}
- Anderson-Darling dos resíduos: estatística={ad.statistic:.4f}, normalidade={'Aprovada' if ad_normal else 'Reprovada'}

**Conclusão**
{chr(10).join(conclusao)}
"""

    return texto.strip(), grafico_base64




def analise_regressao_linear_multipla(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) < 2:
        return "❌ A regressão linear múltipla requer 1 Y e pelo menos 1 X.", None

    y_col = colunas_usadas[0]
    xs_continuo = [c for c in colunas_usadas[1:] if c not in ["", "Xs_discreto"]]
    xs_discreto = []
    if "Xs_discreto" in colunas_usadas:
        idx = colunas_usadas.index("Xs_discreto")
        xs_discreto = colunas_usadas[idx+1:]

    # Validação
    cols_todas = [y_col] + xs_continuo + xs_discreto
    df_valid = df[cols_todas].dropna()
    if len(df_valid) < len(xs_continuo) + len(xs_discreto) + 3:
        return "❌ O modelo requer mais dados válidos.", None

    Y = df_valid[y_col].values
    X_cont = df_valid[xs_continuo] if xs_continuo else pd.DataFrame(index=df_valid.index)
    X_disc = pd.get_dummies(df_valid[xs_discreto], drop_first=True) if xs_discreto else pd.DataFrame(index=df_valid.index)
    X_final = pd.concat([X_cont, X_disc], axis=1)
    x_cols_final = X_final.columns.tolist()
    n, p_full = X_final.shape

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    from itertools import combinations

    def calcular_modelo(X_sub, cols_sub):
        model = LinearRegression().fit(X_sub, Y)
        Y_pred = model.predict(X_sub)
        r2 = r2_score(Y, Y_pred)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - X_sub.shape[1] - 1)

        # R² preditivo leave-one-out
        erros = []
        for i in range(n):
            Xi = np.delete(X_sub, i, axis=0)
            Yi = np.delete(Y, i)
            m_lo = LinearRegression().fit(Xi, Yi)
            y_lo = m_lo.predict(X_sub[i].reshape(1, -1))[0]
            erros.append((Y[i] - y_lo) ** 2)
        ss_res_pred = np.sum(erros)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r2_pred = 1 - ss_res_pred / ss_tot

        # VIF
        vif = []
        if X_sub.shape[1] > 1:
            X_sm = sm.add_constant(X_sub)
            for i in range(1, X_sm.shape[1]):  # ignora const
                vif.append(variance_inflation_factor(X_sm, i))
        else:
            vif.append(1.0)

        # Mallows Cp
        resid = Y - Y_pred
        mse_full = np.sum((Y - LinearRegression().fit(X_final, Y).predict(X_final)) ** 2) / (n - p_full - 1)
        cp = (np.sum(resid ** 2) / mse_full) - (n - 2 * (X_sub.shape[1] + 1))

        # Durbin-Watson
        dw = sm.stats.stattools.durbin_watson(resid)

        return {
            "modelo": model,
            "cols": cols_sub,
            "r2": r2,
            "r2_adj": r2_adj,
            "r2_pred": r2_pred,
            "vif": vif,
            "cp": cp,
            "dw": dw,
            "Y_pred": Y_pred
        }

    # Subset selection (limite de 5 para praticidade)
    resultados = []
    if len(x_cols_final) > 5:
        x_cols_final = x_cols_final[:5]  # limita para evitar sobrecarga

    for k in range(1, len(x_cols_final) + 1):
        for subset in combinations(range(len(x_cols_final)), k):
            cols_sub = [x_cols_final[i] for i in subset]
            X_sub = X_final[cols_sub].values
            resultados.append(calcular_modelo(X_sub, cols_sub))

    melhor = max(resultados, key=lambda r: r["r2_pred"])
    simples = sorted(resultados, key=lambda r: len(r["cols"]))
    modelo_recomendado = melhor
    for m in simples:
        if (melhor["r2_pred"] - m["r2_pred"]) < 0.01:
            modelo_recomendado = m
            break

    # Gráfico resíduos vs preditos
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(modelo_recomendado["Y_pred"], Y - modelo_recomendado["Y_pred"], color='black')
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Valores preditos")
    ax.set_ylabel("Resíduos")
    ax.set_title("Resíduos vs Valores Preditos")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Texto
    linhas = [f"Modelo recomendado: {', '.join(modelo_recomendado['cols'])}"]
    linhas.append(f"R²: {modelo_recomendado['r2']:.4f}")
    linhas.append(f"R² ajustado: {modelo_recomendado['r2_adj']:.4f}")
    linhas.append(f"R² preditivo: {modelo_recomendado['r2_pred']:.4f}")
    linhas.append(f"Mallows Cp: {modelo_recomendado['cp']:.4f}")
    linhas.append(f"Durbin-Watson: {modelo_recomendado['dw']:.4f}")
    linhas.append("VIFs: " + ", ".join([f"{c}={v:.2f}" for c, v in zip(modelo_recomendado['cols'], modelo_recomendado['vif'])]))

    conclusao = []
    if modelo_recomendado['r2_pred'] > 0.5:
        conclusao.append("✅ R² preditivo adequado.")
    else:
        conclusao.append("⚠ R² preditivo baixo.")
    if all(v < 10 for v in modelo_recomendado['vif']):
        conclusao.append("✅ Sem multicolinearidade severa (VIF < 10).")
    else:
        conclusao.append("⚠ Multicolinearidade identificada (VIF >= 10).")

    texto = f"""
**Regressão Linear Múltipla**
{chr(10).join(linhas)}

**Conclusão**
{chr(10).join(conclusao)}
"""

    return texto.strip(), grafico_base64


def analise_regressao_logistica_binaria(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) < 2:
        return "❌ A regressão logística binária requer 1 Y e pelo menos 1 X.", None

    y_col = colunas_usadas[0]
    xs_continuo = [c for c in colunas_usadas[1:] if c not in ["", "Xs_discreto"]]
    xs_discreto = []
    if "Xs_discreto" in colunas_usadas:
        idx = colunas_usadas.index("Xs_discreto")
        xs_discreto = colunas_usadas[idx+1:]

    # Validação
    cols_todas = [y_col] + xs_continuo + xs_discreto
    df_valid = df[cols_todas].dropna()
    if len(df_valid) < len(xs_continuo) + len(xs_discreto) + 3:
        return "❌ O modelo requer mais dados válidos.", None

    Y = df_valid[y_col].values
    if len(set(Y)) != 2:
        return "❌ A variável Y deve ser binária (duas categorias).", None

    X_cont = df_valid[xs_continuo] if xs_continuo else pd.DataFrame(index=df_valid.index)
    X_disc = pd.get_dummies(df_valid[xs_discreto], drop_first=True) if xs_discreto else pd.DataFrame(index=df_valid.index)
    X_final = pd.concat([X_cont, X_disc], axis=1)
    x_cols_final = X_final.columns.tolist()

    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X_sm = sm.add_constant(X_final)
    model = sm.Logit(Y, X_sm).fit(disp=0)
    Y_pred_prob = model.predict(X_sm)
    Y_pred_class = (Y_pred_prob >= 0.5).astype(int)

    # Pseudo R²
    ll_null = sm.Logit(Y, np.ones((len(Y),1))).fit(disp=0).llf
    ll_model = model.llf
    r2_mcf = 1 - ll_model / ll_null

    # VIF
    vif = []
    for i in range(1, X_sm.shape[1]):  # ignora const
        vif.append(variance_inflation_factor(X_sm.values, i))

    # AUC
    auc = roc_auc_score(Y, Y_pred_prob)

    # Matriz de confusão
    cm = confusion_matrix(Y, Y_pred_class)
    acerto = (cm.diagonal().sum()) / cm.sum()

    # Gráfico
    fig, ax = plt.subplots(figsize=(6,4))
    if len(x_cols_final) == 1 and xs_continuo:
        X_plot = np.linspace(X_final.iloc[:,0].min(), X_final.iloc[:,0].max(), 100)
        X_plot_sm = sm.add_constant(X_plot)
        if X_plot_sm.ndim == 1:
            X_plot_sm = X_plot_sm.reshape(-1,1)
        coef = model.params.values
        logit = coef[0] + coef[1] * X_plot
        prob = 1 / (1 + np.exp(-logit))
        ax.scatter(X_final.iloc[:,0], Y, color='black', alpha=0.5, label='Dados')
        ax.plot(X_plot, prob, color='blue', label='Curva ajustada')
        ax.set_xlabel(x_cols_final[0])
        ax.set_ylabel('Probabilidade')
        ax.set_title('Regressão Logística Binária')
        ax.legend()
    else:
        fpr, tpr, _ = roc_curve(Y, Y_pred_prob)
        ax.plot(fpr, tpr, color='blue', label=f'ROC AUC = {auc:.2f}')
        ax.plot([0,1], [0,1], color='grey', linestyle='--')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('Curva ROC')
        ax.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Texto
    linhas = []
    for name, coef, pval in zip(['Const'] + x_cols_final, model.params, model.pvalues):
        linhas.append(f"- {name}: coef={coef:.4f}, p-valor={pval:.4f}")
    linhas.append(f"R² de McFadden: {r2_mcf:.4f}")
    linhas.append(f"AUC: {auc:.4f}")
    linhas.append(f"Percentual de acerto: {acerto*100:.2f}%")
    linhas.append("VIFs: " + ", ".join([f"{c}={v:.2f}" for c,v in zip(x_cols_final, vif)]))

    conclusao = []
    if r2_mcf > 0.2:
        conclusao.append("✅ Modelo apresenta bom ajuste (R² de McFadden adequado).")
    else:
        conclusao.append("⚠ R² de McFadden baixo, modelo pode não ter bom ajuste.")

    if auc > 0.7:
        conclusao.append("✅ Capacidade discriminativa aceitável (AUC > 0.7).")
    else:
        conclusao.append("⚠ Capacidade discriminativa baixa (AUC <= 0.7).")

    if all(v < 10 for v in vif):
        conclusao.append("✅ Sem multicolinearidade severa (VIF < 10).")
    else:
        conclusao.append("⚠ Multicolinearidade identificada (VIF >= 10).")

    texto = f"""
**Regressão Logística Binária**
{chr(10).join(linhas)}

**Conclusão**
{chr(10).join(conclusao)}
"""

    return texto.strip(), grafico_base64



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
    "Tipo de modelo de regressão": analise_tipo_modelo_regressao,
    "Regressão linear simples": analise_regressao_linear_simples,
    "Regressão linear múltipla": analise_regressao_linear_multipla,
    "Regressão logística binária": analise_regressao_logistica_binaria



    

   
    "Regressão logística nominal": analise_regressao_logistica_nominal,
    "Regressão logística ordinal": analise_regressao_logistica_ordinal,
    "Qui- quadrado": analise_chi_quadrado
}
