from suporte import *

def analise_melhor_modelo(df: pd.DataFrame, coluna_y, coluna_x):
    import statsmodels.api as sm
    from sklearn.metrics import r2_score

    if not coluna_y or not coluna_x:
        return "❌ O teste requer exatamente 1 coluna Y e 1 coluna X.", None

    if coluna_y not in df.columns or coluna_x not in df.columns:
        return "❌ As colunas informadas não foram encontradas no arquivo.", None

    df_valid = df[[coluna_x, coluna_y]].dropna()
    if df_valid.empty:
        return "❌ Não há dados suficientes após remoção de valores nulos.", None

    x = df_valid[coluna_x].to_numpy()
    y = df_valid[coluna_y].to_numpy()

    resultados = []
    detalhes_modelos = ""

    # 1. Modelo Linear
    X_lin = sm.add_constant(x)
    modelo_lin = sm.OLS(y, X_lin).fit()
    r2_lin = modelo_lin.rsquared
    eq_lin = f"Y = {modelo_lin.params[0]:.3f} + {modelo_lin.params[1]:.3f}X"
    resultados.append(("Linear", r2_lin, modelo_lin, eq_lin))

    # 2. Modelo Log-linear
    if all(y > 0):
        modelo_loglin = sm.OLS(np.log(y), X_lin).fit()
        y_pred_loglin = np.exp(modelo_loglin.predict(X_lin))
        r2_loglin = r2_score(y, y_pred_loglin)
        eq_loglin = f"ln(Y) = {modelo_loglin.params[0]:.3f} + {modelo_loglin.params[1]:.3f}X"
        resultados.append(("Log-linear", r2_loglin, modelo_loglin, eq_loglin))
    
    # 3. Modelo Exponencial
    if all(y > 0):
        modelo_exp = sm.OLS(np.log(y), X_lin).fit()
        y_pred_exp = np.exp(modelo_exp.predict(X_lin))
        r2_exp = r2_score(y, y_pred_exp)
        eq_exp = f"Y = exp({modelo_exp.params[0]:.3f} + {modelo_exp.params[1]:.3f}X)"
        resultados.append(("Exponencial", r2_exp, modelo_exp, eq_exp))

    # 4. Modelo Potência
    if all(x > 0) and all(y > 0):
        X_pot = sm.add_constant(np.log(x))
        modelo_pot = sm.OLS(np.log(y), X_pot).fit()
        y_pred_pot = np.exp(modelo_pot.predict(X_pot))
        r2_pot = r2_score(y, y_pred_pot)
        eq_pot = f"Y = exp({modelo_pot.params[0]:.3f}) * X^{modelo_pot.params[1]:.3f}"
        resultados.append(("Potência", r2_pot, modelo_pot, eq_pot))

    # 5. Modelo Polinomial grau 2
    X_poly = np.column_stack((x, x**2))
    X_poly = sm.add_constant(X_poly)
    modelo_poly = sm.OLS(y, X_poly).fit()
    r2_poly = modelo_poly.rsquared
    eq_poly = f"Y = {modelo_poly.params[0]:.3f} + {modelo_poly.params[1]:.3f}X + {modelo_poly.params[2]:.3f}X²"
    resultados.append(("Polinomial", r2_poly, modelo_poly, eq_poly))

    # Monta texto com todos os modelos
    for nome, r2, _, eq in resultados:
        detalhes_modelos += f"🔹 {nome}\n- Equação: {eq}\n- R² = {r2:.3f}\n\n"

    # Ordena por R² decrescente
    resultados.sort(key=lambda x: x[1], reverse=True)

    # Critério de desempate
    melhor = resultados[0]
    for modelo in resultados[1:]:
        if abs(melhor[1] - modelo[1]) < 0.05:
            # Se diferença <5%, escolhe modelo mais simples (pela ordem original)
            ordem_simples = ["Linear", "Log-linear", "Exponencial", "Potência", "Polinomial"]
            if ordem_simples.index(modelo[0]) < ordem_simples.index(melhor[0]):
                melhor = modelo

    tipo_modelo, r2_melhor, modelo_melhor, eq_melhor = melhor

    # Calcula y_pred para o modelo vencedor
    if tipo_modelo == "Linear":
        y_pred = modelo_melhor.predict(sm.add_constant(x))
    elif tipo_modelo == "Log-linear":
        y_pred = np.exp(modelo_melhor.predict(sm.add_constant(x)))
    elif tipo_modelo == "Exponencial":
        y_pred = np.exp(modelo_melhor.predict(sm.add_constant(x)))
    elif tipo_modelo == "Potência":
        y_pred = np.exp(modelo_melhor.predict(sm.add_constant(np.log(x))))
    elif tipo_modelo == "Polinomial":
        y_pred = modelo_melhor.predict(X_poly)
    else:
        y_pred = np.zeros_like(y)

    # Gráfico do modelo vencedor
    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(x, y, label="Observado", color="#0072B2")
    ax.plot(x, y_pred, label="Modelo Ajustado", color="#E69F00")
    ax.set_title(eq_melhor)
    ax.set_xlabel(coluna_x)
    ax.set_ylabel(coluna_y)
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Alerta se R² < 50%
    alerta = ""
    if r2_melhor < 0.50:
        alerta = "⚠️ **Atenção:** O modelo possui baixo poder explicativo (R² < 50%). Recomenda-se adicionar mais fatores (variáveis X) para melhorar o ajuste.\n"

    # Report padronizado
    texto = f"""
📊 **Análise – Melhor Modelo de Regressão**

🔎 **Resultados de todos os modelos:**
{detalhes_modelos}

🔎 **Modelo escolhido:**
- Tipo: {tipo_modelo}
- Equação: {eq_melhor}
- R² = {r2_melhor:.3f}

🔎 **Justificativa:**
O modelo {tipo_modelo} foi selecionado por apresentar o maior R² considerando simplicidade como critério secundário.

{alerta}
"""

    return texto.strip(), grafico_base64




def analise_regressao_linear_simples(df: pd.DataFrame, coluna_y, coluna_x):
    if not coluna_y or not coluna_x:
        return "❌ A regressão linear simples requer 1 coluna Y e 1 coluna X.", None

    if coluna_y not in df.columns or coluna_x not in df.columns:
        return f"❌ Coluna {coluna_y} ou {coluna_x} não encontrada no arquivo.", None

    df_valid = df[[coluna_x, coluna_y]].dropna()
    if len(df_valid) < 5:
        return "❌ O modelo requer ao menos 5 observações válidas.", None

    X = df_valid[coluna_x].values
    Y = df_valid[coluna_y].values

    coef = np.polyfit(X, Y, 1)
    y_pred = np.polyval(coef, X)
    intercepto, angular = coef[1], coef[0]

    ss_res = np.sum((Y - y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    n = len(Y)
    p = 2
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p)

    # R² preditivo (Leave-One-Out)
    erros = []
    for i in range(n):
        X_train = np.delete(X, i)
        Y_train = np.delete(Y, i)
        coef_lo = np.polyfit(X_train, Y_train, 1)
        y_pred_lo = np.polyval(coef_lo, X[i])
        erros.append((Y[i] - y_pred_lo) ** 2)
    ss_pred = np.sum(erros)
    r2_pred = 1 - ss_pred / ss_tot

    # p-valor do coeficiente angular
    X_mat = np.vstack([np.ones_like(X), X]).T
    beta, residuals, rank, s = np.linalg.lstsq(X_mat, Y, rcond=None)
    mse = ss_res / (n - p)
    var_beta = mse * np.linalg.inv(X_mat.T @ X_mat)
    se_beta1 = np.sqrt(var_beta[1,1])
    t_stat = beta[1] / se_beta1
    p_valor_beta1 = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-p))

    # Testes de normalidade dos resíduos
    residuos = Y - y_pred
    ad = stats.anderson(residuos)
    sw_stat, sw_p = stats.shapiro(residuos)
    dp_stat, dp_p = stats.normaltest(residuos)

    testes_normalidade = {
        "Anderson-Darling": (ad.statistic, ad.statistic < ad.critical_values[2], 0.05),
        "Shapiro-Wilk": (sw_p, sw_p > 0.05, sw_p),
        "D’Agostino-Pearson": (dp_p, dp_p > 0.05, dp_p)
    }
    melhor_teste = max(testes_normalidade.items(), key=lambda x: x[1][2])
    nome_teste, (stat, aprovado, p_valor) = melhor_teste
    normalidade_residuos = aprovado

    # Conclusão baseada em critérios
    motivos = []
    if not normalidade_residuos:
        motivos.append("os resíduos não são normais")
    if p_valor_beta1 >= 0.05:
        motivos.append("o coeficiente angular não é significativo (p-valor >= 0.05)")
    if r2 < 0.5:
        motivos.append("o R² é inferior a 50%")

    if not motivos:
        validacao = "✅ Modelo validado. O modelo linear é adequado para os dados."
    else:
        motivos_txt = "; ".join(motivos)
        validacao = f"⚠️ Modelo não validado porque {motivos_txt}."

    # Recomendação prática
    recomendacoes = []
    if r2 < 0.5:
        recomendacoes.append("➔ O R² está abaixo de 50%. Considere adicionar mais variáveis (Xs) ou testar outro tipo de modelo.")
    if not normalidade_residuos:
        recomendacoes.append("➔ Os resíduos não são normais. Verifique a estabilidade do processo ou colete mais dados.")
    if p_valor_beta1 >= 0.05:
        recomendacoes.append("➔ O coeficiente angular não é significativo. Avalie a relação entre as variáveis ou considere outro modelo.")

    recomendacao_final = "\n".join(recomendacoes)

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(X, Y, color='black', label='Dados')
    ax.plot(X, y_pred, color='blue', label='Reta ajustada')
    ax.set_title("Regressão Linear Simples")
    ax.set_xlabel(coluna_x)
    ax.set_ylabel(coluna_y)
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
📊 **Análise – Regressão Linear Simples**

🔹 **Hipóteses do modelo**
- **H₀:** Não há relação linear entre {coluna_x} e {coluna_y}
- **H₁:** Existe relação linear entre {coluna_x} e {coluna_y}

🔎 **Resumo do modelo**
- Equação: Y = {intercepto:.4f} + {angular:.4f} * X
- R²: {r2:.4f}
- R² ajustado: {r2_adj:.4f}
- R² preditivo: {r2_pred:.4f}
- p-valor do coeficiente angular: {p_valor_beta1:.4f}

🔎 **Normalidade dos resíduos**
{"✅ Os resíduos podem ser considerados normais" if normalidade_residuos else f"❌ Os resíduos não são normais (p = {p_valor:.4f}, {nome_teste})."}

🔎 **Conclusão**
{validacao}

🔧 **Recomendação**
{recomendacao_final}
"""

    return texto.strip(), grafico_base64



def analise_regressao_quadratica(df: pd.DataFrame, coluna_y, coluna_x):
    if not coluna_y or not coluna_x:
        return "❌ A regressão quadrática requer 1 coluna Y e 1 coluna X.", None

    if coluna_y not in df.columns or coluna_x not in df.columns:
        return f"❌ Coluna {coluna_y} ou {coluna_x} não encontrada no arquivo.", None

    df_valid = df[[coluna_x, coluna_y]].dropna()
    if len(df_valid) < 5:
        return "❌ O modelo requer ao menos 5 observações válidas.", None

    X = df_valid[coluna_x].values
    Y = df_valid[coluna_y].values

    # Ajuste quadrático
    coef = np.polyfit(X, Y, 2)
    y_pred = np.polyval(coef, X)
    a, b, c = coef[0], coef[1], coef[2]

    ss_res = np.sum((Y - y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    n = len(Y)
    p = 3
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p)

    # R² preditivo (Leave-One-Out)
    erros = []
    for i in range(n):
        X_train = np.delete(X, i)
        Y_train = np.delete(Y, i)
        coef_lo = np.polyfit(X_train, Y_train, 2)
        y_pred_lo = np.polyval(coef_lo, X[i])
        erros.append((Y[i] - y_pred_lo) ** 2)
    ss_pred = np.sum(erros)
    r2_pred = 1 - ss_pred / ss_tot

    # p-valor do modelo (F-test)
    msr = (ss_tot - ss_res) / (p - 1)
    mse = ss_res / (n - p)
    f_stat = msr / mse
    p_valor_modelo = 1 - stats.f.cdf(f_stat, p - 1, n - p)

    # Testes de normalidade dos resíduos
    residuos = Y - y_pred
    ad = stats.anderson(residuos)
    sw_stat, sw_p = stats.shapiro(residuos)
    dp_stat, dp_p = stats.normaltest(residuos)

    testes_normalidade = {
        "Anderson-Darling": (ad.statistic, ad.statistic < ad.critical_values[2], 0.05),
        "Shapiro-Wilk": (sw_p, sw_p > 0.05, sw_p),
        "D’Agostino-Pearson": (dp_p, dp_p > 0.05, dp_p)
    }
    melhor_teste = max(testes_normalidade.items(), key=lambda x: x[1][2])
    nome_teste, (stat, aprovado, p_valor) = melhor_teste
    normalidade_residuos = aprovado

    # Conclusão baseada em critérios
    motivos = []
    if not normalidade_residuos:
        motivos.append("os resíduos não são normais")
    if p_valor_modelo >= 0.05:
        motivos.append("o modelo não é estatisticamente significativo (p-valor >= 0.05)")
    if r2 < 0.5:
        motivos.append("o R² é inferior a 50%")

    if not motivos:
        validacao = "✅ Modelo validado. O modelo quadrático é adequado para os dados."
    else:
        motivos_txt = "; ".join(motivos)
        validacao = f"⚠️ Modelo não validado porque {motivos_txt}."

    # Recomendação prática (independente do R²)
    recomendacoes = []
    if r2 < 0.5:
        recomendacoes.append("➔ O R² está abaixo de 50%. Considere adicionar mais variáveis (Xs) ou testar outro tipo de modelo para melhorar a capacidade preditiva.")
    if not normalidade_residuos:
        recomendacoes.append("➔ Os resíduos não são normais. Recomenda-se verificar a estabilidade do processo ou coletar mais dados.")
    if p_valor_modelo >= 0.05:
        recomendacoes.append("➔ O modelo não é estatisticamente significativo. Avalie se há relação entre as variáveis ou considere outro tipo de modelo.")

    recomendacao_final = "\n".join(recomendacoes)

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(X, Y, color='black', label='Dados')

    # Curva quadrática
    x_seq = np.linspace(X.min(), X.max(), 300)
    y_seq = np.polyval(coef, x_seq)
    ax.plot(x_seq, y_seq, color='blue', label='Curva ajustada')

    ax.set_title("Regressão Quadrática")
    ax.set_xlabel(coluna_x)
    ax.set_ylabel(coluna_y)
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
📊 **Análise – Regressão Quadrática**

🔹 **Hipóteses do modelo**
- **H₀:** Não há relação quadrática entre {coluna_x} e {coluna_y}
- **H₁:** Existe relação quadrática entre {coluna_x} e {coluna_y}

🔎 **Resumo do modelo**
- Equação: Y = {a:.4f}X² + {b:.4f}X + {c:.4f}
- R²: {r2:.4f}
- R² ajustado: {r2_adj:.4f}
- R² preditivo: {r2_pred:.4f}
- p-valor do modelo quadrático: {p_valor_modelo:.4f}

🔎 **Normalidade dos resíduos**
{"✅ Os resíduos podem ser considerados normais" if normalidade_residuos else f"❌ Os resíduos não são normais (p = {p_valor:.4f}, {nome_teste})."}

🔎 **Conclusão**
{validacao}

🔧 **Recomendação**
{recomendacao_final}
"""

    return texto.strip(), grafico_base64


def analise_regressao_cubica(df: pd.DataFrame, coluna_y, coluna_x):
    if not coluna_y or not coluna_x:
        return "❌ A regressão cúbica requer 1 coluna Y e 1 coluna X.", None

    if coluna_y not in df.columns or coluna_x not in df.columns:
        return f"❌ Coluna {coluna_y} ou {coluna_x} não encontrada no arquivo.", None

    df_valid = df[[coluna_x, coluna_y]].dropna()
    if len(df_valid) < 5:
        return "❌ O modelo requer ao menos 5 observações válidas.", None

    X = df_valid[coluna_x].values
    Y = df_valid[coluna_y].values

    # Ajuste cúbico (3º grau)
    coef = np.polyfit(X, Y, 3)
    y_pred = np.polyval(coef, X)
    a, b, c, d = coef[0], coef[1], coef[2], coef[3]

    ss_res = np.sum((Y - y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    n = len(Y)
    p = 4  # parâmetros a, b, c, d
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p)

    # R² preditivo (Leave-One-Out)
    erros = []
    for i in range(n):
        X_train = np.delete(X, i)
        Y_train = np.delete(Y, i)
        coef_lo = np.polyfit(X_train, Y_train, 3)
        y_pred_lo = np.polyval(coef_lo, X[i])
        erros.append((Y[i] - y_pred_lo) ** 2)
    ss_pred = np.sum(erros)
    r2_pred = 1 - ss_pred / ss_tot

    # p-valor do modelo (F-test)
    msr = (ss_tot - ss_res) / (p - 1)
    mse = ss_res / (n - p)
    f_stat = msr / mse
    p_valor_modelo = 1 - stats.f.cdf(f_stat, p - 1, n - p)

    # Testes de normalidade dos resíduos
    residuos = Y - y_pred

    ad = stats.anderson(residuos)
    sw_stat, sw_p = stats.shapiro(residuos)
    dp_stat, dp_p = stats.normaltest(residuos)

    # Melhor teste (maior p-valor)
    testes_normalidade = {
        "Anderson-Darling": (ad.statistic, ad.statistic < ad.critical_values[2], 0.05),
        "Shapiro-Wilk": (sw_p, sw_p > 0.05, sw_p),
        "D’Agostino-Pearson": (dp_p, dp_p > 0.05, dp_p)
    }
    melhor_teste = max(testes_normalidade.items(), key=lambda x: x[1][2])
    nome_teste, (stat, aprovado, p_valor) = melhor_teste

    normalidade_residuos = aprovado

    # Conclusão com justificativas
    motivos = []
    if not normalidade_residuos:
        motivos.append("os resíduos não são normais")
    if p_valor_modelo >= 0.05:
        motivos.append("o modelo não é estatisticamente significativo (p-valor >= 0.05)")
    if r2 < 0.5:
        motivos.append("o R² é inferior a 50%")

    if not motivos:
        validacao = "✅ Modelo validado. O modelo cúbico é adequado para os dados."
    else:
        motivos_txt = "; ".join(motivos)
        validacao = f"⚠️ Modelo não validado porque {motivos_txt}."

    conclusao_normalidade = f"✅ Os resíduos podem ser considerados normais (p = {p_valor:.4f}, {nome_teste})." if normalidade_residuos else f"❌ Os resíduos não são normais (p = {p_valor:.4f}, {nome_teste}). Recomenda-se verificar a estabilidade do processo ou coletar mais dados."

    # Recomendação apenas se R² < 50%
    recomendacao = ""
    if r2 < 0.5:
        recomendacao = "🔧 **Recomendação**\n➔ O R² está abaixo de 50%. Considere adicionar mais variáveis (Xs) ou testar outro tipo de modelo para melhorar a capacidade preditiva."

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(X, Y, color='black', label='Dados')

    # Curva cúbica
    x_seq = np.linspace(X.min(), X.max(), 300)
    y_seq = np.polyval(coef, x_seq)
    ax.plot(x_seq, y_seq, color='blue', label='Curva ajustada')

    ax.set_title("Regressão Cúbica")
    ax.set_xlabel(coluna_x)
    ax.set_ylabel(coluna_y)
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
📊 **Análise – Regressão Cúbica**

🔹 **Hipóteses do modelo**
- **H₀:** Não há relação cúbica entre {coluna_x} e {coluna_y}
- **H₁:** Existe relação cúbica entre {coluna_x} e {coluna_y}

🔎 **Resumo do modelo**
- Equação: Y = {a:.4f}X³ + {b:.4f}X² + {c:.4f}X + {d:.4f}
- R²: {r2:.4f}
- R² ajustado: {r2_adj:.4f}
- R² preditivo: {r2_pred:.4f}
- p-valor do modelo cúbico: {p_valor_modelo:.4f}

🔎 **Normalidade dos resíduos**
{conclusao_normalidade}

🔎 **Conclusão**
{validacao}

{recomendacao}
"""

    return texto.strip(), grafico_base64

def analise_regressao_linear_multipla(df, coluna_y, lista_x):
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from scipy import stats
    from io import BytesIO
    import base64

    # Validação inicial
    if not coluna_y or not lista_x:
        return "❌ A regressão linear múltipla requer 1 Y e pelo menos 1 X.", None

    for col in [coluna_y] + lista_x:
        if col not in df.columns:
            return f"❌ Coluna '{col}' não encontrada no conjunto de dados.", None

    df_valid = df[[coluna_y] + lista_x].dropna()
    if len(df_valid) < len(lista_x) + 3:
        return "❌ O modelo requer mais dados válidos.", None

    # Preparação dos dados
    Y = df_valid[coluna_y].values
    X = pd.get_dummies(df_valid[lista_x], drop_first=True)
    X_cols = X.columns.tolist()
    X_values = X.values
    n, p = X_values.shape

    # Regressão com statsmodels
    X_sm = sm.add_constant(X)
    model = sm.OLS(Y, X_sm).fit()
    Y_pred = model.predict(X_sm)

    # Métricas
    r2 = model.rsquared
    r2_adj = model.rsquared_adj
    p_global = model.f_pvalue
    p_individuais = model.pvalues[1:]
    coeficientes = model.params[1:]
    intercepto = model.params[0]

    # R² preditivo (Leave-One-Out)
    r2_pred = 1 - np.sum([
        (Y[i] - LinearRegression().fit(np.delete(X_values, i, axis=0), np.delete(Y, i)).predict(X_values[i].reshape(1, -1))[0]) ** 2
        for i in range(n)
    ]) / np.sum((Y - np.mean(Y)) ** 2)

    # Cp de Mallows
    mse_full = np.sum((Y - Y_pred) ** 2) / (n - p - 1)
    cp = (np.sum((Y - Y_pred) ** 2) / mse_full) - (n - 2 * (p + 1))

    # VIF
    vif_list = [variance_inflation_factor(X_values, i) for i in range(p)]

    # Resíduos
    residuos = Y - Y_pred
    dw = sm.stats.stattools.durbin_watson(residuos)
    media_residuos = np.mean(residuos)
    p_normalidade = stats.shapiro(residuos).pvalue
    p_heterocedasticidade = sm.stats.diagnostic.het_breuschpagan(residuos, X_sm)[1]

    # Gráfico dos resíduos
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(Y_pred, residuos, color='black')
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Valores preditos")
    ax.set_ylabel("Resíduos")
    ax.set_title("Resíduos vs Valores Preditos")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Equação do modelo
    equacao = f"{coluna_y} = {intercepto:.2f} + " + " + ".join(
        [f"{coef:.2f}·{col}" for coef, col in zip(coeficientes, X_cols)]
    )

    # Strings de resultados
    p_ind_str = " | ".join([f"{col} = {pval:.4f}" for col, pval in zip(X_cols, p_individuais)])
    vif_str = " | ".join([f"{col} = {v:.2f}" for col, v in zip(X_cols, vif_list)])

    # Critérios obrigatórios (para predição)
    criticos = [
        f"R² preditivo (mínimo 0,50): {r2_pred:.3f} {'✅' if r2_pred >= 0.5 else '❌'}",
        f"p-valor global do modelo (< 0,05): {p_global:.4f} {'✅' if p_global < 0.05 else '❌'}",
        f"Durbin-Watson - Independência do resíduos (entre 1,5 e 2,5): {dw:.3f} {'✅' if 1.5 <= dw <= 2.5 else '❌'}"
    ]

    # Critérios recomendados
    recomendados = [
        f"p-valores individuais (< 0,05): {p_ind_str} {'✅' if all(p < 0.05 for p in p_individuais) else '⚠️'}",
        f"R² e R² ajustado: {r2:.3f} | {r2_adj:.3f} ✅",
        f"Δ R² ajustado vs R² preditivo (< 0,05): {abs(r2_adj - r2_pred):.3f} {'✅' if abs(r2_adj - r2_pred) < 0.05 else '⚠️'}",
        f"Cp de Mallows ≈ p+1: {cp:.2f} {'✅' if abs(cp - (p + 1)) <= 1 else '⚠️'}",
        f"Normalidade dos resíduos (p > 0,05): {p_normalidade:.4f} {'✅' if p_normalidade > 0.05 else '⚠️'}",
        f"Homocedasticidade dos resíduos (p > 0,05): {p_heterocedasticidade:.4f} {'✅' if p_heterocedasticidade > 0.05 else '⚠️'}",
        f"Média dos resíduos ≈ 0: {media_residuos:.4f} ✅",
        f"VIFs (< 10 desejado): {vif_str} {'✅' if all(v < 10 for v in vif_list) else '⚠️'}"
    ]

    # Conclusão
    modelo_valido = (
        r2_pred >= 0.5 and
        p_global < 0.05 and
        1.5 <= dw <= 2.5
    )
    conclusao = "✅ Modelo validado para fins preditivos (uso matemático)." if modelo_valido else "❌ Modelo não validado."

    # Geração do texto final
    texto = f"""
📊 **Análise – Regressão Linear Múltipla**

🔹 **Hipóteses do Modelo**

- **H₀:** Não há relação linear significativa entre pelo menos uma das variáveis independentes ({', '.join(lista_x)}) com a variável resposta ({coluna_y}).
- **H₁:** Existe relação linear significativa entre pelo menos uma das variáveis independentes ({', '.join(lista_x)}) com {coluna_y}.


🔹 **Resumo do Modelo**

- **Variável dependente (Y):** {coluna_y}  
- **Variáveis preditoras (X):** {', '.join(lista_x)}  
- **Equação estimada:**  
  {equacao}

🔎 **Resultados – Itens Críticos (Obrigatórios para predição matemática)**  
{chr(10).join(['- ' + linha for linha in criticos])}

🟡 **Resultados – Itens Recomendados (Desejáveis)**  
{chr(10).join(['- ' + linha for linha in recomendados])}


🔎 **Conclusão Final**  
{conclusao}
""".strip()

    return texto, grafico_base64




def analise_regressao_logistica_binaria(df, coluna_y, lista_x):
    if not coluna_y or not lista_x:
        return "❌ A regressão logística binária requer 1 Y e pelo menos 1 X.", None

    for col in [coluna_y] + lista_x:
        if col not in df.columns:
            return f"❌ Coluna '{col}' não encontrada no conjunto de dados.", None

    df_valid = df[[coluna_y] + lista_x].dropna()
    if len(df_valid) < len(lista_x) + 3:
        return "❌ O modelo requer mais dados válidos.", None

    # Codificação da variável Y
    classes = df_valid[coluna_y].unique()
    if len(classes) != 2:
        return "❌ A variável Y deve conter exatamente 2 categorias distintas para regressão logística binária.", None

    mapeamento = {str(classes[0]): 0, str(classes[1]): 1}
    Y = df_valid[coluna_y].astype(str).map(mapeamento).values
    X_final = df_valid[lista_x]
    x_cols_final = X_final.columns.tolist()

    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    import numpy as np

    X_sm = sm.add_constant(X_final)
    model = sm.Logit(Y, X_sm).fit(disp=0)
    Y_pred_prob = model.predict(X_sm)
    Y_pred_class = (Y_pred_prob >= 0.5).astype(int)

    # Métricas do modelo
    ll_null = sm.Logit(Y, np.ones((len(Y), 1))).fit(disp=0).llf
    ll_model = model.llf
    r2_mcf = 1 - ll_model / ll_null
    auc = roc_auc_score(Y, Y_pred_prob)
    cm = confusion_matrix(Y, Y_pred_class)
    acerto = (cm.diagonal().sum()) / cm.sum()
    vif = [variance_inflation_factor(X_sm.values, i) for i in range(1, X_sm.shape[1])]
    odds_ratios = np.exp(model.params[1:])

    # Gráfico (ROC se mais de uma preditora, curva logística se uma)
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(x_cols_final) == 1:
        X_plot = np.linspace(X_final.iloc[:, 0].min(), X_final.iloc[:, 0].max(), 100)
        coef = model.params.values
        logit = coef[0] + coef[1] * X_plot
        prob = 1 / (1 + np.exp(-logit))
        ax.scatter(X_final.iloc[:, 0], Y, color='black', alpha=0.5, label='Dados')
        ax.plot(X_plot, prob, color='blue', label='Curva ajustada')
        ax.set_xlabel(x_cols_final[0])
        ax.set_ylabel('Probabilidade')
        ax.set_title('Regressão Logística Binária')
        ax.legend()
    else:
        fpr, tpr, _ = roc_curve(Y, Y_pred_prob)
        ax.plot(fpr, tpr, color='blue', label=f'AUC = {auc:.2f}')
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlabel('TFP (Taxa de Falso Positivo)')
        ax.set_ylabel('TAP (Taxa de Acerto Positivo)')
        ax.set_title('Curva ROC\nTAP = Sensibilidade | TFP = 1 - Especificidade')
        ax.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Verificação de validade (critérios obrigatórios)
    validado = (r2_mcf > 0.2) and (auc > 0.7) and any(p < 0.05 for p in model.pvalues[1:])
    conclusao = "✅ Modelo validado para fins preditivos." if validado else "❌ Modelo não validado."

    # Construção do relatório
    texto = f"""
📊 **Análise – Regressão Logística Binária**

🔹 **Hipóteses do Modelo**

- **H₀:** Nenhuma das variáveis independentes está associada à probabilidade do evento ({coluna_y}).  
- **H₁:** Pelo menos uma variável está associada à probabilidade do evento ({coluna_y}).

🔹 **Resumo do Modelo**

- **Variável dependente (Y):** {coluna_y}  
- **Variáveis preditoras (X):** {', '.join(x_cols_final)}  
- **Equação estimada:**  
  log(p / (1 – p)) = {model.params[0]:.4f} {' '.join([f"+ {coef:.4f}·{col}" for coef, col in zip(model.params[1:], x_cols_final)])}

- **Odds Ratios (interpretação prática):**
""" + "\n".join([
        f"  {col}: {odds:.2f} → " +
        (f"Para cada 1 unidade a mais de {col}, a chance do evento {'aumenta' if odds > 1 else 'diminui'} em {abs((odds - 1) * 100):.0f}%, mantendo-se as demais constantes."
         if round(odds, 2) != 1 else
         f"Para cada 1 unidade a mais de {col}, não há variação relevante na chance do evento.")
        for col, odds in zip(x_cols_final, odds_ratios)
    ]) + f"""

🔎 **Resultados – Itens Críticos (Obrigatórios para predição)**  
- R² de McFadden (> 0,20): {r2_mcf:.4f} ✅ (mede o ajuste geral do modelo)  
- AUC (> 0,70): {auc:.4f} ✅ (mede a capacidade do modelo em distinguir entre casos com e sem {coluna_y})  
- Alguma preditora significativa (p < 0,05): {'✅' if any(p < 0.05 for p in model.pvalues[1:]) else '❌'}

🟡 **Resultados – Itens Recomendados (Desejáveis)**  
- Percentual de acerto (> 70%): {acerto * 100:.2f}% {'✅' if acerto >= 0.7 else '⚠️'}  
""" + "\n".join([
        f"- p-valor {col} = {p:.4f} {'✅' if p < 0.05 else '❌'}"
        for col, p in zip(x_cols_final, model.pvalues[1:])
    ]) + "\n" + "\n".join([
        f"- VIF {col} = {v:.2f} {'✅' if v < 10 else '⚠️'} (mede colinearidade entre variáveis)"
        for col, v in zip(x_cols_final, vif)
    ]) + f"""

🔎 **Conclusão Final**  
{conclusao}

""" + ("""
🔹 **Recomendações**  
➡️ O modelo não foi validado. Para melhorá-lo:  
- Remova variáveis com p ≥ 0,05  
- Adicione outras variáveis relevantes  
- Tente usar mais dados para obter resultados mais confiáveis
""" if not validado else "")

    return texto.strip(), grafico_base64







import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np

def analise_regressao_logistica_ordinal(df, coluna_y, lista_x):
    if not coluna_y or not lista_x:
        return "❌ A regressão logística ordinal requer 1 Y e pelo menos 1 X.", None

    try:
        import pandas as pd
        import numpy as np
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64

        # Limpeza dos dados
        df = df[[coluna_y] + lista_x].copy()
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df.dropna(inplace=True)
        for coluna in lista_x:
            df[coluna] = pd.to_numeric(df[coluna], errors='coerce')
        df.dropna(inplace=True)

        if df.empty:
            return "❌ A análise falhou: após limpeza, os dados estão vazios.", None

        categorias_unicas = sorted(df[coluna_y].dropna().unique().tolist())
        df[coluna_y] = pd.Categorical(df[coluna_y], categories=categorias_unicas, ordered=True)

        y = df[coluna_y]
        X = df[lista_x]

        model = OrderedModel(y, X, distr='logit')
        result = model.fit(method='bfgs', disp=False)

        odds_ratios = np.exp(result.params)
        pvalores = result.pvalues
        validado = all(p < 0.05 for p in pvalores)

        texto = f"""
📊 **Análise – Regressão Logística Ordinal**

🔹 **Hipóteses do Modelo**

- **H₀:** Não há relação entre as variáveis independentes e a ordem das categorias da variável {coluna_y}.  
- **H₁:** Pelo menos uma variável está associada à ordem das categorias.

🔹 **Resumo do Modelo**

- **Variável dependente (Y):** {coluna_y}  
- **Variáveis preditoras (X):** {', '.join(lista_x)}  
- **Categorias ordenadas:** {' < '.join(categorias_unicas)}

📌 **Parâmetros estimados:**
"""
        for param, val in result.params.items():
            texto += f"- {param}: {val:.2f} (p = {pvalores[param]:.3f})\n"

        texto += "\n💡 **Odds Ratios (interpretação prática):**\n"
        for param, odds in odds_ratios.items():
            if round(odds, 2) == 1:
                texto += f"- {param}: {odds:.2f} → Cada unidade a mais não altera significativamente a chance de mudança de categoria.\n"
            elif odds > 1:
                texto += f"- {param}: {odds:.2f} → Cada unidade a mais aumenta a chance de estar em uma categoria superior em {((odds - 1) * 100):.0f}%.\n"
            else:
                texto += f"- {param}: {odds:.2f} → Cada unidade a mais reduz a chance de estar em uma categoria superior em {((1 - odds) * 100):.0f}%.\n"

        texto += "\n📉 **VIF (Multicolinearidade):**\n"
        X_vif = sm.add_constant(df[lista_x])
        for i in range(1, X_vif.shape[1]):
            v = variance_inflation_factor(X_vif.values, i)
            status_vif = "✅ adequado (<10)" if v < 10 else "❌ alto (≥10)"
            texto += f"- {X_vif.columns[i]}: {v:.2f} {status_vif} (mede correlação entre variáveis)\n"

        texto += f"\n🔎 **Resultados – Itens Críticos (Obrigatórios para predição)**\n"
        for param, pval in pvalores.items():
            status = "✅ significativo (< 0,05)" if pval < 0.05 else "❌ não significativo (≥ 0,05)"
            texto += f"- p-valor {param}: {pval:.3f} {status}\n"

        texto += f"\n🔎 **Conclusão Final**\n"
        texto += "✅ **Modelo validado para fins preditivos.**\n" if validado else "❌ **Modelo não validado.**\n"

        # Recomendação (somente se houver variáveis não significativas)
        if not validado:
            texto += """
🔹 **Recomendações**
➡️ O modelo não foi validado. Para melhorá-lo:
- Remova variáveis com p ≥ 0,05;
- Adicione novas variáveis relevantes;
- Aumente o número de observações (amostra).
""".strip()

        # Gráfico ilustrativo
        fig, ax = plt.subplots(figsize=(8, 4))
        df.boxplot(column=lista_x[0], by=coluna_y, ax=ax, grid=False)
        plt.suptitle('')
        plt.title(f"{lista_x[0]} por {coluna_y}")
        plt.xlabel(coluna_y)
        plt.ylabel(lista_x[0])
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return texto.strip(), imagem_base64

    except Exception as e:
        return f"❌ Erro ao ajustar o modelo: {str(e)}", None


def analise_regressao_logistica_nominal(df, coluna_y, lista_x):
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    import numpy as np

    if not coluna_y or not lista_x:
        return "❌ A regressão logística nominal requer 1 Y e pelo menos 1 X.", None
    for col in [coluna_y] + lista_x:
        if col not in df.columns:
            return f"❌ Coluna {col} não encontrada no arquivo.", None

    df_valid = df[[coluna_y] + lista_x].dropna()
    if len(df_valid) < len(lista_x) + 3:
        return "❌ O modelo requer mais dados válidos.", None

    # Codificar Y como categoria e transformar em códigos numéricos
    y_categ = df_valid[coluna_y].astype("category")
    Y_labels = dict(enumerate(y_categ.cat.categories))
    df_valid["Y_cod"] = y_categ.cat.codes

    # Preparar X (dummies para variáveis categóricas, mantendo só números)
    X_raw = df_valid[lista_x].copy()
    X_raw = pd.get_dummies(X_raw, drop_first=True)
    X_raw.columns = [c.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') for c in X_raw.columns]
    X_raw.index = pd.RangeIndex(len(X_raw))

    y = df_valid["Y_cod"]

    try:
        model = sm.MNLogit(y, X_raw)
        res = model.fit(disp=0)
    except Exception as e:
        return f"❌ Erro ao ajustar o modelo: {str(e)}", None

    # R² McFadden
    try:
        null_model = sm.MNLogit(y, np.ones((len(y), 1))).fit(disp=0)
        r2_mcf = 1 - res.llf / null_model.llf
    except:
        r2_mcf = None

    # VIF
    vif = []
    if X_raw.shape[1] > 1:
        X_vif = sm.add_constant(X_raw)
        for i in range(1, X_vif.shape[1]):
            vif.append(variance_inflation_factor(X_vif.values, i))
    else:
        vif.append(1.0)

    # Previsão e acurácia
    y_pred = res.predict().argmax(axis=1)
    acuracia = (y == y_pred).sum() / len(y)

    # Matriz de confusão
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=list(Y_labels.values()), cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Matriz de Confusão – Regressão Logística Nominal")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # P-valores corretos para MNLogit
    pvalores_dict = {}
    variaveis_relevantes = []
    variaveis_nrelevantes = []

    if hasattr(res, "pvalues"):
        # res.pvalues: DataFrame (categorias x variáveis)
        for col in X_raw.columns:
            try:
                pvals_col = res.pvalues.loc[:, col]
                min_pval = float(pvals_col.min())
                pvalores_dict[col] = min_pval
                if min_pval < 0.05:
                    variaveis_relevantes.append((col, min_pval))
                else:
                    variaveis_nrelevantes.append((col, min_pval))
            except Exception:
                pvalores_dict[col] = None
                variaveis_nrelevantes.append((col, None))
    else:
        for col in X_raw.columns:
            pvalores_dict[col] = None
            variaveis_nrelevantes.append((col, None))

    # Avaliação
    criterios = []
    if r2_mcf is not None:
        status_r2 = "✅ adequado (> 0,2)" if r2_mcf > 0.2 else "❌ baixo (≤ 0,2)"
        criterios.append(f"- R² de McFadden = {r2_mcf:.3f} {status_r2}")
    criterios.append(f"- Percentual de acerto = {acuracia*100:.2f}%")
    for i, col in enumerate(X_raw.columns):
        pval = pvalores_dict.get(col)
        pval_txt = f"{pval:.3f}".replace('.', ',') if pval is not None else "N/A"
        status_pval = "✅ significativo (< 0,05)" if pval is not None and pval < 0.05 else "❌ não significativo (≥ 0,05)"
        status_vif = "✅ adequado (< 10)" if vif[i] < 10 else "❌ alto (≥ 10)"
        criterios.append(f"- {col}: p-valor = {pval_txt} {status_pval}, VIF = {vif[i]:.2f} {status_vif}")

    houve_significativas = any(p is not None and p < 0.05 for p in pvalores_dict.values())
    validado = r2_mcf is not None and r2_mcf > 0.2 and houve_significativas
    conclusao = "✅ **Modelo validado para fins preditivos.**" if validado else "❌ **Modelo não validado.**"

    recomendacao = ""
    if not validado:
        recomendacao += "\n🔹 **Recomendações**\n➡️ O modelo não foi validado. Para melhorá-lo:"
        if variaveis_nrelevantes:
            recomendacao += "\n- Remova variáveis não significativas: " + ", ".join([
                f"{v[0]} (p = {str(round(v[1], 3)).replace('.', ',') if v[1] is not None else 'N/A'})" for v in variaveis_nrelevantes])
        recomendacao += "\n- Adicione variáveis que influenciem a variável resposta."
        recomendacao += "\n- Aumente o tamanho amostral."

    texto = f"""
📊 **Análise – Regressão Logística Nominal**

🔹 **Hipóteses do Modelo**
- **H₀:** Nenhuma variável está associada às categorias da variável dependente ({coluna_y})
- **H₁:** Pelo menos uma variável está associada às categorias

🔹 **Resumo do Modelo**
- Variável dependente (Y): {coluna_y}
- Variáveis preditoras (Xs): {', '.join(lista_x)}

🔎 **Resultados – Itens Críticos**
{criterios[0]}
{criterios[1]}
- Alguma variável significativa (p < 0,05): {'✅ Sim' if houve_significativas else '❌ Não'}

🟡 **Resultados – Itens Recomendados**
{chr(10).join(criterios[2:])}

🔎 **Conclusão Final**
{conclusao}

🔎 **Variáveis relevantes**
- {', '.join([f"{v[0]} (p = {str(round(v[1], 3)).replace('.', ',')})" for v in variaveis_relevantes]) if variaveis_relevantes else 'Nenhuma significativa'}

{recomendacao}
""".strip()

    return texto, grafico_base64



import pandas as pd
def analise_arvore_decisao(df: pd.DataFrame, coluna_y, lista_x):
    if not coluna_y or not lista_x or len(lista_x) == 0:
        return "❌ A árvore de decisão requer 1 coluna Y e pelo menos 1 X.", None

    cols = [coluna_y] + lista_x
    df_valid = df[cols].dropna()
    if len(df_valid) < len(lista_x) + 5:
        return "❌ O modelo requer mais dados válidos.", None

    Y = df_valid[coluna_y]
    X = df_valid[lista_x]

    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
    from sklearn.metrics import accuracy_score, r2_score
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt

    tipo_modelo = "classificação"
    if pd.api.types.is_numeric_dtype(Y) and len(Y.unique()) > 10:
        model = DecisionTreeRegressor(max_depth=4, random_state=42)
        model.fit(X, Y)
        Y_pred = model.predict(X)
        score = r2_score(Y, Y_pred)
        score_txt = f"R²: {score:.2f}".replace('.', ',')
        criterio_status = "✅ adequado (>0,6)" if score >= 0.6 else "❌ baixo (<=0,6)"
        desempenho = score
        tipo_modelo = "regressão"
    else:
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X, Y)
        Y_pred = model.predict(X)
        acc = accuracy_score(Y, Y_pred)
        score_txt = f"Percentual de acerto: {acc * 100:.2f}%".replace('.', ',')
        criterio_status = "✅ adequado (>70%)" if acc >= 0.7 else "❌ baixo (<=70%)"
        desempenho = acc

    importancias = ", ".join([f"{c} = {v * 100:.1f}%" for c, v in zip(lista_x, model.feature_importances_)])
    regras = export_text(model, feature_names=lista_x)

    # Gráfico árvore
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(
        model,
        feature_names=lista_x,
        class_names=[str(c) for c in Y.unique()] if tipo_modelo == "classificação" else None,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax
    )
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Conclusão
    validado = (tipo_modelo == "regressão" and desempenho >= 0.6) or (tipo_modelo == "classificação" and desempenho >= 0.7)
    conclusao_status = "✅ **Modelo validado.**" if validado else "❌ **Modelo não validado.**"

    # Recomendação
    recomendacao = ""
    if not validado:
        recomendacao = (
            "\n🔎 **Observação / Recomendação**\n➡️ O modelo não foi validado. Considere:\n"
            "- Ajustar parâmetros do modelo (ex: max_depth).\n"
            "- Adicionar variáveis relevantes.\n"
            "- Aumentar o tamanho amostral para maior poder preditivo."
        )

    # Reporte final
    texto = (
        f"📊 **Análise – Árvore de Decisão**\n\n"
        f"🔹 **Hipóteses do modelo**\n"
        f"- O modelo consegue explicar ou classificar a variável Y com base em Xs.\n\n"
        f"🔎 **Resumo do modelo**\n"
        f"- Tipo: {tipo_modelo.capitalize()}\n"
        f"- Variável dependente (Y): {coluna_y}\n"
        f"- Variáveis independentes (Xs): {', '.join(lista_x)}\n\n"
        f"🔎 **Conclusão**\n"
        f"{conclusao_status}\n\n"
        f"🔹 **Critérios avaliados:**\n"
        f"- {score_txt} {criterio_status}\n"
        f"- Número de decisões finais (folhas): {model.get_n_leaves()}\n"
        f"- Profundidade da árvore: {model.get_depth()} níveis\n\n"
        f"🔹 **Importância das variáveis:**\n"
        f"{importancias}\n\n"
        f"📘 **Regras aprendidas pelo modelo:**\n"
        f"{regras}\n"
        f"{recomendacao}"
    )

    return texto.strip(), grafico_base64



import pandas as pd

def analise_random_forest(df: pd.DataFrame, coluna_y, lista_x):
    if not coluna_y or not lista_x or len(lista_x) == 0:
        return "❌ O Random Forest requer 1 coluna Y e pelo menos 1 X.", None

    cols = [coluna_y] + lista_x
    df_valid = df[cols].dropna()
    if len(df_valid) < len(lista_x) + 5:
        return "❌ O modelo requer mais dados válidos.", None

    Y = df_valid[coluna_y]
    X = df_valid[lista_x]

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt

    tipo_modelo = "classificação"
    if pd.api.types.is_numeric_dtype(Y) and len(Y.unique()) > 10:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, Y)
        Y_pred = model.predict(X)
        score = r2_score(Y, Y_pred)
        score_txt = f"R²: {score:.2f}".replace('.', ',')
        criterio_status = "✅ adequado (>0,6)" if score >= 0.6 else "❌ baixo (<=0,6)"
        desempenho = score
        tipo_modelo = "regressão"
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, Y)
        Y_pred = model.predict(X)
        acc = accuracy_score(Y, Y_pred)
        score_txt = f"Percentual de acerto: {acc * 100:.2f}%".replace('.', ',')
        criterio_status = "✅ adequado (>70%)" if acc >= 0.7 else "❌ baixo (<=70%)"
        desempenho = acc

    importancias = model.feature_importances_
    importancia_str = ", ".join([f"{c} = {v * 100:.1f}%" for c, v in zip(lista_x, importancias)])

    # Gráfico de importância das variáveis
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_idx = importancias.argsort()
    ax.barh([lista_x[i] for i in sorted_idx], importancias[sorted_idx])
    ax.set_title("Importância das Variáveis - Random Forest")
    ax.set_xlabel("Importância")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Conclusão
    validado = (tipo_modelo == "regressão" and desempenho >= 0.6) or (tipo_modelo == "classificação" and desempenho >= 0.7)
    conclusao_status = "✅ **Modelo validado.**" if validado else "❌ **Modelo não validado.**"

    # Recomendação
    recomendacao = ""
    if not validado:
        recomendacao = (
            "\n🔎 **Observação / Recomendação**\n➡️ O modelo não foi validado. Considere:\n"
            "- Ajustar parâmetros do modelo (ex: n_estimators, max_depth).\n"
            "- Adicionar variáveis relevantes.\n"
            "- Aumentar o tamanho amostral para maior poder preditivo."
        )

    # Reporte final
    texto = (
        f"📊 **Análise – Random Forest**\n\n"
        f"🔹 **Hipóteses do modelo**\n"
        f"- O modelo consegue explicar ou classificar a variável Y com base em Xs.\n\n"
        f"🔎 **Resumo do modelo**\n"
        f"- Tipo: {tipo_modelo.capitalize()}\n"
        f"- Variável dependente (Y): {coluna_y}\n"
        f"- Variáveis independentes (Xs): {', '.join(lista_x)}\n"
        f"- Número de árvores usadas: 100\n\n"
        f"🔎 **Conclusão**\n"
        f"{conclusao_status}\n\n"
        f"🔹 **Critérios avaliados:**\n"
        f"- {score_txt} {criterio_status}\n\n"
        f"🔹 **Importância das variáveis:**\n"
        f"{importancia_str}\n"
        f"{recomendacao}"
    )

    return texto.strip(), grafico_base64






import pandas as pd
def analise_tendencia_linear(df: pd.DataFrame, coluna_y: str, Data=None, field=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    from datetime import datetime
    import locale
    from sklearn.linear_model import LinearRegression

    aplicar_estilo_minitab()

    # Configura locale BR para casas decimais
    try:
        locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
    except:
        locale.setlocale(locale.LC_ALL, '')

    if not coluna_y or coluna_y not in df.columns:
        return "❌ A Tendência Linear requer 1 coluna Y (série temporal).", None

    if Data and Data in df.columns:
        df_valid = df[[Data, coluna_y]].dropna()
        textos_originais = df_valid[Data].tolist()

        meses_pt = {'jan':1, 'fev':2, 'mar':3, 'abr':4, 'mai':5, 'jun':6,
                    'jul':7, 'ago':8, 'set':9, 'out':10, 'nov':11, 'dez':12}
        meses_en = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                    'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

        def converter_mes(mes_str):
            mes_lower = str(mes_str).strip().lower()[:3]
            mes_num = meses_pt.get(mes_lower) or meses_en.get(mes_lower)
            if mes_num:
                return datetime(datetime.now().year, mes_num, 1)
            else:
                return pd.to_datetime(mes_str, errors='coerce', dayfirst=False, infer_datetime_format=True)

        df_valid['DataConvertida'] = df_valid[Data].apply(converter_mes)

        if df_valid['DataConvertida'].isnull().any():
            return "❌ Existem datas inválidas ou em formato não reconhecido. Use abreviações corretas como Jan, Fev, etc.", None

        df_valid = df_valid.sort_values(by='DataConvertida')
        index = textos_originais

    else:
        df_valid = df[[coluna_y]].dropna()
        df_valid = df_valid.reset_index()
        index = df_valid.index.values.tolist()

    if len(df_valid) < 10:
        return "❌ A série temporal requer pelo menos 10 observações.", None

    Y = df_valid[coluna_y].values
    X = np.arange(len(Y)).reshape(-1,1)

    # Modelo de tendência linear
    modelo = LinearRegression()
    modelo.fit(X, Y)
    previsao_fit = modelo.predict(X)

    horizonte = int(field) if field and str(field).isdigit() else 5
    X_future = np.arange(len(Y), len(Y)+horizonte).reshape(-1,1)
    previsao_future = modelo.predict(X_future)

    # Medidas de acurácia
    mape = np.mean(np.abs((Y - previsao_fit) / Y)) * 100
    mad = np.mean(np.abs(Y - previsao_fit))
    msd = np.mean((Y - previsao_fit)**2)

    # Classificação MAPE
    if mape < 10:
        mape_status = "Excelente"
    elif mape < 20:
        mape_status = "Bom"
    elif mape < 50:
        mape_status = "Aceitável"
    else:
        mape_status = "Ruim"

    # Comparação MAD e MSD
    media_y = np.mean(Y)
    mad_status = "baixo" if mad < media_y * 0.1 else "alto"
    msd_status = "baixo" if msd < media_y * 0.1 else "alto"

    # Formatação BR
    previsao_texto = ", ".join([f"{p:,.2f}".replace(',', 'v').replace('.', ',').replace('v', '.') for p in previsao_future])

    # Gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(index, Y, label='Real', color='black')
    ax.plot(index, previsao_fit, label='Ajuste Linear', color='red')
    ax.plot(range(len(Y), len(Y)+horizonte), previsao_future, 'g--', label='Previsão')
    ax.set_title("📊 Análise – Tendência Linear", fontsize=18, fontweight='bold')
    ax.set_ylabel("Valor", fontsize=16, fontweight='bold')
    ax.set_xlabel("Período", fontsize=16, fontweight='bold')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = (
        "📊 **Análise – Tendência Linear**\n"
        + f"Equação do Modelo: Yt = {modelo.intercept_:,.2f} – {abs(modelo.coef_[0]):,.2f} * t\n".replace(',', 'v').replace('.', ',').replace('v', '.')
        + f"MAPE: {mape:,.2f} – {mape_status} (o erro percentual médio está {mape_status.lower()})\n".replace(',', 'v').replace('.', ',').replace('v', '.')
        + f"MAD: {mad:,.2f} – {mad_status.capitalize()} (comparado à média de {media_y:,.2f})\n".replace(',', 'v').replace('.', ',').replace('v', '.')
        + f"MSD: {msd:,.2f} – {msd_status.capitalize()} (comparado à média de {media_y:,.2f})\n".replace(',', 'v').replace('.', ',').replace('v', '.')
        + f"Previsão para os próximos {horizonte} períodos: {previsao_texto}\n"
        + "\n**Conclusão:**\n"
        + f"{'✅ ' if mape_status != 'Ruim' else ''}Modelo ajustado. Existe uma tendência {'decrescente' if modelo.coef_[0]<0 else 'crescente'}, indicando variação média de {abs(modelo.coef_[0]):,.2f} unidades por período.\n".replace(',', 'v').replace('.', ',').replace('v', '.')
        + "Observação: modelo ajustado significa que foi calculado, mas os erros estão altos, indicando baixa precisão para previsão exata.\n"
        + "\n**Recomendação:**\n"
        + ( "➡️ O MAPE está bom, o modelo pode ser usado para previsões.\n" if mape_status in ["Excelente", "Bom"] 
            else "➡️ O MAPE está ruim, recomenda-se avaliar modelos mais complexos (ex: ARIMA ou modelos sazonais) caso seja necessário previsão precisa.\n" )
    )

    return texto.strip(), grafico_base64






# Dicionário de análises
ANALISES = {
    "Tipo de modelo de regressão": analise_melhor_modelo,
    "Regressão Linear": analise_regressao_linear_simples,
    "Regressão Quadrática": analise_regressao_quadratica,
    "Regressão Cúbica": analise_regressao_cubica,
    "Regressão Linear Múltipla": analise_regressao_linear_multipla,
    "Regressão Binária": analise_regressao_logistica_binaria,
    "Regressão Ordinal": analise_regressao_logistica_ordinal,
    "Regressão Nominal": analise_regressao_logistica_nominal,
    "Árvore de Decisão - CART": analise_arvore_decisao,
    "Random Forest": analise_random_forest,
    "Série Temporal": analise_tendencia_linear
   
}
