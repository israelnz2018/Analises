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
    if not coluna_y or not lista_x:
        return "❌ A regressão linear múltipla requer 1 Y e pelo menos 1 X.", None

    for col in [coluna_y] + lista_x:
        if col not in df.columns:
            return f"❌ Coluna {col} não encontrada no arquivo.", None

    df_valid = df[[coluna_y] + lista_x].dropna()
    if len(df_valid) < len(lista_x) + 3:
        return "❌ O modelo requer mais dados válidos.", None

    Y = df_valid[coluna_y].values
    X_final = pd.get_dummies(df_valid[lista_x], drop_first=True)
    x_cols_final = X_final.columns.tolist()
    X_values = X_final.values
    n, p_full = X_values.shape

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    from itertools import combinations
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    import numpy as np
    from scipy import stats

    def calcular_modelo(X_sub, cols_sub):
        model = LinearRegression().fit(X_sub, Y)
        Y_pred = model.predict(X_sub)
        r2 = r2_score(Y, Y_pred)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - X_sub.shape[1] - 1)

        # R² preditivo (Leave-One-Out)
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
            for i in range(1, X_sm.shape[1]):
                vif.append(variance_inflation_factor(X_sm, i))
        else:
            vif.append(1.0)

        # Mallows Cp
        resid = Y - Y_pred
        mse_full = np.sum((Y - LinearRegression().fit(X_values, Y).predict(X_values)) ** 2) / (n - p_full - 1)
        cp = (np.sum(resid ** 2) / mse_full) - (n - 2 * (X_sub.shape[1] + 1))

        # Durbin-Watson
        dw = sm.stats.stattools.durbin_watson(resid)

        # p-valor do modelo (F-test)
        ss_res = np.sum((Y - Y_pred) ** 2)
        msr = (ss_tot - ss_res) / X_sub.shape[1]
        mse = ss_res / (n - X_sub.shape[1] - 1)
        f_stat = msr / mse
        p_valor_modelo = 1 - stats.f.cdf(f_stat, X_sub.shape[1], n - X_sub.shape[1] - 1)

        return {
            "cols": cols_sub,
            "model": model,
            "r2": r2,
            "r2_adj": r2_adj,
            "r2_pred": r2_pred,
            "vif": vif,
            "cp": cp,
            "dw": dw,
            "p_valor_modelo": p_valor_modelo,
            "Y_pred": Y_pred
        }

    # ➔ Modelo completo com todas as preditoras
    modelo_completo = calcular_modelo(X_values, x_cols_final)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(modelo_completo["Y_pred"], Y - modelo_completo["Y_pred"], color='black')
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Valores preditos")
    ax.set_ylabel("Resíduos")
    ax.set_title("Resíduos vs Valores Preditos")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Equação
    coef_str = " + ".join([f"{coef:.2f}·{col}" for coef, col in zip(modelo_completo["model"].coef_, modelo_completo["cols"])])
    equacao = f"Y = {modelo_completo['model'].intercept_:.2f} + {coef_str}"

    # Diagnóstico
    conclusao = []
    if modelo_completo['r2_pred'] > 0.5:
        conclusao.append("✅ Modelo validado (R² preditivo adequado).")
    else:
        conclusao.append("⚠ Modelo não validado (R² preditivo baixo).")

    if all(v < 10 for v in modelo_completo['vif']):
        conclusao.append("✅ Sem multicolinearidade severa (VIF < 10).")
    else:
        conclusao.append("⚠ Multicolinearidade identificada (VIF ≥ 10). VIF acima de 10 indica alta correlação entre preditoras.")

    if abs(modelo_completo["cp"] - (len(modelo_completo["cols"]) + 1)) < 2:
        conclusao.append("✅ Cp dentro do esperado (Cp ≈ p+1 indica modelo sem viés).")
    else:
        conclusao.append("⚠ Cp elevado, modelo pode estar superajustado (Cp muito maior que p+1).")

    if 1.5 < modelo_completo["dw"] < 2.5:
        conclusao.append("✅ Sem autocorrelação nos resíduos (Durbin-Watson entre 1.5 e 2.5).")
    else:
        conclusao.append("⚠ Autocorrelação identificada nos resíduos (DW fora do ideal). DW <1.5 indica autocorrelação positiva; >2.5 negativa.")

    texto = f"""
📊 **Análise – Regressão Linear Múltipla**

🔹 **Hipóteses do modelo**
- **H₀:** Não há relação linear entre {', '.join(lista_x)} e {coluna_y}
- **H₁:** Existe relação linear entre {', '.join(lista_x)} e {coluna_y}

🔎 **Resumo do modelo**
- Y: {coluna_y}
- Xs: {', '.join(modelo_completo['cols'])}
- Equação: {equacao}
- R²: {modelo_completo['r2']:.3f}
- R² ajustado: {modelo_completo['r2_adj']:.3f}
- R² preditivo: {modelo_completo['r2_pred']:.3f}
- p-valor do modelo: {modelo_completo['p_valor_modelo']:.4f}
- Mallows Cp: {modelo_completo['cp']:.3f}
- Durbin-Watson: {modelo_completo['dw']:.3f}
- VIFs: {', '.join([f"{c}={v:.2f}" for c, v in zip(modelo_completo['cols'], modelo_completo['vif'])])}

🔎 **Conclusão**
{chr(10).join(conclusao)}
""".strip()

    return texto, grafico_base64








def analise_regressao_logistica_binaria(df, coluna_y, lista_x):
    if not coluna_y or not lista_x:
        return "❌ A regressão logística binária requer 1 Y e pelo menos 1 X.", None

    for col in [coluna_y] + lista_x:
        if col not in df.columns:
            return f"❌ Coluna {col} não encontrada no arquivo.", None

    df_valid = df[[coluna_y] + lista_x].dropna()
    if len(df_valid) < len(lista_x) + 3:
        return "❌ O modelo requer mais dados válidos.", None

    # Mapeamento automático do Y categórico
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

    ll_null = sm.Logit(Y, np.ones((len(Y), 1))).fit(disp=0).llf
    ll_model = model.llf
    r2_mcf = 1 - ll_model / ll_null

    vif = []
    for i in range(1, X_sm.shape[1]):
        vif.append(variance_inflation_factor(X_sm.values, i))

    auc = roc_auc_score(Y, Y_pred_prob)
    cm = confusion_matrix(Y, Y_pred_class)
    acerto = (cm.diagonal().sum()) / cm.sum()

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
        ax.plot(fpr, tpr, color='blue', label=f'ROC AUC = {auc:.2f}')
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('Curva ROC')
        ax.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    linhas = []
    for name, coef, pval in zip(['Const'] + x_cols_final, model.params, model.pvalues):
        linhas.append(f"- {name}: coef={coef:.4f}, p-valor={pval:.4f}")
    linhas.append(f"R² de McFadden: {r2_mcf:.4f}")
    linhas.append(f"AUC: {auc:.4f}")
    linhas.append(f"Percentual de acerto: {acerto*100:.2f}%")
    linhas.append("VIFs: " + ", ".join([f"{c}={v:.2f}" for c, v in zip(x_cols_final, vif)]))

    # Novo modelo de reporte bonito e completo
    validado = (r2_mcf > 0.2) and (auc > 0.7) and all(v < 10 for v in vif) and any(p < 0.05 for p in model.pvalues[1:])

    conclusao_status = "✅ Modelo validado." if validado else "❌ Modelo não validado."

    criterios = []
    criterios.append(f"- R² de McFadden = {r2_mcf:.4f} {'✅ adequado (>0.2)' if r2_mcf > 0.2 else '❌ baixo (<=0.2)'}")
    criterios.append(f"- AUC = {auc:.4f} {'✅ adequado (>0.7)' if auc > 0.7 else '❌ baixo (<=0.7)'}")
    criterios.append(f"- Percentual de acerto = {acerto*100:.2f}%")
    for name, pval in zip(x_cols_final, model.pvalues[1:]):
        criterios.append(f"- p-valor {name} = {pval:.4f} {'✅ significativo (<0.05)' if pval < 0.05 else '❌ não significativo (>=0.05)'}")
    criterios.append("- VIFs: " + ", ".join([f"{c}={v:.2f}" for c, v in zip(x_cols_final, vif)]))

    recomendacao = ""
    if not validado:
        recomendacao = """
🔎 Observação / Recomendação
➡️ O modelo não foi validado. Considere:
- Remover preditoras não significativas (p >= 0.05).
- Transformar variáveis ou criar categorias.
- Aumentar o tamanho amostral para maior poder estatístico.
""".strip()

    texto = f"""
📊 Análise – Regressão Logística Binária

🔹 Hipóteses do modelo
- H₀: Nenhuma variável está associada à probabilidade do evento
- H₁: Pelo menos uma variável está associada à probabilidade do evento

🔎 Resumo do modelo
- Variável dependente (Y): {coluna_y}
- Variáveis independentes (Xs): {', '.join(x_cols_final)}
{chr(10).join(linhas)}

🔎 Conclusão
{conclusao_status}

🔹 Critérios avaliados:
{chr(10).join(criterios)}

{recomendacao}
""".strip()

    return texto, grafico_base64



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
        # Selecionar e limpar dados
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
        X = df[lista_x]  # NÃO adicionar constante para OrderedModel

        model = OrderedModel(y, X, distr='logit')
        result = model.fit(method='bfgs', disp=False)

        odds_ratios = np.exp(result.params)
        pvalores = result.pvalues

        texto_resultado = "📊 **Regressão Logística Ordinal**\n\n"
        texto_resultado += "Categorias (ordem): " + " < ".join(categorias_unicas) + "\n\n"
        texto_resultado += "📌 **Parâmetros estimados**:\n"
        for param, val in result.params.items():
            texto_resultado += f"- {param}: {val:.4f} (p = {pvalores[param]:.4f})\n"

        texto_resultado += "\n💡 **Odds Ratios**:\n"
        for param, val in odds_ratios.items():
            texto_resultado += f"- {param}: {val:.4f}\n"

        # VIF
        vif_texto = "\n📉 **VIF (Multicolinearidade)**:\n"
        X_vif = df[lista_x]
        X_vif_const = sm.add_constant(X_vif)
        for i in range(1, X_vif_const.shape[1]):
            vif = variance_inflation_factor(X_vif_const.values, i)
            vif_texto += f"- {X_vif_const.columns[i]}: {vif:.2f}\n"
        texto_resultado += vif_texto

        # Gráfico de boxplot para visualização
        fig, ax = plt.subplots(figsize=(8, 4))
        df.boxplot(column=lista_x[0], by=coluna_y, ax=ax, grid=False)
        plt.suptitle('')
        plt.title(f"{lista_x[0]} por {coluna_y}")
        plt.xlabel(coluna_y)
        plt.ylabel(lista_x[0])
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        return texto_resultado, imagem_base64

    except Exception as e:
        return f"❌ Erro ao ajustar o modelo: {str(e)}", None




def analise_regressao_logistica_nominal(df: pd.DataFrame, coluna_y, lista_x):
    if not coluna_y or not lista_x:
        return "❌ A regressão logística nominal requer 1 Y e pelo menos 1 X.", None

    for col in [coluna_y] + lista_x:
        if col not in df.columns:
            return f"❌ Coluna {col} não encontrada no arquivo.", None

    df_valid = df[[coluna_y] + lista_x].dropna()
    if len(df_valid) < len(lista_x) + 3:
        return "❌ O modelo requer mais dados válidos.", None

    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    import numpy as np

    # Converte Y em categorias numéricas
    Y = df_valid[coluna_y].astype("category")
    Y_codes = Y.cat.codes
    Y_codes.name = "target"
    Y_labels = dict(enumerate(Y.cat.categories))

    # Prepara X com nomes seguros (strings) e únicos
    X_final = df_valid[lista_x].copy()
    X_final.columns = [str(i) for i in range(X_final.shape[1])]
    X_final.columns = [str(col) for col in X_final.columns]
    nomes_originais = dict(zip(X_final.columns, lista_x))

    # Ajusta modelo MNLogit
    try:
        model = sm.MNLogit(Y_codes, X_final)
        res = model.fit(disp=0)
    except Exception as e:
        return f"❌ Erro ao ajustar o modelo: {str(e)}", None

    try:
        ll_null = sm.MNLogit(Y_codes, np.ones((len(Y_codes), 1))).fit(disp=0).llf
        ll_model = res.llf
        r2_mcf = 1 - ll_model / ll_null
    except:
        r2_mcf = None

    # Calcula VIFs
    vif = []
    if X_final.shape[1] > 1:
        X_vif = sm.add_constant(X_final.copy())
        for i in range(1, X_vif.shape[1]):
            vif.append(variance_inflation_factor(X_vif.values, i))
    else:
        vif.append(1.0)

    # Previsões e matriz de confusão
    Y_pred = res.predict().argmax(axis=1)
    cm = confusion_matrix(Y_codes, Y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(Y_labels.values()))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de Confusão - Regressão Logística Nominal")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Resultados do modelo
    linhas = []
    melhor_var = None
    menor_pval = 1.0

    for (cat, coef, pval) in zip(res.params.index, res.params.values, res.pvalues.values):
        coef_str = ", ".join([f"{c:.4f}" for c in coef])
        pval_str = ", ".join([f"{p:.4f}" for p in pval])
        linhas.append(f"- {cat}: coef=[{coef_str}], p-valor=[{pval_str}]")

        for nome_coluna, p in zip(X_final.columns, pval):
            if p < menor_pval:
                menor_pval = p
                melhor_var = nomes_originais[nome_coluna]

    if r2_mcf is not None:
        linhas.append(f"R² de McFadden: {r2_mcf:.4f}")
    acerto = (cm.diagonal().sum()) / cm.sum()
    linhas.append(f"Percentual de acerto: {acerto*100:.2f}%")
    linhas.append("VIFs: " + ", ".join([f"{nomes_originais[c]}={v:.2f}" for c, v in zip(X_final.columns, vif)]))

    # Sugestões e conclusão
    sugestao = ""
    if melhor_var:
        sugestao = f"\n📌 A variável **{melhor_var}** teve o menor p-valor ({menor_pval:.4f}) e pode ser a explicação mais relevante isoladamente."
    if len(lista_x) > 1 and any(v >= 10 for v in vif):
        sugestao += "\n⚠ Considere remover variáveis com VIF alto ou p-valor elevado para melhorar o modelo."

    conclusao = []
    if r2_mcf is not None:
        if r2_mcf > 0.2:
            conclusao.append("✅ Modelo apresenta bom ajuste (R² de McFadden adequado).")
        else:
            conclusao.append("⚠ R² de McFadden baixo, modelo pode não ter bom ajuste.")
    if all(v < 10 for v in vif):
        conclusao.append("✅ Sem multicolinearidade severa (VIF < 10).")
    else:
        conclusao.append("⚠ Multicolinearidade identificada (VIF >= 10).")

    texto = f"""
**Regressão Logística Nominal**
{chr(10).join(linhas)}
{sugestao}

**Conclusão**
{chr(10).join(conclusao)}
"""

    return texto.strip(), grafico_base64










import pandas as pd

def analise_arvore_decisao(df: pd.DataFrame, coluna_y, lista_x):
    if not coluna_y or not lista_x or len(lista_x) == 0:
        return "A árvore de decisão requer 1 coluna Y e pelo menos 1 X.", None

    cols = [coluna_y] + lista_x
    df_valid = df[cols].dropna()
    if len(df_valid) < len(lista_x) + 5:
        return "O modelo requer mais dados válidos.", None

    Y = df_valid[coluna_y]
    X = df_valid[lista_x]

    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
    from sklearn.metrics import accuracy_score, r2_score
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt

    if pd.api.types.is_numeric_dtype(Y) and len(Y.unique()) > 10:
        model = DecisionTreeRegressor(max_depth=4, random_state=42)
        model.fit(X, Y)
        Y_pred = model.predict(X)
        score = r2_score(Y, Y_pred)
        score_txt = f"R²: {score:.4f}"
    else:
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X, Y)
        Y_pred = model.predict(X)
        acc = accuracy_score(Y, Y_pred)
        score_txt = f"Percentual de acerto: {acc * 100:.2f}%"

    importancias = ", ".join([f"{c} = {v * 100:.2f}%" for c, v in zip(lista_x, model.feature_importances_)])
    regras = export_text(model, feature_names=lista_x)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(
        model,
        feature_names=lista_x,
        class_names=[str(c) for c in Y.unique()] if not pd.api.types.is_numeric_dtype(Y) else None,
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



    texto = (
        f"📊 **Análise com Árvore de Decisão**\n\n"
        f"🔹 **Desempenho do modelo**:\n{score_txt}\n\n"
        f"🔹 **Estrutura da árvore**:\n"
        f"- Número de decisões finais (folhas): {model.get_n_leaves()}\n"
        f"- Profundidade da árvore: {model.get_depth()} níveis\n\n"
        f"🔹 **Importância das variáveis utilizadas**:\n"
        f"{importancias}\n\n"
        f"📘 **Regras aprendidas pelo modelo**\n"
        f"{regras}"
    )

    return texto.strip(), grafico_base64


import pandas as pd

def analise_random_forest(df: pd.DataFrame, coluna_y, lista_x):
    if not coluna_y or not lista_x or len(lista_x) == 0:
        return "O Random Forest requer 1 coluna Y e pelo menos 1 X.", None

    cols = [coluna_y] + lista_x
    df_valid = df[cols].dropna()
    if len(df_valid) < len(lista_x) + 5:
        return "O modelo requer mais dados válidos.", None

    Y = df_valid[coluna_y]
    X = df_valid[lista_x]

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt

    if pd.api.types.is_numeric_dtype(Y) and len(Y.unique()) > 10:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, Y)
        Y_pred = model.predict(X)
        score = r2_score(Y, Y_pred)
        score_txt = f"R²: {score:.4f}"
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, Y)
        Y_pred = model.predict(X)
        acc = accuracy_score(Y, Y_pred)
        score_txt = f"Percentual de acerto: {acc * 100:.2f}%"

    importancias = model.feature_importances_
    importancia_str = ", ".join([f"{c} = {v * 100:.2f}%" for c, v in zip(lista_x, importancias)])

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

    texto = (
        "Random Forest - Resultado\n\n"
        f"Desempenho do modelo:\n{score_txt}\n\n"
        "Número de árvores usadas: 100\n\n"
        "Importância das variáveis:\n"
        f"{importancia_str}"
    )

    return texto.strip(), grafico_base64





import pandas as pd

def analise_arima(df: pd.DataFrame, coluna_y: str, field=None):
    if not coluna_y or coluna_y not in df.columns:
        return "O ARIMA requer 1 coluna Y (série temporal).", None

    df_valid = df[[coluna_y]].dropna()
    if len(df_valid) < 10:
        return "A série temporal requer pelo menos 10 observações.", None

    Y = df_valid[coluna_y].values

    try:
        import pmdarima as pm
    except ImportError:
        return "O pacote pmdarima não está disponível neste ambiente.", None

    modelo = pm.auto_arima(Y, seasonal=False, stepwise=True, suppress_warnings=True)
    ordem = modelo.order
    aic = modelo.aic()
    bic = modelo.bic()

    horizonte = int(field) if field and str(field).isdigit() else 5
    previsao = modelo.predict(n_periods=horizonte)

    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(Y)), Y, label='Série Original', color='black')
    ax.plot(range(len(Y), len(Y) + horizonte), previsao, label='Previsão', color='blue')
    ax.set_title(f'Modelo ARIMA{ordem} - Previsão de {horizonte} períodos')
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    previsao_texto = ", ".join([f"{p:.2f}" for p in previsao])

    texto = (
        f"Modelo ARIMA\n\n"
        f"Configuração do modelo: ARIMA{ordem}\n"
        f"Qualidade do ajuste:\n"
        f"- AIC: {aic:.2f}\n"
        f"- BIC: {bic:.2f}\n\n"
        f"Previsão para os próximos {horizonte} períodos:\n"
        f"{previsao_texto}\n\n"
        f"O modelo foi ajustado automaticamente. "
        f"Use o gráfico para avaliar a tendência e decidir se precisa refinar manualmente."
    )

    return texto.strip(), grafico_base64




import pandas as pd

def analise_holt_winters(df: pd.DataFrame, coluna_y, field=None):
    if not coluna_y or len(coluna_y) != 1:
        return "O Holt-Winters requer exatamente 1 coluna Y (série temporal).", None

    y_col = coluna_y[0]
    if y_col not in df.columns:
        return f"Coluna {y_col} não encontrada no arquivo.", None

    df_valid = df[[y_col]].dropna()
    if len(df_valid) < 10:
        return "A série temporal requer pelo menos 10 observações.", None

    Y = df_valid[y_col].values

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        return "O pacote statsmodels não está disponível neste ambiente.", None

    horizonte = int(field) if field and str(field).isdigit() else 5
    modelo = ExponentialSmoothing(Y, trend="add", seasonal=None, damped_trend=True).fit()
    previsao = modelo.forecast(horizonte)

    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(Y)), Y, label="Série Original", color="black")
    ax.plot(range(len(Y), len(Y) + horizonte), previsao, label="Previsão", color="blue")
    ax.set_title(f"Holt-Winters - Previsão de {horizonte} períodos")
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    previsao_texto = ", ".join([f"{p:.2f}" for p in previsao])

    texto = (
        "Holt-Winters (Suavização Exponencial)\n\n"
        "Tendência: aditiva (com amortecimento)\n"
        "Sazonalidade: não aplicada\n\n"
        f"Previsão para os próximos {horizonte} períodos:\n{previsao_texto}\n\n"
        "Conclusão:\n"
        "Modelo ajustado automaticamente. Simples e robusto para séries com tendência.\n"
        "Use o gráfico para avaliar se ajustes adicionais são necessários (ex: sazonalidade)."
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
    "Árvore de Decisão": analise_arvore_decisao,
    "Random Forest": analise_random_forest,
    "ARIMA": analise_arima,
    "Holt-Winters": analise_holt_winters
}
