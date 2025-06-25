from suporte import *

def analise_tipo_modelo_regressao(df: pd.DataFrame, coluna_y):
    if not coluna_y:
        return "❌ O Tipo de modelo de regressão requer 1 coluna Y.", None

    if coluna_y not in df.columns:
        return f"❌ Coluna {coluna_y} não encontrada no arquivo.", None

    colunas_x = [col for col in df.columns if col != coluna_y and pd.api.types.is_numeric_dtype(df[col])]
    if not colunas_x:
        return "❌ Nenhuma coluna X numérica encontrada automaticamente no arquivo.", None

    coluna_x = colunas_x[0]
    df_valid = df[[coluna_x, coluna_y]].dropna()
    if len(df_valid) < 5:
        return "❌ São necessários pelo menos 5 pares válidos para análise.", None

    X_raw = df_valid[coluna_x].values
    Y = df_valid[coluna_y].values

    modelos = {
        "Linear": np.polyfit(X_raw, Y, 1),
        "Quadrático": np.polyfit(X_raw, Y, 2),
        "Cúbico": np.polyfit(X_raw, Y, 3)
    }

    X_log = X_raw[X_raw > 0]
    Y_log = Y[X_raw > 0]
    if len(X_log) >= 5:
        modelos["Logarítmico"] = np.polyfit(np.log(X_log), Y_log, 1)

    Y_exp = Y[Y > 0]
    X_exp = X_raw[Y > 0]
    if len(Y_exp) >= 5:
        modelos["Exponencial"] = np.polyfit(X_exp, np.log(Y_exp), 1)

    resultados = {}
    for nome, coef in modelos.items():
        if nome == "Logarítmico":
            y_pred = np.polyval(coef, np.log(X_raw[X_raw > 0]))
            y_pred_full = np.full_like(Y, np.nan)
            y_pred_full[X_raw > 0] = y_pred
            y_pred = y_pred_full
        elif nome == "Exponencial":
            y_pred = np.exp(np.polyval(coef, X_raw))
        else:
            y_pred = np.polyval(coef, X_raw)

        ss_res = np.nansum((Y - y_pred) ** 2)
        ss_tot = np.nansum((Y - np.nanmean(Y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        adj = 1 - (1 - r2) * (len(Y) - 1) / (len(Y) - len(coef) - 1)

        resultados[nome] = {
            "coef": coef,
            "r2": r2,
            "r2_adj": adj
        }

    resultados_validos = {k: v for k, v in resultados.items() if v["r2_adj"] >= 0}

    if not resultados_validos:
        return "❌ Nenhum modelo apresentou R² ajustado positivo. Regressão não recomendada.", None

    melhor = max(resultados_validos.items(), key=lambda x: x[1]["r2_adj"])
    nome_vencedor, res_vencedor = melhor

    aviso = ""
    if all(r["r2_adj"] < 0.3 for r in resultados_validos.values()):
        aviso = "⚠ Todos os modelos apresentaram baixo poder de explicação (R² ajustado < 0.3). Use com cautela.\n"

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X_raw, Y, color='black', label='Dados')

    if nome_vencedor == "Logarítmico":
        X_plot = X_raw[X_raw > 0]
        y_fit = np.polyval(res_vencedor["coef"], np.log(X_plot))
        ax.plot(X_plot, y_fit, color='blue', label=f'Modelo: {nome_vencedor}')
    elif nome_vencedor == "Exponencial":
        y_fit = np.exp(np.polyval(res_vencedor["coef"], X_raw))
        ax.plot(X_raw, y_fit, color='blue', label=f'Modelo: {nome_vencedor}')
    else:
        y_fit = np.polyval(res_vencedor["coef"], X_raw)
        ax.plot(X_raw, y_fit, color='blue', label=f'Modelo: {nome_vencedor}')

    ax.set_xlabel(coluna_x)
    ax.set_ylabel(coluna_y)
    ax.set_title("Tipo de Modelo de Regressão")
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    linhas = []
    for nome, r in resultados.items():
        coef_txt = " + ".join([f"{c:.4f}*X^{i}" for i, c in enumerate(r["coef"][::-1])])
        linhas.append(f"- {nome}: R²={r['r2']:.4f}, R² ajustado={r['r2_adj']:.4f}\n  Equação: {coef_txt}")

    texto = f"""
**Tipo de Modelo de Regressão**
{chr(10).join(linhas)}

**Conclusão**
{aviso}Modelo recomendado: {nome_vencedor} (R² ajustado = {res_vencedor['r2_adj']:.4f})
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
    r2_adj = 1 - (1 - r2) * (len(Y) - 1) / (len(Y) - 2)

    X_mat = np.vstack([np.ones_like(X), X]).T
    beta, residuals, rank, s = np.linalg.lstsq(X_mat, Y, rcond=None)
    mse = ss_res / (len(Y) - 2)
    var_beta = mse * np.linalg.inv(X_mat.T @ X_mat)
    se_beta1 = np.sqrt(var_beta[1,1])
    t_stat = beta[1] / se_beta1
    p_valor_beta1 = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(Y)-2))

    erros = []
    for i in range(len(Y)):
        X_train = np.delete(X, i)
        Y_train = np.delete(Y, i)
        coef_lo = np.polyfit(X_train, Y_train, 1)
        y_pred_lo = np.polyval(coef_lo, X[i])
        erros.append((Y[i] - y_pred_lo) ** 2)
    ss_pred = np.sum(erros)
    r2_pred = 1 - ss_pred / ss_tot

    residuos = Y - y_pred
    ad = stats.anderson(residuos)
    ad_crit = ad.critical_values
    ad_sig = list(ad.significance_level)
    if 5 in ad_sig:
        idx = ad_sig.index(5)
        ad_normal = ad.statistic < ad_crit[idx]
    else:
        ad_normal = False

    conclusao = []
    conclusao.append("✅ Resíduos seguem distribuição normal (Anderson-Darling)." if ad_normal else "⚠ Resíduos podem não ser normais (Anderson-Darling).")
    conclusao.append(f"✅ Coeficiente angular significativo (p = {p_valor_beta1:.4f})." if p_valor_beta1 < 0.05 else f"⚠ Coeficiente angular não significativo (p = {p_valor_beta1:.4f}).")
    if ad_normal and p_valor_beta1 < 0.05 and r2_pred > 0.5:
        conclusao.append("✅ Modelo validado.")
    else:
        conclusao.append("⚠ Modelo pode não ser adequado. Verifique os indicadores acima.")

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

    def calcular_modelo(X_sub, cols_sub):
        model = LinearRegression().fit(X_sub, Y)
        Y_pred = model.predict(X_sub)
        r2 = r2_score(Y, Y_pred)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - X_sub.shape[1] - 1)

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

        vif = []
        if X_sub.shape[1] > 1:
            X_sm = sm.add_constant(X_sub)
            for i in range(1, X_sm.shape[1]):
                vif.append(variance_inflation_factor(X_sm, i))
        else:
            vif.append(1.0)

        resid = Y - Y_pred
        mse_full = np.sum((Y - LinearRegression().fit(X_values, Y).predict(X_values)) ** 2) / (n - p_full - 1)
        cp = (np.sum(resid ** 2) / mse_full) - (n - 2 * (X_sub.shape[1] + 1))
        dw = sm.stats.stattools.durbin_watson(resid)

        return {
            "cols": cols_sub,
            "model": model,
            "r2": r2,
            "r2_adj": r2_adj,
            "r2_pred": r2_pred,
            "vif": vif,
            "cp": cp,
            "dw": dw,
            "Y_pred": Y_pred
        }

    resultados = []
    if len(x_cols_final) > 5:
        x_cols_final = x_cols_final[:5]

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

    fig, ax = plt.subplots(figsize=(6, 4))
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

    # Equação
    coef_str = " + ".join([f"{coef:.2f}·{col}" for coef, col in zip(modelo_recomendado["model"].coef_, modelo_recomendado["cols"])])
    equacao = f"Y = {modelo_recomendado['model'].intercept_:.2f} + {coef_str}"

    # Diagnóstico
    conclusao = []
    if modelo_recomendado['r2_pred'] > 0.5:
        conclusao.append("✅ R² preditivo adequado.")
    else:
        conclusao.append("⚠ R² preditivo baixo.")
    if all(v < 10 for v in modelo_recomendado['vif']):
        conclusao.append("✅ Sem multicolinearidade severa (VIF < 10).")
    else:
        conclusao.append("⚠ Multicolinearidade identificada (VIF ≥ 10).")
    if abs(modelo_recomendado["cp"] - (len(modelo_recomendado["cols"]) + 1)) < 2:
        conclusao.append("✅ Cp dentro do esperado.")
    else:
        conclusao.append("⚠ Cp elevado, modelo pode estar superajustado.")
    if 1.5 < modelo_recomendado["dw"] < 2.5:
        conclusao.append("✅ Sem autocorrelação nos resíduos (DW adequado).")
    else:
        conclusao.append("⚠ Autocorrelação identificada nos resíduos (DW fora do ideal).")

    texto = f"""
**Regressão Linear Múltipla**

📌 Modelo recomendado: {', '.join(modelo_recomendado['cols'])}  
📈 Equação: {equacao}  
R²: {modelo_recomendado['r2']:.4f}  
R² ajustado: {modelo_recomendado['r2_adj']:.4f}  
R² preditivo: {modelo_recomendado['r2_pred']:.4f}  
Mallows Cp: {modelo_recomendado['cp']:.4f}  
Durbin-Watson: {modelo_recomendado['dw']:.4f}  
VIFs: {', '.join([f"{c}={v:.2f}" for c, v in zip(modelo_recomendado['cols'], modelo_recomendado['vif'])])}

**Conclusão**  
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

    for col in [coluna_y] + lista_x:
        if col not in df.columns:
            return f"❌ Coluna {col} não encontrada no arquivo.", None

    if df[coluna_y].dtype == 'object' or str(df[coluna_y].dtype).startswith("category"):
        categorias_unicas = sorted(df[coluna_y].dropna().unique())
        df[coluna_y] = pd.Categorical(df[coluna_y], categories=categorias_unicas, ordered=True)
    else:
        return "❌ A variável Y precisa ser categórica (com níveis ordenáveis, como 'Insatisfatorio', 'Satisfatorio', etc).", None

    df_valid = df[[coluna_y] + lista_x].dropna()
    if len(df_valid) < len(lista_x) + 3:
        return "❌ O modelo requer mais dados válidos.", None

    Y = df_valid[coluna_y]
    X_real = df_valid[lista_x].copy()
    X_real.columns = [str(i) for i in range(X_real.shape[1])]
    nomes_originais = dict(zip(X_real.columns, lista_x))
    x_cols_final = X_real.columns.tolist()

    try:
        model = OrderedModel(Y, X_real, distr='logit')
        res = model.fit(method='bfgs', disp=0)
        ll_null = OrderedModel(Y, pd.DataFrame(index=Y.index), distr='logit').fit(method='bfgs', disp=0).llf
    except Exception as e:
        return f"❌ Erro ao ajustar o modelo: {str(e)}", None

    ll_model = res.llf
    r2_mcf = 1 - ll_model / ll_null

    vif = []
    if X_real.shape[1] > 1:
        X_vif = sm.add_constant(X_real.copy())
        for i in range(1, X_vif.shape[1]):
            vif.append(variance_inflation_factor(X_vif.values, i))
    else:
        vif.append(1.0)

    Y_pred = res.model.predict(res.params, exog=X_real).idxmax(axis=1)
    acerto = (Y_pred == Y).mean()

    comentario_odds = "⚠ Teste de proporcionalidade dos odds não implementado diretamente no Python. Avalie graficamente ou com software complementar (ex: Stata, R)."

    fig, ax = plt.subplots(figsize=(6, 4))

    if len(x_cols_final) == 1:
        nome_coluna = x_cols_final[0]
        valores_x = X_real[nome_coluna].astype(float)
        if valores_x.nunique() > 1:
            X_plot = np.linspace(valores_x.min(), valores_x.max(), 100)
            df_plot = pd.DataFrame({nome_coluna: X_plot})
            try:
                probas = res.model.predict(res.params, exog=df_plot)
                for cat in probas.columns:
                    ax.plot(X_plot, probas[cat], label=f'Prob(Y={cat})')
                ax.set_xlabel(nomes_originais[nome_coluna])
                ax.set_ylabel('Probabilidade')
                ax.set_title('Regressão Logística Ordinal')
                ax.legend()
            except:
                ax.text(0.5, 0.5, 'Erro ao gerar gráfico.', ha='center', va='center')
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Valores de X constantes — gráfico indisponível.', ha='center', va='center')
            ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'Gráfico indisponível para múltiplas X.', ha='center', va='center')
        ax.axis('off')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    linhas = []
    melhor_var = None
    menor_pval = 1.0

    for name, coef, pval in zip(res.model.exog_names, res.params, res.pvalues):
        nome_exibicao = nomes_originais.get(name, name)
        linhas.append(f"- {nome_exibicao}: coef={coef:.4f}, p-valor={pval:.4f}")
        if name in x_cols_final and pval < menor_pval:
            menor_pval = pval
            melhor_var = nome_exibicao

    sugestao = ""
    if melhor_var:
        sugestao = f"\n📌 A variável **{melhor_var}** apresentou maior significância (p-valor = {menor_pval:.4f}) e pode ser a melhor explicação individual da variável Y."

    linhas.append(f"R² de McFadden: {r2_mcf:.4f}")
    linhas.append(f"Percentual de acerto: {acerto*100:.2f}%")
    linhas.append("VIFs: " + ", ".join([f"{nomes_originais[c]}={v:.2f}" for c, v in zip(x_cols_final, vif)]))

    conclusao = []
    if r2_mcf > 0.2:
        conclusao.append("✅ Modelo apresenta bom ajuste (R² de McFadden adequado).")
    else:
        conclusao.append("⚠ R² de McFadden baixo, modelo pode não ter bom ajuste.")
    if all(v < 10 for v in vif):
        conclusao.append("✅ Sem multicolinearidade severa (VIF < 10).")
    else:
        conclusao.append("⚠ Multicolinearidade identificada (VIF >= 10).")
    conclusao.append(comentario_odds)

    texto = f"""
**Regressão Logística Ordinal**
{chr(10).join(linhas)}
{sugestao}

**Conclusão**
{chr(10).join(conclusao)}
"""

    return texto.strip(), grafico_base64




def analise_regressao_logistica_nominal(df: pd.DataFrame, coluna_y, lista_x):
    if not coluna_y or not lista_x:
        return "❌ A regressão logística nominal requer 1 Y e pelo menos 1 X.", None

    for col in [coluna_y] + lista_x:
        if col not in df.columns:
            return f"❌ Coluna {col} não encontrada no arquivo.", None

    df_valid = df[[coluna_y] + lista_x].dropna()
    if len(df_valid) < len(lista_x) + 3:
        return "❌ O modelo requer mais dados válidos.", None

    # ✅ Converte Y para códigos numéricos
    Y = df_valid[coluna_y].astype("category")
    Y_codes = Y.cat.codes
    Y_labels = dict(enumerate(Y.cat.categories))

    # ✅ X com nomes seguros (0, 1, 2...) e sem MultiIndex
    X_final = df_valid[lista_x].copy()
    X_final.columns = [str(i) for i in range(X_final.shape[1])]
    nomes_originais = dict(zip(X_final.columns, lista_x))

    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    import numpy as np

    model = sm.MNLogit(Y_codes, X_final)
    res = model.fit(disp=0)

    ll_null = sm.MNLogit(Y_codes, np.ones((len(Y_codes), 1))).fit(disp=0).llf
    ll_model = res.llf
    r2_mcf = 1 - ll_model / ll_null

    vif = []
    if X_final.shape[1] > 1:
        X_vif = sm.add_constant(X_final.copy())
        for i in range(1, X_vif.shape[1]):
            vif.append(variance_inflation_factor(X_vif.values, i))
    else:
        vif.append(1.0)

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

    linhas.append(f"R² de McFadden: {r2_mcf:.4f}")
    acerto = (cm.diagonal().sum()) / cm.sum()
    linhas.append(f"Percentual de acerto: {acerto*100:.2f}%")
    linhas.append("VIFs: " + ", ".join([f"{nomes_originais[c]}={v:.2f}" for c, v in zip(X_final.columns, vif)]))

    sugestao = ""
    if melhor_var:
        sugestao = f"\n📌 A variável **{melhor_var}** teve o menor p-valor ({menor_pval:.4f}) e pode ser a explicação mais relevante isoladamente."

    if len(lista_x) > 1 and any(v >= 10 for v in vif):
        sugestao += "\n⚠ Considere remover variáveis com VIF alto ou p-valor elevado para melhorar o modelo."

    conclusao = []
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
        score_txt = f"Percentual de acerto: {acc*100:.2f}%"

    importancias = ", ".join([f"{c}={v*100:.2f}%" for c, v in zip(lista_x, model.feature_importances_)])
    regras = export_text(model, feature_names=lista_x)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(model, feature_names=lista_x,
              class_names=[str(c) for c in Y.unique()] if not pd.api.types.is_numeric_dtype(Y) else None,
              filled=True, rounded=True, fontsize=8, ax=ax)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
**Árvore de Decisão**
{score_txt}
- Número de folhas: {model.get_n_leaves()}
- Profundidade da árvore: {model.get_depth()}
- Importância das variáveis: {importancias}

**Regras**
{regras}
"""

    return texto.strip(), grafico_base64


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
        score_txt = f"Percentual de acerto: {acc*100:.2f}%"

    importancias = model.feature_importances_
    importancia_str = ", ".join([f"{c}={v*100:.2f}%" for c, v in zip(lista_x, importancias)])

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

    texto = f"""
**Random Forest**
{score_txt}
- Número de árvores: 100
- Importância das variáveis: {importancia_str}
"""

    return texto.strip(), grafico_base64


def analise_arima(df: pd.DataFrame, coluna_y: str, field=None):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ O ARIMA requer 1 coluna Y (série temporal).", None

    df_valid = df[[coluna_y]].dropna()
    if len(df_valid) < 10:
        return "❌ A série temporal requer pelo menos 10 observações.", None

    Y = df_valid[coluna_y].values

    try:
        import pmdarima as pm
    except ImportError:
        return "❌ O pacote pmdarima não está disponível neste ambiente.", None

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
    ax.set_title(f'ARIMA{ordem} - Previsão {horizonte} períodos')
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    texto = f"""
**ARIMA**
- Equação ARIMA: ARIMA{ordem}
- AIC: {aic:.2f}
- BIC: {bic:.2f}
- Previsão para os próximos {horizonte} períodos: {', '.join([f"{p:.2f}" for p in previsao])}

**Conclusão**
✅ Modelo ajustado automaticamente. AIC e BIC indicam qualidade do ajuste. Avalie o gráfico e, se necessário, refine o modelo manualmente.
"""

    return texto.strip(), grafico_base64


def analise_holt_winters(df: pd.DataFrame, coluna_y, field=None):
    if not coluna_y or len(coluna_y) != 1:
        return "❌ O Holt-Winters requer exatamente 1 coluna Y (série temporal).", None

    y_col = coluna_y[0]
    if y_col not in df.columns:
        return f"❌ Coluna {y_col} não encontrada no arquivo.", None

    df_valid = df[[y_col]].dropna()
    if len(df_valid) < 10:
        return "❌ A série temporal requer pelo menos 10 observações.", None

    Y = df_valid[y_col].values

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        return "❌ O pacote statsmodels não está disponível neste ambiente.", None

    horizonte = int(field) if field and str(field).isdigit() else 5
    modelo = ExponentialSmoothing(Y, trend="add", seasonal=None, damped_trend=True).fit()
    previsao = modelo.forecast(horizonte)

    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(Y)), Y, label="Série Original", color="black")
    ax.plot(range(len(Y), len(Y) + horizonte), previsao, label="Previsão", color="blue")
    ax.set_title(f"Holt-Winters (Suavização Exponencial) - Previsão {horizonte} períodos")
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    texto = f"""
**Holt-Winters (Suavização Exponencial)**
- Tendência: aditiva (com amortecimento)
- Sazonalidade: não aplicada
- Previsão para os próximos {horizonte} períodos: {', '.join([f"{p:.2f}" for p in previsao])}

**Conclusão**
✅ Modelo ajustado automaticamente. Simples e robusto para séries com tendência. Avalie o gráfico e ajuste os parâmetros (ex.: sazonalidade) se necessário.
"""

    return texto.strip(), grafico_base64



ANALISES = {
    "Tipo de modelo de regressão": analise_tipo_modelo_regressao,
    "Regressão linear simples": analise_regressao_linear_simples,
    "Regressão linear múltipla": analise_regressao_linear_multipla,
    "Regressão logística binária": analise_regressao_logistica_binaria,
    "Regressão logística ordinal": analise_regressao_logistica_ordinal,
    "Regressão logística nominal": analise_regressao_logistica_nominal,
    "Árvore de decisão": analise_arvore_decisao,
    "Random Forest": analise_random_forest,
    "ARIMA": analise_arima,
    "Holt-Winters": analise_holt_winters
   
}
