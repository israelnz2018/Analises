# MSA.py — Análises do Sistema de Medição (Measurement System Analysis)
# Segue o padrão dos demais módulos do projeto (Inferencial.py, Capabilidade.py, etc.)
# Referência metodológica: AIAG MSA 4ª edição (com layout visual estilo Minitab)
#
# Mapeamento dos campos para o usuário:
#   coluna_y  → Medição
#   coluna_x  → Peça
#   subgrupo  → Operador
#   field_LIE → Limite Inferior de Especificação (opcional)
#   field_LSE → Limite Superior de Especificação (opcional)
 
from suporte import *
 
 
def gage_rr(df, coluna_y, coluna_x, subgrupo, field_LIE=None, field_LSE=None):
    """
    Gage R&R Cruzado pelo método ANOVA (AIAG MSA 4ª ed.).
 
    Parâmetros (nomes técnicos do projeto):
      - coluna_y  : nome da coluna com as medições
      - coluna_x  : nome da coluna com as peças
      - subgrupo  : nome da coluna com os operadores
      - field_LIE : Limite Inferior de Especificação (string, opcional)
      - field_LSE : Limite Superior de Especificação (string, opcional)
 
    Modelo: Yijk = μ + Pi + Oj + (PO)ij + εijk
      - Pi: efeito da peça (aleatório)
      - Oj: efeito do operador (aleatório)
      - (PO)ij: interação peça × operador (aleatório)
      - εijk: erro/repetibilidade
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import base64
    from io import BytesIO
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from suporte import aplicar_estilo_minitab
 
    # ================== VALIDAÇÕES DE ENTRADA ==================
    if not coluna_y or not coluna_x or not subgrupo:
        return "⚠ É obrigatório informar Medição, Peça e Operador.", None
 
    for col in [coluna_y, coluna_x, subgrupo]:
        if col not in df.columns:
            return f"⚠ Coluna '{col}' não encontrada no arquivo.", None
 
    dados = df[[coluna_x, subgrupo, coluna_y]].dropna().copy()
    if dados.empty:
        return "⚠ Não há dados válidos nas colunas informadas.", None
 
    # Garante numérico na medição
    try:
        dados[coluna_y] = pd.to_numeric(dados[coluna_y], errors='coerce')
        dados = dados.dropna(subset=[coluna_y])
    except Exception:
        return "⚠ A coluna de Medição deve conter valores numéricos.", None
 
    # Garante string nos identificadores
    dados[coluna_x] = dados[coluna_x].astype(str)
    dados[subgrupo] = dados[subgrupo].astype(str)
 
    # ================== TAMANHOS DO ESTUDO ==================
    n_pecas = dados[coluna_x].nunique()
    n_operadores = dados[subgrupo].nunique()
    contagem = dados.groupby([coluna_x, subgrupo]).size()
 
    if contagem.empty:
        return "⚠ Não foi possível agrupar peças por operador.", None
 
    n_replicas = int(contagem.min())
    if n_replicas < 2:
        return "⚠ É preciso pelo menos 2 réplicas (medições repetidas) para estimar a repetibilidade.", None
    if n_pecas < 2:
        return "⚠ É preciso pelo menos 2 peças para o estudo Gage R&R.", None
    if n_operadores < 2:
        return "⚠ É preciso pelo menos 2 operadores para o estudo Gage R&R Cruzado.", None
 
    # ================== ANOVA DE DOIS FATORES ==================
    # Renomeia para nomes simples na fórmula (evita problemas com acentos/espaços)
    dados_anova = dados.rename(columns={
        coluna_x: "Peca",
        subgrupo: "Operador",
        coluna_y: "Y"
    })
 
    try:
        modelo = ols('Y ~ C(Peca) + C(Operador) + C(Peca):C(Operador)', data=dados_anova).fit()
        anova_completa = sm.stats.anova_lm(modelo, typ=2)
    except Exception as e:
        return f"⚠ Não foi possível ajustar o modelo ANOVA: {str(e)}", None
 
    # Extrai MS (mean squares) — método EMS do AIAG
    ms_peca = anova_completa.loc['C(Peca)', 'sum_sq'] / anova_completa.loc['C(Peca)', 'df']
    ms_op = anova_completa.loc['C(Operador)', 'sum_sq'] / anova_completa.loc['C(Operador)', 'df']
    ms_int = anova_completa.loc['C(Peca):C(Operador)', 'sum_sq'] / anova_completa.loc['C(Peca):C(Operador)', 'df']
    ms_erro = anova_completa.loc['Residual', 'sum_sq'] / anova_completa.loc['Residual', 'df']
 
    p_int = anova_completa.loc['C(Peca):C(Operador)', 'PR(>F)']
 
    # Decisão sobre incluir ou não a interação (regra Minitab/AIAG: alpha = 0.05)
    incluir_interacao = p_int < 0.05
 
    # ================== COMPONENTES DE VARIÂNCIA ==================
    if incluir_interacao:
        var_repetibilidade = ms_erro
        var_interacao = max(0.0, (ms_int - ms_erro) / n_replicas)
        var_operador = max(0.0, (ms_op - ms_int) / (n_pecas * n_replicas))
        var_peca = max(0.0, (ms_peca - ms_int) / (n_operadores * n_replicas))
        var_reprodutibilidade = var_operador + var_interacao
    else:
        df_int = anova_completa.loc['C(Peca):C(Operador)', 'df']
        df_erro = anova_completa.loc['Residual', 'df']
        ss_int = anova_completa.loc['C(Peca):C(Operador)', 'sum_sq']
        ss_erro = anova_completa.loc['Residual', 'sum_sq']
        ms_erro_pooled = (ss_int + ss_erro) / (df_int + df_erro)
        var_repetibilidade = ms_erro_pooled
        var_interacao = 0.0
        var_operador = max(0.0, (ms_op - ms_erro_pooled) / (n_pecas * n_replicas))
        var_peca = max(0.0, (ms_peca - ms_erro_pooled) / (n_operadores * n_replicas))
        var_reprodutibilidade = var_operador
 
    var_gage = var_repetibilidade + var_reprodutibilidade
    var_total = var_gage + var_peca
 
    # ================== MÉTRICAS DE VARIAÇÃO ==================
    if var_total > 0:
        pct_contrib_gage = 100 * var_gage / var_total
        pct_contrib_repet = 100 * var_repetibilidade / var_total
        pct_contrib_reprod = 100 * var_reprodutibilidade / var_total
        pct_contrib_peca = 100 * var_peca / var_total
    else:
        pct_contrib_gage = pct_contrib_repet = pct_contrib_reprod = pct_contrib_peca = 0.0
 
    sd_gage = np.sqrt(var_gage)
    sd_repet = np.sqrt(var_repetibilidade)
    sd_reprod = np.sqrt(var_reprodutibilidade)
    sd_peca = np.sqrt(var_peca)
    sd_total = np.sqrt(var_total)
 
    if sd_total > 0:
        pct_sv_gage = 100 * sd_gage / sd_total
        pct_sv_repet = 100 * sd_repet / sd_total
        pct_sv_reprod = 100 * sd_reprod / sd_total
        pct_sv_peca = 100 * sd_peca / sd_total
    else:
        pct_sv_gage = pct_sv_repet = pct_sv_reprod = pct_sv_peca = 0.0
 
    # %Tolerance (se LIE e LSE foram informados)
    pct_tol_gage = None
    tolerancia = None
    try:
        if field_LIE is not None and field_LSE is not None and str(field_LIE).strip() != "" and str(field_LSE).strip() != "":
            lie = float(field_LIE)
            lse = float(field_LSE)
            tolerancia = lse - lie
            if tolerancia > 0:
                pct_tol_gage = 100 * (6 * sd_gage) / tolerancia
    except (ValueError, TypeError):
        tolerancia = None
 
    # Number of Distinct Categories (ndc)
    if sd_gage > 0:
        ndc = int(np.floor(1.41 * (sd_peca / sd_gage)))
    else:
        ndc = 0
 
    # ================== VEREDITO AIAG ==================
    def classificar_aiag(pct):
        if pct < 10:
            return "✅ Aceitável"
        elif pct < 30:
            return "⚠️ Marginal"
        else:
            return "❌ Inaceitável"
 
    veredito_sv = classificar_aiag(pct_sv_gage)
    veredito_tol = classificar_aiag(pct_tol_gage) if pct_tol_gage is not None else "—"
    veredito_ndc = "✅ Aceitável" if ndc >= 5 else ("⚠️ Marginal" if ndc >= 2 else "❌ Inaceitável")
 
    def br(v, casas=4):
        try:
            return f"{round(v, casas)}".replace(".", ",")
        except Exception:
            return str(v)
 
    def br_pct(v):
        if v is None:
            return "—"
        return f"{round(v, 2)}".replace(".", ",") + "%"
 
    # ================== TEXTO DO RESULTADO ==================
    resultado = (
        f"📊 **Análise – Gage R&R (Método ANOVA)**\n\n"
        f"🔎 **Configuração do Estudo:**\n"
        f"- **Peças:** {n_pecas}\n"
        f"- **Operadores:** {n_operadores}\n"
        f"- **Réplicas:** {n_replicas}\n"
        f"- **Total de medições:** {len(dados)}\n"
        f"- **Modelo usado:** {'Com interação Peça × Operador' if incluir_interacao else 'Sem interação (p ≥ 0,05)'}\n"
        f"- **P-Valor da interação:** {br(p_int, 4)}\n\n"
        f"🔎 **Tabela ANOVA:**\n"
        f"- **MS Peça:** {br(ms_peca)}\n"
        f"- **MS Operador:** {br(ms_op)}\n"
        f"- **MS Peça×Operador:** {br(ms_int)}\n"
        f"- **MS Erro (Repetibilidade):** {br(ms_erro)}\n\n"
        f"🔎 **Componentes de Variância:**\n"
        f"- **VarComp Repetibilidade:** {br(var_repetibilidade)}\n"
        f"- **VarComp Reprodutibilidade:** {br(var_reprodutibilidade)}\n"
        f"   • Operador: {br(var_operador)}\n"
        f"   • Peça×Operador: {br(var_interacao)}\n"
        f"- **VarComp Total Gage R&R:** {br(var_gage)}\n"
        f"- **VarComp Peça:** {br(var_peca)}\n"
        f"- **VarComp Total:** {br(var_total)}\n\n"
        f"🔎 **% Contribuição (% da Variância Total):**\n"
        f"- **Repetibilidade:** {br_pct(pct_contrib_repet)}\n"
        f"- **Reprodutibilidade:** {br_pct(pct_contrib_reprod)}\n"
        f"- **Total Gage R&R:** {br_pct(pct_contrib_gage)}\n"
        f"- **Peça-a-Peça:** {br_pct(pct_contrib_peca)}\n\n"
        f"🔎 **% Variação do Estudo (6 × DP):**\n"
        f"- **Repetibilidade:** {br_pct(pct_sv_repet)}\n"
        f"- **Reprodutibilidade:** {br_pct(pct_sv_reprod)}\n"
        f"- **Total Gage R&R:** {br_pct(pct_sv_gage)} → {veredito_sv}\n"
        f"- **Peça-a-Peça:** {br_pct(pct_sv_peca)}\n\n"
    )
 
    if pct_tol_gage is not None:
        resultado += (
            f"🔎 **% Tolerância (LIE = {br(float(field_LIE), 4)}, LSE = {br(float(field_LSE), 4)}):**\n"
            f"- **Total Gage R&R / Tolerância:** {br_pct(pct_tol_gage)} → {veredito_tol}\n\n"
        )
 
    resultado += (
        f"🔎 **Número de Categorias Distintas (ndc):**\n"
        f"- **ndc = {ndc}** → {veredito_ndc}\n"
        f"   (AIAG recomenda ndc ≥ 5)\n\n"
        f"🔎 **Conclusão (Critério AIAG MSA 4ª ed.):**\n"
        f"- < 10% → Sistema de medição **aceitável**.\n"
        f"- 10% a 30% → Sistema **marginal**, pode ser aceitável dependendo do custo, da importância e do uso.\n"
        f"- > 30% → Sistema **inaceitável**, deve ser melhorado.\n\n"
        f"O total Gage R&R deste estudo foi **{br_pct(pct_sv_gage)}** (% Variação do Estudo), classificando o sistema como **{veredito_sv}**."
    )
 
    # ================== GRÁFICO SIX-PACK (estilo Minitab) ==================
    aplicar_estilo_minitab()
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    fig.suptitle(f"Gage R&R – {coluna_y}", fontsize=13, fontweight='bold')
 
    # 1. Componentes de Variação
    ax1 = axes[0, 0]
    componentes = ['Gage R&R', 'Repetib.', 'Reprodutib.', 'Peça']
    valores_contrib = [pct_contrib_gage, pct_contrib_repet, pct_contrib_reprod, pct_contrib_peca]
    valores_sv = [pct_sv_gage, pct_sv_repet, pct_sv_reprod, pct_sv_peca]
    x = np.arange(len(componentes))
    largura = 0.35
    ax1.bar(x - largura/2, valores_contrib, largura, label='% Contribuição', color='steelblue')
    ax1.bar(x + largura/2, valores_sv, largura, label='% Variação do Estudo', color='lightcoral')
    ax1.set_xticks(x)
    ax1.set_xticklabels(componentes, fontsize=8)
    ax1.set_ylabel("%", fontsize=9)
    ax1.set_title("Componentes de Variação", fontsize=10)
    ax1.legend(fontsize=7)
    ax1.grid(axis='y', linestyle=':', alpha=0.5)
 
    # 2. Carta R por Operador
    ax2 = axes[0, 1]
    ranges_por_op = dados.groupby([subgrupo, coluna_x])[coluna_y].agg(lambda x: x.max() - x.min())
    operadores_unicos = sorted(dados[subgrupo].unique())
    pecas_unicas = sorted(dados[coluna_x].unique())
 
    for op in operadores_unicos:
        valores = [ranges_por_op.get((op, p), np.nan) for p in pecas_unicas]
        ax2.plot(range(len(pecas_unicas)), valores, marker='o', label=str(op), markersize=4)
 
    R_bar = ranges_por_op.mean()
    d4_table = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
    D4 = d4_table.get(n_replicas, 2.0)
    UCL_R = D4 * R_bar
    ax2.axhline(UCL_R, color='red', linestyle='--', linewidth=1, label=f'LSC = {br(UCL_R, 3)}')
    ax2.axhline(R_bar, color='green', linestyle='-', linewidth=1, label=f'R̄ = {br(R_bar, 3)}')
    ax2.set_title("Carta R por Operador", fontsize=10)
    ax2.set_xlabel("Peça (índice)", fontsize=8)
    ax2.set_ylabel("Amplitude", fontsize=8)
    ax2.legend(fontsize=6, loc='upper right')
    ax2.grid(linestyle=':', alpha=0.5)
 
    # 3. Carta X-barra por Operador
    ax3 = axes[1, 0]
    medias_por_op = dados.groupby([subgrupo, coluna_x])[coluna_y].mean()
 
    for op in operadores_unicos:
        valores = [medias_por_op.get((op, p), np.nan) for p in pecas_unicas]
        ax3.plot(range(len(pecas_unicas)), valores, marker='o', label=str(op), markersize=4)
 
    X_bar_bar = medias_por_op.mean()
    a2_table = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
    A2 = a2_table.get(n_replicas, 0.5)
    UCL_X = X_bar_bar + A2 * R_bar
    LCL_X = X_bar_bar - A2 * R_bar
    ax3.axhline(UCL_X, color='red', linestyle='--', linewidth=1, label=f'LSC = {br(UCL_X, 3)}')
    ax3.axhline(LCL_X, color='red', linestyle='--', linewidth=1, label=f'LIC = {br(LCL_X, 3)}')
    ax3.axhline(X_bar_bar, color='green', linestyle='-', linewidth=1, label=f'X̄̄ = {br(X_bar_bar, 3)}')
    ax3.set_title("Carta X̄ por Operador", fontsize=10)
    ax3.set_xlabel("Peça (índice)", fontsize=8)
    ax3.set_ylabel("Média", fontsize=8)
    ax3.legend(fontsize=6, loc='upper right')
    ax3.grid(linestyle=':', alpha=0.5)
 
    # 4. Medições por Peça (boxplot)
    ax4 = axes[1, 1]
    dados_por_peca = [dados[dados[coluna_x] == p][coluna_y].values for p in pecas_unicas]
    ax4.boxplot(dados_por_peca, labels=pecas_unicas, patch_artist=True,
                boxprops=dict(facecolor='lightgrey', color='black'),
                medianprops=dict(color='red'))
    ax4.set_title("Medições por Peça", fontsize=10)
    ax4.set_xlabel("Peça", fontsize=8)
    ax4.set_ylabel(coluna_y, fontsize=8)
    ax4.tick_params(axis='x', labelsize=6, rotation=45)
    ax4.grid(linestyle=':', alpha=0.5)
 
    # 5. Medições por Operador (boxplot)
    ax5 = axes[2, 0]
    dados_por_op = [dados[dados[subgrupo] == op][coluna_y].values for op in operadores_unicos]
    ax5.boxplot(dados_por_op, labels=operadores_unicos, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red'))
    ax5.set_title("Medições por Operador", fontsize=10)
    ax5.set_xlabel("Operador", fontsize=8)
    ax5.set_ylabel(coluna_y, fontsize=8)
    ax5.grid(linestyle=':', alpha=0.5)
 
    # 6. Interação Peça × Operador
    ax6 = axes[2, 1]
    medias_int = dados.groupby([coluna_x, subgrupo])[coluna_y].mean().unstack()
    for op in operadores_unicos:
        if op in medias_int.columns:
            ax6.plot(range(len(medias_int.index)), medias_int[op].values, marker='o', label=str(op), markersize=4)
    ax6.set_title("Interação Peça × Operador", fontsize=10)
    ax6.set_xlabel("Peça (índice)", fontsize=8)
    ax6.set_ylabel(f"Média de {coluna_y}", fontsize=8)
    ax6.legend(fontsize=6)
    ax6.grid(linestyle=':', alpha=0.5)
 
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
 
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")
 
    return resultado, imagem_base64

def vicio_bias_analise(df, coluna_y, field=None, field_LSE=None, field_LIE=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    import io
    import base64
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return "❌ Modulo scipy nao disponivel.", None

    if not coluna_y or coluna_y not in df.columns:
        return "❌ Coluna de medicoes nao encontrada.", None
    if field is None or field == "":
        return "❌ Valor de referencia (Field) e obrigatorio.", None

    try:
        valor_ref = float(field)
    except (ValueError, TypeError):
        return "❌ Valor de referencia invalido.", None

    medicoes = pd.to_numeric(df[coluna_y], errors='coerce').dropna()
    n = len(medicoes)
    if n < 5:
        return "❌ Minimo de 5 medicoes necessarias para o estudo.", None

    media = float(medicoes.mean())
    dp = float(medicoes.std(ddof=1))
    se = dp / (n ** 0.5) if dp > 0 else 0.0
    bias = media - valor_ref

    if se > 0:
        t_stat = bias / se
        p_valor = float(2 * (1 - scipy_stats.t.cdf(abs(t_stat), n - 1)))
    else:
        t_stat = 0.0
        p_valor = 1.0
    t_crit = float(scipy_stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    ic_inferior = bias - t_crit * se
    ic_superior = bias + t_crit * se

    cg = cgk = pct_var_estudo = tolerancia = None
    lse_val = lie_val = None
    if field_LSE not in (None, "") and field_LIE not in (None, ""):
        try:
            lse_val = float(field_LSE)
            lie_val = float(field_LIE)
            tolerancia = lse_val - lie_val
            if tolerancia > 0 and dp > 0:
                cg = (0.20 * tolerancia) / (6.0 * dp)
                cgk = (0.10 * tolerancia - abs(bias)) / (3.0 * dp)
                pct_var_estudo = (6.0 * dp / (0.20 * tolerancia)) * 100.0
        except (ValueError, TypeError):
            pass

    fig, ax = plt.subplots(figsize=(10, 6))
    indices = list(range(1, n + 1))
    ax.plot(indices, medicoes.values, 'o-', color='#3b82f6', markersize=6, linewidth=1.2, label='Medicoes')
    ax.axhline(valor_ref, color='#10b981', linewidth=2.0, label=f'Referencia = {valor_ref:.4f}')
    ax.axhline(media, color='#1d4ed8', linestyle='--', linewidth=1.5, label=f'Media = {media:.4f}')
    if tolerancia and tolerancia > 0:
        ax.axhline(valor_ref + 0.10 * tolerancia, color='#ef4444', linestyle=':', linewidth=1.5,
                   label=f'Ref + 10% Tol ({valor_ref + 0.10 * tolerancia:.4f})')
        ax.axhline(valor_ref - 0.10 * tolerancia, color='#ef4444', linestyle=':', linewidth=1.5,
                   label=f'Ref - 10% Tol ({valor_ref - 0.10 * tolerancia:.4f})')

    # Ajusta ylim com base nos dados reais + margem de 40%
    y_min_dados = float(medicoes.values.min())
    y_max_dados = float(medicoes.values.max())
    y_range = y_max_dados - y_min_dados
    y_margin = y_range * 0.4 if y_range > 0 else abs(media) * 0.01
    ax.set_ylim(y_min_dados - y_margin, y_max_dados + y_margin)

    ax.set_xlabel('Observacao')
    ax.set_ylabel(f'Medicao ({coluna_y})')
    ax.set_title(f'Estudo de Vicio (Tipo 1) - {coluna_y}', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    imagem_base64 = base64.b64encode(buf.getvalue()).decode()

    conclusao_t = "❌ Vicio significativo (p<0.05)" if p_valor < 0.05 else "✅ Vicio nao significativo (p>=0.05)"
    msg = f"""## Estudo de Vicio (Tipo 1) - {coluna_y}

**Estatisticas das medicoes:**
- n = {n}
- Media = {media:.6f}
- Desvio padrao (DP) = {dp:.6f}
- Erro padrao (SE) = {se:.6f}

**Analise do Vicio:**
- Valor de referencia = {valor_ref:.6f}
- Vicio (Bias) = {bias:.6f}
- IC 95% do Bias = [{ic_inferior:.6f}, {ic_superior:.6f}]

**Teste t (H0: Bias = 0):**
- Estatistica t = {t_stat:.4f}
- p-valor = {p_valor:.6f}
- Conclusao: {conclusao_t}
"""
    if cg is not None:
        crit_cg = "✅ Adequado" if cg >= 1.33 else "❌ Inadequado"
        crit_cgk = "✅ Adequado" if cgk >= 1.33 else "❌ Inadequado"
        msg += f"""
**Capabilidade do Instrumento (Tolerancia = {tolerancia:.4f}):**
- LSE = {lse_val:.4f}, LIE = {lie_val:.4f}
- Cg = {cg:.4f} -> {crit_cg} (criterio: Cg >= 1.33)
- Cgk = {cgk:.4f} -> {crit_cgk} (criterio: Cgk >= 1.33)
- %Var(estudo) = {pct_var_estudo:.2f}%
"""
    else:
        msg += "\n**Cg/Cgk nao calculados** (LSE e LIE nao informados)."

    return msg, imagem_base64

def linearidade_analise(df, coluna_y, coluna_x, field_LSE=None, field_LIE=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import io
    import base64
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return "❌ Modulo scipy nao disponivel.", None

    if not coluna_y or coluna_y not in df.columns:
        return "❌ Coluna de medicoes (Y) nao encontrada.", None
    if not coluna_x or coluna_x not in df.columns:
        return "❌ Coluna de valor de referencia (X) nao encontrada.", None

    dados = df[[coluna_x, coluna_y]].copy()
    dados[coluna_x] = pd.to_numeric(dados[coluna_x], errors='coerce')
    dados[coluna_y] = pd.to_numeric(dados[coluna_y], errors='coerce')
    dados = dados.dropna()

    if len(dados) < 6:
        return "❌ Minimo de 6 medicoes necessarias para o estudo de linearidade.", None

    refs_unicos = sorted(dados[coluna_x].unique())
    if len(refs_unicos) < 2:
        return "❌ E necessario pelo menos 2 valores de referencia diferentes.", None

    # Bias individual de cada medicao
    dados['Bias'] = dados[coluna_y] - dados[coluna_x]

    # Estatisticas por peca (referencia)
    stats_por_peca = []
    for ref in refs_unicos:
        grupo = dados[dados[coluna_x] == ref]
        n_g = len(grupo)
        media_med = float(grupo[coluna_y].mean())
        media_bias = float(grupo['Bias'].mean())
        dp_bias = float(grupo['Bias'].std(ddof=1)) if n_g > 1 else 0.0
        stats_por_peca.append({
            'ref': ref,
            'n': n_g,
            'media_medicao': media_med,
            'media_bias': media_bias,
            'dp_bias': dp_bias,
        })

    # Regressao linear: Bias = a + b * Referencia (usa TODAS as medicoes)
    x_vals = dados[coluna_x].values.astype(float)
    y_vals = dados['Bias'].values.astype(float)
    n_total = len(x_vals)

    slope, intercept, r_value, p_slope, se_slope = scipy_stats.linregress(x_vals, y_vals)
    r_squared = r_value ** 2

    # Erro padrao do intercept e p-valor
    x_mean = float(np.mean(x_vals))
    sxx = float(np.sum((x_vals - x_mean) ** 2))
    y_pred_all = intercept + slope * x_vals
    residuos = y_vals - y_pred_all
    sse = float(np.sum(residuos ** 2))
    s_res = (sse / (n_total - 2)) ** 0.5 if n_total > 2 else 0.0
    se_intercept = s_res * ((1.0 / n_total + (x_mean ** 2) / sxx) ** 0.5) if sxx > 0 else 0.0
    if se_intercept > 0:
        t_intercept = intercept / se_intercept
        p_intercept = float(2 * (1 - scipy_stats.t.cdf(abs(t_intercept), n_total - 2)))
    else:
        t_intercept = 0.0
        p_intercept = 1.0

    if se_slope > 0:
        t_slope = slope / se_slope
    else:
        t_slope = 0.0

    # Tolerancia e %Linearidade
    tolerancia = None
    pct_linearidade = None
    lse_val = lie_val = None
    if field_LSE not in (None, "") and field_LIE not in (None, ""):
        try:
            lse_val = float(field_LSE)
            lie_val = float(field_LIE)
            tolerancia = lse_val - lie_val
            if tolerancia > 0:
                # %Linearidade = |slope| * 100 (estilo Minitab simplificado: bias/processo)
                pct_linearidade = abs(slope) * 100.0
        except (ValueError, TypeError):
            tolerancia = None

    # Bias medio global
    bias_medio_global = float(dados['Bias'].mean())

    # ========== GRAFICO ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    # Pontos individuais (bias de cada medicao)
    ax.scatter(x_vals, y_vals, color='#3b82f6', alpha=0.5, s=40, label='Bias individual', zorder=2)

    # Media de bias por peca (destacada)
    refs_arr = np.array([s['ref'] for s in stats_por_peca])
    medias_bias_arr = np.array([s['media_bias'] for s in stats_por_peca])
    ax.scatter(refs_arr, medias_bias_arr, color='#dc2626', s=110, marker='D',
               edgecolor='black', linewidth=1.2, label='Bias medio por peca', zorder=4)

    # Linha de regressao
    x_line = np.linspace(min(x_vals), max(x_vals), 100)
    y_line = intercept + slope * x_line
    sinal = '+' if intercept >= 0 else '-'
    eq_label = f'Regressao: Bias = {slope:.4f}*Ref {sinal} {abs(intercept):.4f}'
    ax.plot(x_line, y_line, color='#1d4ed8', linewidth=2.0, label=eq_label, zorder=3)

    # Faixa IC 95% da regressao
    if n_total > 2 and sxx > 0:
        t_crit_ic = float(scipy_stats.t.ppf(0.975, n_total - 2))
        se_pred = s_res * np.sqrt(1.0 / n_total + ((x_line - x_mean) ** 2) / sxx)
        ic_sup = y_line + t_crit_ic * se_pred
        ic_inf = y_line - t_crit_ic * se_pred
        ax.fill_between(x_line, ic_inf, ic_sup, color='#1d4ed8', alpha=0.15, label='IC 95%', zorder=1)

    # Linha do Bias = 0 (referencia ideal)
    ax.axhline(0, color='#10b981', linewidth=2.0, linestyle='-', label='Bias = 0 (ideal)', zorder=2)

    # Ajusta ylim com margem
    y_min_dados = float(min(y_vals.min(), 0))
    y_max_dados = float(max(y_vals.max(), 0))
    y_range = y_max_dados - y_min_dados
    y_margin = y_range * 0.3 if y_range > 0 else 0.01
    ax.set_ylim(y_min_dados - y_margin, y_max_dados + y_margin)

    ax.set_xlabel(f'Valor de Referencia ({coluna_x})')
    ax.set_ylabel('Bias (Medicao - Referencia)')
    ax.set_title(f'Estudo de Linearidade - {coluna_y}', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    imagem_base64 = base64.b64encode(buf.getvalue()).decode()

    # ========== TEXTO ==========
    conclusao_slope = "❌ Slope significativo - linearidade comprometida (p<0.05)" if p_slope < 0.05 else "✅ Slope nao significativo - boa linearidade (p>=0.05)"
    conclusao_intercept = "❌ Intercept significativo - bias constante presente (p<0.05)" if p_intercept < 0.05 else "✅ Intercept nao significativo (p>=0.05)"

    msg = f"""## Estudo de Linearidade - {coluna_y}

**Dados do estudo:**
- Total de medicoes = {n_total}
- Numero de pecas (referencias) = {len(refs_unicos)}
- Faixa de referencia: {min(refs_unicos):.4f} a {max(refs_unicos):.4f}
- Bias medio global = {bias_medio_global:.6f}

**Estatisticas por peca:**
"""
    for s in stats_por_peca:
        msg += f"- Ref={s['ref']:.4f}: n={s['n']}, Bias medio={s['media_bias']:.6f}, DP={s['dp_bias']:.6f}\n"

    msg += f"""
**Regressao Linear (Bias vs Referencia):**
- Slope (inclinacao) = {slope:.6f}
- Intercept (intercepto) = {intercept:.6f}
- R = {r_value:.4f}
- R-quadrado = {r_squared:.4f}
- S (erro padrao residual) = {s_res:.6f}

**Teste t do Slope (H0: slope = 0):**
- Estatistica t = {t_slope:.4f}
- p-valor = {p_slope:.6f}
- Conclusao: {conclusao_slope}

**Teste t do Intercept (H0: intercept = 0):**
- Estatistica t = {t_intercept:.4f}
- p-valor = {p_intercept:.6f}
- Conclusao: {conclusao_intercept}
"""
    if pct_linearidade is not None:
        crit_lin = "✅ Adequada" if pct_linearidade < 5 else ("⚠️ Marginal" if pct_linearidade < 10 else "❌ Inadequada")
        msg += f"""
**Capabilidade (Tolerancia = {tolerancia:.4f}):**
- LSE = {lse_val:.4f}, LIE = {lie_val:.4f}
- %Linearidade = {pct_linearidade:.2f}% -> {crit_lin}
  (criterio: <5% adequada, <10% marginal)
"""
    else:
        msg += "\n**%Linearidade nao calculada** (LSE e LIE nao informados)."

    return msg, imagem_base64

# ====================================================================
# DICIONÁRIO DE EXPORTAÇÃO
# ====================================================================
ANALISES = {
    "Gage R&R": gage_rr,
    "Vício (Bias)": vicio_bias_analise,
    "Linearidade": linearidade_analise,
}

