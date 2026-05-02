# MSA.py — Análises do Sistema de Medição (Measurement System Analysis)
# Segue o padrão dos demais módulos do projeto (Inferencial.py, Capabilidade.py, etc.)
# Referência metodológica: AIAG MSA 4ª edição (com layout visual estilo Minitab)
 
from suporte import *
 
 
def gage_rr_cruzado(df, coluna_peca, coluna_operador, coluna_medicao, field_LIE=None, field_LSE=None):
    """
    Gage R&R Cruzado pelo método ANOVA (AIAG MSA 4ª ed.).
 
    Modelo: Yijk = μ + Pi + Oj + (PO)ij + εijk
      - Pi: efeito da peça (aleatório)
      - Oj: efeito do operador (aleatório)
      - (PO)ij: interação peça × operador (aleatório)
      - εijk: erro/repetibilidade
 
    Componentes de variância obtidos pelo método EMS (Expected Mean Squares),
    igual ao Minitab.
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
    if not coluna_peca or not coluna_operador or not coluna_medicao:
        return "⚠ É obrigatório informar Peça, Operador e Medição.", None
 
    for col in [coluna_peca, coluna_operador, coluna_medicao]:
        if col not in df.columns:
            return f"⚠ Coluna '{col}' não encontrada no arquivo.", None
 
    dados = df[[coluna_peca, coluna_operador, coluna_medicao]].dropna().copy()
    if dados.empty:
        return "⚠ Não há dados válidos nas colunas informadas.", None
 
    # Garante numérico na medição
    try:
        dados[coluna_medicao] = pd.to_numeric(dados[coluna_medicao], errors='coerce')
        dados = dados.dropna(subset=[coluna_medicao])
    except Exception:
        return "⚠ A coluna de Medição deve conter valores numéricos.", None
 
    # Garante string nos identificadores
    dados[coluna_peca] = dados[coluna_peca].astype(str)
    dados[coluna_operador] = dados[coluna_operador].astype(str)
 
    # ================== TAMANHOS DO ESTUDO ==================
    n_pecas = dados[coluna_peca].nunique()
    n_operadores = dados[coluna_operador].nunique()
    contagem = dados.groupby([coluna_peca, coluna_operador]).size()
 
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
    dados_anova = dados.rename(columns={
        coluna_peca: "Peca",
        coluna_operador: "Operador",
        coluna_medicao: "Y"
    })
 
    try:
        modelo = ols('Y ~ C(Peca) + C(Operador) + C(Peca):C(Operador)', data=dados_anova).fit()
        anova_completa = sm.stats.anova_lm(modelo, typ=2)
    except Exception as e:
        return f"⚠ Não foi possível ajustar o modelo ANOVA: {str(e)}", None
 
    # Extrai MS (mean squares) e DF (degrees of freedom) — método EMS do AIAG
    ms_peca = anova_completa.loc['C(Peca)', 'sum_sq'] / anova_completa.loc['C(Peca)', 'df']
    ms_op = anova_completa.loc['C(Operador)', 'sum_sq'] / anova_completa.loc['C(Operador)', 'df']
    ms_int = anova_completa.loc['C(Peca):C(Operador)', 'sum_sq'] / anova_completa.loc['C(Peca):C(Operador)', 'df']
    ms_erro = anova_completa.loc['Residual', 'sum_sq'] / anova_completa.loc['Residual', 'df']
 
    p_int = anova_completa.loc['C(Peca):C(Operador)', 'PR(>F)']
 
    # Decisão sobre incluir ou não a interação (regra Minitab/AIAG: alpha = 0.05)
    incluir_interacao = p_int < 0.05
 
    # ================== COMPONENTES DE VARIÂNCIA ==================
    if incluir_interacao:
        # COM interação
        var_repetibilidade = ms_erro
        var_interacao = max(0.0, (ms_int - ms_erro) / n_replicas)
        var_operador = max(0.0, (ms_op - ms_int) / (n_pecas * n_replicas))
        var_peca = max(0.0, (ms_peca - ms_int) / (n_operadores * n_replicas))
        var_reprodutibilidade = var_operador + var_interacao
    else:
        # SEM interação — recombina interação com erro
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
    # %Contribution (% da variância total)
    if var_total > 0:
        pct_contrib_gage = 100 * var_gage / var_total
        pct_contrib_repet = 100 * var_repetibilidade / var_total
        pct_contrib_reprod = 100 * var_reprodutibilidade / var_total
        pct_contrib_peca = 100 * var_peca / var_total
    else:
        pct_contrib_gage = pct_contrib_repet = pct_contrib_reprod = pct_contrib_peca = 0.0
 
    # %Study Variation (usando 6 desvios-padrão — convenção AIAG MSA 4ª ed.)
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
 
    # Helper para formato BR
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
    fig.suptitle(f"Gage R&R – {coluna_medicao}", fontsize=13, fontweight='bold')
 
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
 
    # 2. Carta R por Operador (range das réplicas por peça-operador)
    ax2 = axes[0, 1]
    ranges_por_op = dados.groupby([coluna_operador, coluna_peca])[coluna_medicao].agg(lambda x: x.max() - x.min())
    operadores_unicos = sorted(dados[coluna_operador].unique())
    pecas_unicas = sorted(dados[coluna_peca].unique())
 
    for op in operadores_unicos:
        valores = [ranges_por_op.get((op, p), np.nan) for p in pecas_unicas]
        ax2.plot(range(len(pecas_unicas)), valores, marker='o', label=str(op), markersize=4)
 
    # Limite de controle para R-chart (UCL = D4 * R-bar)
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
    medias_por_op = dados.groupby([coluna_operador, coluna_peca])[coluna_medicao].mean()
 
    for op in operadores_unicos:
        valores = [medias_por_op.get((op, p), np.nan) for p in pecas_unicas]
        ax3.plot(range(len(pecas_unicas)), valores, marker='o', label=str(op), markersize=4)
 
    # Limites para X-bar chart usando R-bar
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
    dados_por_peca = [dados[dados[coluna_peca] == p][coluna_medicao].values for p in pecas_unicas]
    ax4.boxplot(dados_por_peca, labels=pecas_unicas, patch_artist=True,
                boxprops=dict(facecolor='lightgrey', color='black'),
                medianprops=dict(color='red'))
    ax4.set_title("Medições por Peça", fontsize=10)
    ax4.set_xlabel("Peça", fontsize=8)
    ax4.set_ylabel(coluna_medicao, fontsize=8)
    ax4.tick_params(axis='x', labelsize=6, rotation=45)
    ax4.grid(linestyle=':', alpha=0.5)
 
    # 5. Medições por Operador (boxplot)
    ax5 = axes[2, 0]
    dados_por_op = [dados[dados[coluna_operador] == op][coluna_medicao].values for op in operadores_unicos]
    ax5.boxplot(dados_por_op, labels=operadores_unicos, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red'))
    ax5.set_title("Medições por Operador", fontsize=10)
    ax5.set_xlabel("Operador", fontsize=8)
    ax5.set_ylabel(coluna_medicao, fontsize=8)
    ax5.grid(linestyle=':', alpha=0.5)
 
    # 6. Interação Peça × Operador
    ax6 = axes[2, 1]
    medias_int = dados.groupby([coluna_peca, coluna_operador])[coluna_medicao].mean().unstack()
    for op in operadores_unicos:
        if op in medias_int.columns:
            ax6.plot(range(len(medias_int.index)), medias_int[op].values, marker='o', label=str(op), markersize=4)
    ax6.set_title("Interação Peça × Operador", fontsize=10)
    ax6.set_xlabel("Peça (índice)", fontsize=8)
    ax6.set_ylabel(f"Média de {coluna_medicao}", fontsize=8)
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
 
 
# ====================================================================
# DICIONÁRIO DE EXPORTAÇÃO — usado pelo main.py para registrar
# as análises deste módulo no roteamento global de ferramentas.
# ====================================================================
ANALISES = {
    "Gage R&R": gage_rr_cruzado,
}
 
