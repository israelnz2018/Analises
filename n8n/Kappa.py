# Kappa.py — Análise de Concordância de Atributos (Attribute Agreement Analysis)
# Replica o comportamento do Minitab (Stat > Quality Tools > Attribute Agreement Analysis)
#
# Mapeamento dos campos para o usuário:
#   coluna_y  → Resultado (a classificação dada pelo avaliador)
#   coluna_x  → Peça (item avaliado)
#   subgrupo  → Avaliador
#   field     → Padrão (opcional, gabarito por peça)
#   ordinal   → flag bool: True se as categorias têm ordem (ex.: 1<2<3)
 
from suporte import *
 
 
def concordancia_atributos(df, coluna_y, coluna_x, subgrupo, field=None, ordinal=False):
    """
    Análise de Concordância de Atributos — Attribute Agreement Analysis.
 
    Detecta automaticamente:
      - Se há réplicas (mesmo avaliador avalia a mesma peça mais de uma vez)
      - Se há padrão informado (coluna gabarito)
      - Tipo de dado (binário/nominal/ordinal) pelo número de categorias e flag
 
    Calcula:
      - Within Appraisers (Kappa intra-avaliador) — se houver réplicas
      - Each Appraiser vs Standard — se houver padrão
      - Between Appraisers (Kappa de Fleiss entre avaliadores)
      - All Appraisers vs Standard — se houver padrão
      - Kendall's W (coeficiente de concordância) — se ordinal=True
      - % Match em todas as tabelas
 
    Critério AIAG: Kappa >= 0,75 indica boa concordância. Ideal >= 0,90.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab
 
    # ================== VALIDAÇÕES DE ENTRADA ==================
    if not coluna_y or not coluna_x or not subgrupo:
        return "⚠ É obrigatório informar Resultado, Peça e Avaliador.", None
 
    for col in [coluna_y, coluna_x, subgrupo]:
        if col not in df.columns:
            return f"⚠ Coluna '{col}' não encontrada no arquivo.", None
 
    tem_padrao = field is not None and str(field).strip() != "" and str(field) in df.columns
    cols_usar = [coluna_x, subgrupo, coluna_y]
    if tem_padrao:
        cols_usar.append(field)
 
    dados = df[cols_usar].dropna().copy()
    if dados.empty:
        return "⚠ Não há dados válidos nas colunas informadas.", None
 
    # Tudo como string (categorias)
    dados[coluna_x] = dados[coluna_x].astype(str)
    dados[subgrupo] = dados[subgrupo].astype(str)
    dados[coluna_y] = dados[coluna_y].astype(str)
    if tem_padrao:
        dados[field] = dados[field].astype(str)
 
    # ================== TAMANHOS DO ESTUDO ==================
    n_pecas = dados[coluna_x].nunique()
    n_avaliadores = dados[subgrupo].nunique()
    categorias = sorted(dados[coluna_y].unique())
    n_categorias = len(categorias)
 
    contagem = dados.groupby([coluna_x, subgrupo]).size()
    if contagem.empty:
        return "⚠ Não foi possível agrupar peças por avaliador.", None
 
    n_replicas = int(contagem.min())
    tem_replicas = n_replicas >= 2
 
    if n_pecas < 5:
        return "⚠ É preciso pelo menos 5 peças para a análise. AIAG recomenda 50 peças.", None
    if n_avaliadores < 2:
        return "⚠ É preciso pelo menos 2 avaliadores.", None
    if n_categorias < 2:
        return "⚠ É preciso pelo menos 2 categorias diferentes na coluna Resultado.", None
 
    # Tipo de dado
    if n_categorias == 2:
        tipo_dado = "Binário"
    elif ordinal:
        tipo_dado = "Ordinal"
    else:
        tipo_dado = "Nominal"
 
    # ================== HELPERS ==================
    def br(v, casas=4):
        try:
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                return "—"
            return f"{round(v, casas)}".replace(".", ",")
        except Exception:
            return str(v)
 
    def br_pct(v):
        if v is None:
            return "—"
        try:
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return "—"
            return f"{round(v, 2)}".replace(".", ",") + "%"
        except Exception:
            return "—"
 
    def classificar_kappa(k):
        if k is None or (isinstance(k, float) and np.isnan(k)):
            return "—"
        if k >= 0.90:
            return "✅ Excelente"
        elif k >= 0.75:
            return "✅ Aceitável (AIAG)"
        else:
            return "❌ Inaceitável (< 0,75)"
 
    # ================== KAPPA DE COHEN (entre 2 conjuntos de classificações) ==================
    def kappa_cohen(y1, y2, categorias_lista=None):
        """Kappa de Cohen entre dois vetores de classificações pareadas."""
        y1 = np.asarray(y1)
        y2 = np.asarray(y2)
        if categorias_lista is None:
            categorias_lista = sorted(set(list(y1) + list(y2)))
        n = len(y1)
        if n == 0:
            return None
        # Tabela de contingência
        idx = {c: i for i, c in enumerate(categorias_lista)}
        k = len(categorias_lista)
        tabela = np.zeros((k, k))
        for a, b in zip(y1, y2):
            tabela[idx[a], idx[b]] += 1
        po = np.trace(tabela) / n
        marg1 = tabela.sum(axis=1) / n
        marg2 = tabela.sum(axis=0) / n
        pe = sum(marg1 * marg2)
        if abs(1 - pe) < 1e-10:
            return None
        return (po - pe) / (1 - pe)
 
    # ================== KAPPA DE FLEISS (M avaliadores) ==================
    def kappa_fleiss(matriz_freq):
        """
        matriz_freq: NxK (N peças, K categorias) — quantas vezes cada peça foi
        classificada em cada categoria, somando todos os avaliadores.
        Cada linha deve somar M (número de avaliadores).
        """
        N, K = matriz_freq.shape
        n_total = matriz_freq.sum(axis=1)
        if not np.all(n_total == n_total[0]):
            # Se as linhas não somam o mesmo, normaliza pelo mínimo (Fleiss padrão exige equilíbrio)
            return None
        m = n_total[0]
        if m < 2 or N < 1:
            return None
        # P_i (concordância por peça)
        P_i = (np.sum(matriz_freq * (matriz_freq - 1), axis=1)) / (m * (m - 1))
        P_bar = P_i.mean()
        # P_e (esperado)
        p_j = matriz_freq.sum(axis=0) / (N * m)
        P_e = np.sum(p_j ** 2)
        if abs(1 - P_e) < 1e-10:
            return None
        return (P_bar - P_e) / (1 - P_e)
 
    # ================== KENDALL'S W ==================
    def kendalls_w(matriz_classif):
        """
        matriz_classif: K x N (K avaliadores, N peças) com valores ordinais.
        Retorna o coeficiente de concordância W de Kendall.
        """
        K, N = matriz_classif.shape
        if N < 2 or K < 2:
            return None
        # Converte para ranks dentro de cada avaliador
        ranks = np.zeros_like(matriz_classif, dtype=float)
        for i in range(K):
            ranks[i, :] = pd.Series(matriz_classif[i, :]).rank().values
        R = ranks.sum(axis=0)
        R_bar = R.mean()
        S = np.sum((R - R_bar) ** 2)
        denom = K ** 2 * (N ** 3 - N) / 12.0
        if denom == 0:
            return None
        return S / denom
 
    # ====================================================================
    # 1. WITHIN APPRAISERS (consistência intra-avaliador) — só se réplicas
    # ====================================================================
    within_results = []  # {avaliador, n_inspecionados, n_match, pct_match, kappa, kendall}
    if tem_replicas:
        for av in sorted(dados[subgrupo].unique()):
            sub = dados[dados[subgrupo] == av]
            pecas_av = sorted(sub[coluna_x].unique())
            n_match = 0
            pares_y1 = []
            pares_y2 = []
            classif_por_peca = []  # para Kendall ordinal
            for peca in pecas_av:
                grupo = sub[sub[coluna_x] == peca][coluna_y].tolist()
                if len(grupo) >= 2:
                    # Match: todas as réplicas iguais
                    if len(set(grupo)) == 1:
                        n_match += 1
                    # Para Kappa intra: comparar primeira com segunda réplica (Cohen)
                    pares_y1.append(grupo[0])
                    pares_y2.append(grupo[1])
                    classif_por_peca.append(grupo)
            n_total_av = len(pecas_av)
            pct = 100 * n_match / n_total_av if n_total_av > 0 else 0
            k_intra = kappa_cohen(pares_y1, pares_y2, categorias) if pares_y1 else None
 
            # Kendall intra (se ordinal)
            kendall_intra = None
            if ordinal and classif_por_peca and len(classif_por_peca[0]) >= 2:
                # Mapeia categorias para rank (assume ordem alfabética/numérica)
                map_ord = {c: i for i, c in enumerate(categorias)}
                try:
                    matriz = np.array([[map_ord[v] for v in linha[:n_replicas]] for linha in classif_por_peca]).T
                    kendall_intra = kendalls_w(matriz)
                except Exception:
                    kendall_intra = None
 
            within_results.append({
                "avaliador": av,
                "n_inspecionados": n_total_av,
                "n_match": n_match,
                "pct_match": pct,
                "kappa": k_intra,
                "kendall": kendall_intra
            })
 
    # ====================================================================
    # 2. EACH APPRAISER VS STANDARD — só se padrão informado
    # ====================================================================
    avaliador_vs_padrao = []
    if tem_padrao:
        # Constrói gabarito por peça (assume único valor de padrão por peça)
        gabarito = dados.groupby(coluna_x)[field].first().to_dict()
 
        for av in sorted(dados[subgrupo].unique()):
            sub = dados[dados[subgrupo] == av]
            pecas_av = sorted(sub[coluna_x].unique())
            n_match = 0
            classifs = []
            padroes = []
            for peca in pecas_av:
                grupo = sub[sub[coluna_x] == peca][coluna_y].tolist()
                std = gabarito.get(peca)
                if std is None:
                    continue
                # Match: todas as réplicas batem com o padrão
                if all(c == std for c in grupo):
                    n_match += 1
                # Para Kappa: pega a moda das réplicas vs padrão
                from collections import Counter
                moda = Counter(grupo).most_common(1)[0][0]
                classifs.append(moda)
                padroes.append(std)
 
            n_total_av = len(pecas_av)
            pct = 100 * n_match / n_total_av if n_total_av > 0 else 0
            k_vs_std = kappa_cohen(classifs, padroes, categorias) if classifs else None
 
            avaliador_vs_padrao.append({
                "avaliador": av,
                "n_inspecionados": n_total_av,
                "n_match": n_match,
                "pct_match": pct,
                "kappa": k_vs_std
            })
 
    # ====================================================================
    # 3. BETWEEN APPRAISERS (Kappa de Fleiss entre avaliadores)
    # ====================================================================
    # Para cada peça, conta quantas vezes cada categoria foi atribuída (somando todos os avaliadores)
    pecas_lista = sorted(dados[coluna_x].unique())
    cat_idx = {c: i for i, c in enumerate(categorias)}
    matriz_fleiss = np.zeros((len(pecas_lista), n_categorias))
    for pi, peca in enumerate(pecas_lista):
        grupo = dados[dados[coluna_x] == peca][coluna_y].tolist()
        for c in grupo:
            matriz_fleiss[pi, cat_idx[c]] += 1
 
    kappa_between = kappa_fleiss(matriz_fleiss)
 
    # % match overall: peça é "match" se TODOS os avaliadores e réplicas concordaram
    n_match_overall = 0
    for peca in pecas_lista:
        grupo = dados[dados[coluna_x] == peca][coluna_y].tolist()
        if len(set(grupo)) == 1:
            n_match_overall += 1
    pct_match_overall = 100 * n_match_overall / len(pecas_lista) if pecas_lista else 0
 
    # Kendall's W entre avaliadores (se ordinal)
    kendall_between = None
    if ordinal:
        map_ord = {c: i for i, c in enumerate(categorias)}
        # Constrói matriz K avaliadores x N peças (média das réplicas, se houver)
        try:
            avaliadores_lista = sorted(dados[subgrupo].unique())
            matriz_k = np.zeros((len(avaliadores_lista), len(pecas_lista)))
            for ai, av in enumerate(avaliadores_lista):
                for pi, peca in enumerate(pecas_lista):
                    vals = dados[(dados[subgrupo] == av) & (dados[coluna_x] == peca)][coluna_y].tolist()
                    if vals:
                        # Média dos índices ordinais das réplicas
                        matriz_k[ai, pi] = np.mean([map_ord[v] for v in vals])
                    else:
                        matriz_k[ai, pi] = np.nan
            kendall_between = kendalls_w(matriz_k)
        except Exception:
            kendall_between = None
 
    # ====================================================================
    # 4. ALL APPRAISERS VS STANDARD — só se padrão informado
    # ====================================================================
    overall_vs_padrao = None
    kappa_overall_vs_std = None
    if tem_padrao:
        gabarito = dados.groupby(coluna_x)[field].first().to_dict()
        n_match_std = 0
        classifs_all = []
        padroes_all = []
        for peca in pecas_lista:
            grupo = dados[dados[coluna_x] == peca][coluna_y].tolist()
            std = gabarito.get(peca)
            if std is None:
                continue
            if all(c == std for c in grupo):
                n_match_std += 1
            for c in grupo:
                classifs_all.append(c)
                padroes_all.append(std)
        pct_overall_std = 100 * n_match_std / len(pecas_lista) if pecas_lista else 0
        kappa_overall_vs_std = kappa_cohen(classifs_all, padroes_all, categorias)
        overall_vs_padrao = {
            "n_pecas": len(pecas_lista),
            "n_match": n_match_std,
            "pct_match": pct_overall_std,
            "kappa": kappa_overall_vs_std
        }
 
    # ====================================================================
    # MONTAGEM DO TEXTO DO RESULTADO
    # ====================================================================
    resultado = (
        f"📊 **Análise de Concordância de Atributos**\n\n"
        f"🔎 **Configuração do Estudo:**\n"
        f"- **Peças:** {n_pecas}\n"
        f"- **Avaliadores:** {n_avaliadores}\n"
        f"- **Réplicas (mín.):** {n_replicas}\n"
        f"- **Categorias encontradas:** {n_categorias} ({', '.join(categorias)})\n"
        f"- **Tipo de dado:** {tipo_dado}\n"
        f"- **Padrão informado:** {'Sim' if tem_padrao else 'Não'}\n\n"
    )
 
    # Within Appraisers
    if tem_replicas:
        resultado += "🔎 **Within Appraisers (consistência intra-avaliador):**\n"
        for r in within_results:
            linha = (f"- **{r['avaliador']}:** {r['n_match']}/{r['n_inspecionados']} "
                     f"= {br_pct(r['pct_match'])} | Kappa = {br(r['kappa'])} → {classificar_kappa(r['kappa'])}")
            if ordinal and r['kendall'] is not None:
                linha += f" | Kendall W = {br(r['kendall'])}"
            resultado += linha + "\n"
        resultado += "\n"
 
    # Each Appraiser vs Standard
    if tem_padrao:
        resultado += "🔎 **Cada Avaliador vs Padrão:**\n"
        for r in avaliador_vs_padrao:
            resultado += (f"- **{r['avaliador']}:** {r['n_match']}/{r['n_inspecionados']} "
                          f"= {br_pct(r['pct_match'])} | Kappa = {br(r['kappa'])} → {classificar_kappa(r['kappa'])}\n")
        resultado += "\n"
 
    # Between Appraisers
    resultado += (
        f"🔎 **Between Appraisers (entre avaliadores):**\n"
        f"- **% Match Overall:** {n_match_overall}/{len(pecas_lista)} = {br_pct(pct_match_overall)}\n"
        f"- **Kappa de Fleiss:** {br(kappa_between)} → {classificar_kappa(kappa_between)}\n"
    )
    if ordinal and kendall_between is not None:
        resultado += f"- **Kendall W:** {br(kendall_between)}\n"
    resultado += "\n"
 
    # All Appraisers vs Standard
    if tem_padrao and overall_vs_padrao:
        resultado += (
            f"🔎 **Todos Avaliadores vs Padrão:**\n"
            f"- **% Match:** {overall_vs_padrao['n_match']}/{overall_vs_padrao['n_pecas']} = {br_pct(overall_vs_padrao['pct_match'])}\n"
            f"- **Kappa:** {br(overall_vs_padrao['kappa'])} → {classificar_kappa(overall_vs_padrao['kappa'])}\n\n"
        )
 
    # Conclusão AIAG
    resultado += (
        f"🔎 **Critério AIAG:**\n"
        f"- Kappa ≥ 0,90 → **Excelente concordância**\n"
        f"- Kappa ≥ 0,75 → **Concordância aceitável**\n"
        f"- Kappa < 0,75 → **Concordância inaceitável**, sistema precisa ser melhorado\n\n"
    )
 
    # Conclusão personalizada
    if kappa_between is not None:
        resultado += f"O Kappa entre avaliadores foi **{br(kappa_between)}** → **{classificar_kappa(kappa_between)}**."
 
    # ====================================================================
    # GRÁFICOS (estilo Minitab)
    # ====================================================================
    aplicar_estilo_minitab()
 
    # Determina layout: até 2 gráficos lado a lado se tem ambos within + vs_padrao
    if tem_replicas and tem_padrao:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    elif tem_replicas or tem_padrao:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]
    else:
        # Sem réplicas e sem padrão: gráfico simples de % match overall
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]
 
    fig.suptitle(f"Concordância de Atributos — {coluna_y}", fontsize=12, fontweight='bold')
 
    idx_ax = 0
 
    # Gráfico 1: Within Appraisers
    if tem_replicas:
        ax = axes[idx_ax]
        avs = [r["avaliador"] for r in within_results]
        pcts = [r["pct_match"] for r in within_results]
        # IC 95% binomial Wilson aproximado
        ic_low = []
        ic_up = []
        for r in within_results:
            n = r["n_inspecionados"]
            x = r["n_match"]
            if n == 0:
                ic_low.append(0); ic_up.append(0); continue
            p = x / n
            z = 1.96
            denom = 1 + z**2/n
            centro = (p + z**2/(2*n)) / denom
            margem = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
            ic_low.append(max(0, (centro - margem)*100))
            ic_up.append(min(100, (centro + margem)*100))
 
        x_pos = np.arange(len(avs))
        ax.errorbar(x_pos, pcts,
                    yerr=[np.array(pcts) - np.array(ic_low), np.array(ic_up) - np.array(pcts)],
                    fmt='o', color='steelblue', ecolor='red', capsize=5, markersize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(avs, fontsize=9)
        ax.set_ylabel("% Match", fontsize=9)
        ax.set_title("Within Appraisers", fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(linestyle=':', alpha=0.5)
        idx_ax += 1
 
    # Gráfico 2: Appraiser vs Standard
    if tem_padrao:
        ax = axes[idx_ax]
        avs = [r["avaliador"] for r in avaliador_vs_padrao]
        pcts = [r["pct_match"] for r in avaliador_vs_padrao]
        ic_low = []
        ic_up = []
        for r in avaliador_vs_padrao:
            n = r["n_inspecionados"]
            x = r["n_match"]
            if n == 0:
                ic_low.append(0); ic_up.append(0); continue
            p = x / n
            z = 1.96
            denom = 1 + z**2/n
            centro = (p + z**2/(2*n)) / denom
            margem = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
            ic_low.append(max(0, (centro - margem)*100))
            ic_up.append(min(100, (centro + margem)*100))
 
        x_pos = np.arange(len(avs))
        ax.errorbar(x_pos, pcts,
                    yerr=[np.array(pcts) - np.array(ic_low), np.array(ic_up) - np.array(pcts)],
                    fmt='o', color='steelblue', ecolor='red', capsize=5, markersize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(avs, fontsize=9)
        ax.set_ylabel("% Match", fontsize=9)
        ax.set_title("Appraiser vs Standard", fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(linestyle=':', alpha=0.5)
        idx_ax += 1
 
    # Caso sem nada (só between): mostra gráfico simples
    if not tem_replicas and not tem_padrao:
        ax = axes[0]
        ax.bar(["% Match Overall"], [pct_match_overall], color='steelblue')
        ax.set_ylim(0, 105)
        ax.set_ylabel("%", fontsize=9)
        ax.set_title("Concordância Overall", fontsize=10)
        ax.grid(linestyle=':', alpha=0.5)
 
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
 
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")
 
    return resultado, imagem_base64
 
 
# ====================================================================
# DICIONÁRIO DE EXPORTAÇÃO
# ====================================================================
ANALISES = {
    "Concordância de Atributos": concordancia_atributos,
}
