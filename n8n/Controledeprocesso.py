from suporte import *

def analise_carta_imr(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) != 1:
        return "❌ A Carta I-MR requer 1 coluna Y (variável do processo).", None

    y_col = colunas_usadas[0]
    dados = df[y_col].dropna()
    if len(dados) < 5:
        return "❌ É necessário pelo menos 5 dados para gerar a Carta I-MR.", None

    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    # Cálculos principais
    media_I = np.mean(dados)
    mr = np.abs(np.diff(dados))
    media_MR = np.mean(mr)

    d2 = 1.128
    sigma = media_MR / d2 if d2 != 0 else 0
    LSC_I = media_I + 3 * sigma
    LIC_I = media_I - 3 * sigma
    LSC_MR = 3.267 * media_MR
    LIC_MR = 0

    # Testes
    testes = []

    # 1 ponto > 3 sigma
    fora_limite = np.where((dados > LSC_I) | (dados < LIC_I))[0]
    if len(fora_limite) > 0:
        testes.append(f"🔴 {len(fora_limite)} ponto(s) fora dos limites de controle.")

    # 9 pontos no mesmo lado da média
    lado = np.where(dados > media_I, 1, -1)
    conta = 0
    for i in range(len(lado)):
        if i == 0 or lado[i] == lado[i-1]:
            conta += 1
            if conta >= 9:
                testes.append("🟠 9 pontos consecutivos do mesmo lado da média.")
                break
        else:
            conta = 1

    # 6 pontos em tendência
    conta_up = conta_down = 0
    for i in range(1, len(dados)):
        if dados[i] > dados[i-1]:
            conta_up += 1
            conta_down = 0
        elif dados[i] < dados[i-1]:
            conta_down += 1
            conta_up = 0
        else:
            conta_up = conta_down = 0
        if conta_up >= 6 or conta_down >= 6:
            testes.append("🟡 6 pontos consecutivos em tendência (subindo ou descendo).")
            break

    # Texto
    texto = f"""
**Carta I-MR**
- Média do processo (Carta I): {media_I:.4f}
- Desvio padrão estimado: {sigma:.4f}
- Média MR: {media_MR:.4f}

- Limites Carta I: LSC={LSC_I:.4f}, LIC={LIC_I:.4f}
- Limites Carta MR: LSC MR={LSC_MR:.4f}, LIC MR={LIC_MR:.4f}

**Resultados dos testes**
"""

    if testes:
        texto += "\n".join(testes)
    else:
        texto += "✅ Processo dentro dos padrões esperados (nenhum alarme nos testes aplicados)."

    # Conclusão
    if testes:
        texto += "\n⚠ Recomenda-se investigar causas especiais e revisar estabilidade do processo."
    else:
        texto += "\n✅ O processo está estável no momento da análise."

    # Gráfico
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Carta I
    ax1.plot(dados, marker='o')
    ax1.axhline(media_I, color='black', linestyle='-', label='Média')
    ax1.axhline(LSC_I, color='red', linestyle='--', label='LSC')
    ax1.axhline(LIC_I, color='red', linestyle='--', label='LIC')
    ax1.set_title("Carta I")
    ax1.legend()

    # Carta MR
    ax2.plot(mr, marker='o')
    ax2.axhline(media_MR, color='black', linestyle='-', label='Média MR')
    ax2.axhline(LSC_MR, color='red', linestyle='--', label='LSC MR')
    ax2.set_title("Carta MR")
    ax2.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return texto.strip(), grafico_base64

def analise_carta_xbarra_r(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) != 2:
        return "❌ A Carta X-Barra R requer 1 coluna Y (variável) e 1 coluna Subgrupo.", None

    y_col = colunas_usadas[0]
    subgrupo_col = colunas_usadas[1]

    if subgrupo_col not in df.columns:
        return f"❌ A coluna de subgrupo '{subgrupo_col}' não foi encontrada.", None

    dados = df[[y_col, subgrupo_col]].dropna()
    if dados.shape[0] < 5:
        return "❌ É necessário pelo menos 5 dados para gerar a Carta X-Barra R.", None

    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    # Agrupa dados
    grupos = dados.groupby(subgrupo_col)[y_col]
    medias = grupos.mean()
    ranges = grupos.max() - grupos.min()
    n_sub = grupos.size().mean()

    if n_sub < 2:
        return "❌ Cada subgrupo deve ter pelo menos 2 elementos.", None

    # Constantes para n_sub médio (aproximação se subgrupos forem irregulares)
    # Para simplicidade, usaremos n=5: A2=0.577, D3=0, D4=2.114
    # Em produção, ideal calcular pelo n real ou ajustar
    A2, D3, D4 = 0.577, 0, 2.114

    media_X = medias.mean()
    media_R = ranges.mean()

    # Limites
    LSC_X = media_X + A2 * media_R
    LIC_X = media_X - A2 * media_R
    LSC_R = D4 * media_R
    LIC_R = D3 * media_R

    # Testes
    testes = []

    # 1 ponto > 3 sigma (aprox pelo limite de controle)
    fora_limite = medias[(medias > LSC_X) | (medias < LIC_X)]
    if not fora_limite.empty:
        testes.append(f"🔴 {fora_limite.shape[0]} média(s) de subgrupo fora dos limites de controle.")

    # 9 pontos no mesmo lado
    lado = np.where(medias > media_X, 1, -1)
    conta = 0
    for i in range(len(lado)):
        if i == 0 or lado[i] == lado[i-1]:
            conta += 1
            if conta >= 9:
                testes.append("🟠 9 médias consecutivas no mesmo lado da linha central.")
                break
        else:
            conta = 1

    # 6 pontos em tendência
    conta_up = conta_down = 0
    for i in range(1, len(medias)):
        if medias.iloc[i] > medias.iloc[i-1]:
            conta_up += 1
            conta_down = 0
        elif medias.iloc[i] < medias.iloc[i-1]:
            conta_down += 1
            conta_up = 0
        else:
            conta_up = conta_down = 0
        if conta_up >= 6 or conta_down >= 6:
            testes.append("🟡 6 médias consecutivas em tendência (subindo ou descendo).")
            break

    # Texto
    texto = f"""
**Carta X-Barra R**
- Média das médias (X-Barra): {media_X:.4f}
- Média das amplitudes (R): {media_R:.4f}
- Limites X-Barra: LSC={LSC_X:.4f}, LIC={LIC_X:.4f}
- Limites R: LSC={LSC_R:.4f}, LIC={LIC_R:.4f}

**Resultados dos testes**
"""

    if testes:
        texto += "\n".join(testes)
    else:
        texto += "✅ Processo dentro dos padrões esperados (nenhum alarme nos testes aplicados)."

    texto += "\n"
    if testes:
        texto += "⚠ Recomenda-se investigar causas especiais e revisar estabilidade do processo."
    else:
        texto += "✅ O processo está estável no momento da análise."

    # Gráfico
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # X-Barra
    ax1.plot(medias.index, medias.values, marker='o')
    ax1.axhline(media_X, color='black', linestyle='-', label='Média')
    ax1.axhline(LSC_X, color='red', linestyle='--', label='LSC')
    ax1.axhline(LIC_X, color='red', linestyle='--', label='LIC')
    ax1.set_title("Carta X-Barra")
    ax1.legend()

    # R
    ax2.plot(ranges.index, ranges.values, marker='o')
    ax2.axhline(media_R, color='black', linestyle='-', label='Média R')
    ax2.axhline(LSC_R, color='red', linestyle='--', label='LSC R')
    if LIC_R > 0:
        ax2.axhline(LIC_R, color='red', linestyle='--', label='LIC R')
    ax2.set_title("Carta R")
    ax2.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return texto.strip(), grafico_base64

def analise_carta_xbarra_s(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) != 2:
        return "❌ A Carta X-Barra S requer 1 coluna Y (variável) e 1 coluna Subgrupo.", None

    y_col = colunas_usadas[0]
    subgrupo_col = colunas_usadas[1]

    if subgrupo_col not in df.columns:
        return f"❌ A coluna de subgrupo '{subgrupo_col}' não foi encontrada.", None

    dados = df[[y_col, subgrupo_col]].dropna()
    if dados.shape[0] < 5:
        return "❌ É necessário pelo menos 5 dados para gerar a Carta X-Barra S.", None

    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    # Agrupa dados
    grupos = dados.groupby(subgrupo_col)[y_col]
    medias = grupos.mean()
    desvios = grupos.std(ddof=1)
    n_sub = grupos.size().mean()

    if n_sub < 2:
        return "❌ Cada subgrupo deve ter pelo menos 2 elementos.", None

    # Constantes para n_sub médio (exemplo n=5): A3=1.427, B3=0, B4=2.089
    A3, B3, B4 = 1.427, 0, 2.089

    media_X = medias.mean()
    media_S = desvios.mean()

    # Limites
    LSC_X = media_X + A3 * media_S
    LIC_X = media_X - A3 * media_S
    LSC_S = B4 * media_S
    LIC_S = B3 * media_S

    # Testes
    testes = []

    # 1 ponto > 3 sigma (aprox pelo limite de controle)
    fora_limite = medias[(medias > LSC_X) | (medias < LIC_X)]
    if not fora_limite.empty:
        testes.append(f"🔴 {fora_limite.shape[0]} média(s) de subgrupo fora dos limites de controle.")

    # 9 pontos no mesmo lado
    lado = np.where(medias > media_X, 1, -1)
    conta = 0
    for i in range(len(lado)):
        if i == 0 or lado[i] == lado[i-1]:
            conta += 1
            if conta >= 9:
                testes.append("🟠 9 médias consecutivas no mesmo lado da linha central.")
                break
        else:
            conta = 1

    # 6 pontos em tendência
    conta_up = conta_down = 0
    for i in range(1, len(medias)):
        if medias.iloc[i] > medias.iloc[i-1]:
            conta_up += 1
            conta_down = 0
        elif medias.iloc[i] < medias.iloc[i-1]:
            conta_down += 1
            conta_up = 0
        else:
            conta_up = conta_down = 0
        if conta_up >= 6 or conta_down >= 6:
            testes.append("🟡 6 médias consecutivas em tendência (subindo ou descendo).")
            break

    # Texto
    texto = f"""
**Carta X-Barra S**
- Média das médias (X-Barra): {media_X:.4f}
- Média dos desvios padrão (S): {media_S:.4f}
- Limites X-Barra: LSC={LSC_X:.4f}, LIC={LIC_X:.4f}
- Limites S: LSC={LSC_S:.4f}, LIC={LIC_S:.4f}

**Resultados dos testes**
"""

    if testes:
        texto += "\n".join(testes)
    else:
        texto += "✅ Processo dentro dos padrões esperados (nenhum alarme nos testes aplicados)."

    texto += "\n"
    if testes:
        texto += "⚠ Recomenda-se investigar causas especiais e revisar estabilidade do processo."
    else:
        texto += "✅ O processo está estável no momento da análise."

    # Gráfico
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # X-Barra
    ax1.plot(medias.index, medias.values, marker='o')
    ax1.axhline(media_X, color='black', linestyle='-', label='Média')
    ax1.axhline(LSC_X, color='red', linestyle='--', label='LSC')
    ax1.axhline(LIC_X, color='red', linestyle='--', label='LIC')
    ax1.set_title("Carta X-Barra")
    ax1.legend()

    # S
    ax2.plot(desvios.index, desvios.values, marker='o')
    ax2.axhline(media_S, color='black', linestyle='-', label='Média S')
    ax2.axhline(LSC_S, color='red', linestyle='--', label='LSC S')
    if LIC_S > 0:
        ax2.axhline(LIC_S, color='red', linestyle='--', label='LIC S')
    ax2.set_title("Carta S")
    ax2.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return texto.strip(), grafico_base64

def analise_carta_p(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) != 2:
        return "❌ A Carta P requer 1 coluna Y (nº de não conformes) e 1 coluna Subgrupo.", None

    y_col = colunas_usadas[0]
    subgrupo_col = colunas_usadas[1]

    if subgrupo_col not in df.columns:
        return f"❌ A coluna de subgrupo '{subgrupo_col}' não foi encontrada.", None

    dados = df[[y_col, subgrupo_col]].dropna()
    if dados.shape[0] < 5:
        return "❌ É necessário pelo menos 5 dados para gerar a Carta P.", None

    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    # Agrupa dados
    grupos = dados.groupby(subgrupo_col)[y_col]
    contagem_nc = grupos.sum()
    n_subgrupos = grupos.count()

    p = contagem_nc / n_subgrupos
    p_barra = contagem_nc.sum() / n_subgrupos.sum()

    # Desvio padrão e limites por subgrupo
    sigma_p = np.sqrt(p_barra * (1 - p_barra) / n_subgrupos)
    LSC = p_barra + 3 * sigma_p
    LIC = p_barra - 3 * sigma_p
    LIC = np.clip(LIC, 0, None)  # LIC não pode ser negativo

    # Testes
    testes = []

    # 1 ponto fora do limite
    fora_limite = p[(p > LSC) | (p < LIC)]
    if not fora_limite.empty:
        testes.append(f"🔴 {fora_limite.shape[0]} subgrupo(s) com proporção fora dos limites de controle.")

    # 9 pontos no mesmo lado
    lado = np.where(p > p_barra, 1, -1)
    conta = 0
    for i in range(len(lado)):
        if i == 0 or lado[i] == lado[i-1]:
            conta += 1
            if conta >= 9:
                testes.append("🟠 9 proporções consecutivas no mesmo lado da média.")
                break
        else:
            conta = 1

    # 6 pontos em tendência
    conta_up = conta_down = 0
    for i in range(1, len(p)):
        if p.iloc[i] > p.iloc[i-1]:
            conta_up += 1
            conta_down = 0
        elif p.iloc[i] < p.iloc[i-1]:
            conta_down += 1
            conta_up = 0
        else:
            conta_up = conta_down = 0
        if conta_up >= 6 or conta_down >= 6:
            testes.append("🟡 6 proporções consecutivas em tendência (subindo ou descendo).")
            break

    # Texto
    texto = f"""
**Carta P**
- Proporção média de não conformes (p̄): {p_barra:.4f}

**Resultados dos testes**
"""

    if testes:
        texto += "\n".join(testes)
    else:
        texto += "✅ Processo dentro dos padrões esperados (nenhum alarme nos testes aplicados)."

    texto += "\n"
    if testes:
        texto += "⚠ Recomenda-se investigar causas especiais e revisar estabilidade do processo."
    else:
        texto += "✅ O processo está estável no momento da análise."

    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(p.index, p.values, marker='o', label='Proporção')
    ax.plot(p.index, LSC, 'r--', label='LSC')
    ax.plot(p.index, LIC, 'r--', label='LIC')
    ax.axhline(p_barra, color='black', linestyle='-', label='Média (p̄)')
    ax.set_title("Carta P")
    ax.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return texto.strip(), grafico_base64

def analise_carta_np(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) != 2:
        return "❌ A Carta NP requer 1 coluna Y (nº de não conformes) e 1 coluna Subgrupo.", None

    y_col = colunas_usadas[0]
    subgrupo_col = colunas_usadas[1]

    if subgrupo_col not in df.columns:
        return f"❌ A coluna de subgrupo '{subgrupo_col}' não foi encontrada.", None

    dados = df[[y_col, subgrupo_col]].dropna()
    if dados.shape[0] < 5:
        return "❌ É necessário pelo menos 5 dados para gerar a Carta NP.", None

    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    # Agrupa dados
    grupos = dados.groupby(subgrupo_col)[y_col]
    nc = grupos.sum()
    n_subgrupos = grupos.count()

    if n_subgrupos.nunique() != 1:
        return "❌ A Carta NP requer subgrupos com mesmo tamanho. Use Carta P para tamanhos variáveis.", None

    n = n_subgrupos.iloc[0]
    p_barra = nc.sum() / (n * len(nc))
    np_barra = n * p_barra
    sigma_np = np.sqrt(n * p_barra * (1 - p_barra))

    LSC = np_barra + 3 * sigma_np
    LIC = np_barra - 3 * sigma_np
    LIC = max(LIC, 0)

    # Testes
    testes = []

    # 1 ponto fora do limite
    fora_limite = nc[(nc > LSC) | (nc < LIC)]
    if not fora_limite.empty:
        testes.append(f"🔴 {fora_limite.shape[0]} subgrupo(s) com contagem fora dos limites de controle.")

    # 9 pontos no mesmo lado
    lado = np.where(nc > np_barra, 1, -1)
    conta = 0
    for i in range(len(lado)):
        if i == 0 or lado[i] == lado[i-1]:
            conta += 1
            if conta >= 9:
                testes.append("🟠 9 contagens consecutivas no mesmo lado da linha central.")
                break
        else:
            conta = 1

    # 6 pontos em tendência
    conta_up = conta_down = 0
    for i in range(1, len(nc)):
        if nc.iloc[i] > nc.iloc[i-1]:
            conta_up += 1
            conta_down = 0
        elif nc.iloc[i] < nc.iloc[i-1]:
            conta_down += 1
            conta_up = 0
        else:
            conta_up = conta_down = 0
        if conta_up >= 6 or conta_down >= 6:
            testes.append("🟡 6 contagens consecutivas em tendência (subindo ou descendo).")
            break

    # Texto
    texto = f"""
**Carta NP**
- Número médio de não conformes (np̄): {np_barra:.4f}
- Limites: LSC={LSC:.4f}, LIC={LIC:.4f}

**Resultados dos testes**
"""

    if testes:
        texto += "\n".join(testes)
    else:
        texto += "✅ Processo dentro dos padrões esperados (nenhum alarme nos testes aplicados)."

    texto += "\n"
    if testes:
        texto += "⚠ Recomenda-se investigar causas especiais e revisar estabilidade do processo."
    else:
        texto += "✅ O processo está estável no momento da análise."

    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(nc.index, nc.values, marker='o', label='Não conformes')
    ax.axhline(np_barra, color='black', linestyle='-', label='Média (np̄)')
    ax.axhline(LSC, color='red', linestyle='--', label='LSC')
    ax.axhline(LIC, color='red', linestyle='--', label='LIC')
    ax.set_title("Carta NP")
    ax.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return texto.strip(), grafico_base64

def analise_carta_c(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) != 1:
        return "❌ A Carta C requer apenas 1 coluna Y (nº de defeitos por unidade).", None

    y_col = colunas_usadas[0]
    dados = df[y_col].dropna()
    if len(dados) < 5:
        return "❌ É necessário pelo menos 5 dados para gerar a Carta C.", None

    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    c_barra = np.mean(dados)
    sigma_c = np.sqrt(c_barra)

    LSC = c_barra + 3 * sigma_c
    LIC = max(c_barra - 3 * sigma_c, 0)

    # Testes
    testes = []

    # 1 ponto fora do limite
    fora_limite = dados[(dados > LSC) | (dados < LIC)]
    if len(fora_limite) > 0:
        testes.append(f"🔴 {len(fora_limite)} unidade(s) com defeitos fora dos limites de controle.")

    # 9 pontos no mesmo lado
    lado = np.where(dados > c_barra, 1, -1)
    conta = 0
    for i in range(len(lado)):
        if i == 0 or lado[i] == lado[i-1]:
            conta += 1
            if conta >= 9:
                testes.append("🟠 9 contagens consecutivas no mesmo lado da linha central.")
                break
        else:
            conta = 1

    # 6 pontos em tendência
    conta_up = conta_down = 0
    for i in range(1, len(dados)):
        if dados.iloc[i] > dados.iloc[i-1]:
            conta_up += 1
            conta_down = 0
        elif dados.iloc[i] < dados.iloc[i-1]:
            conta_down += 1
            conta_up = 0
        else:
            conta_up = conta_down = 0
        if conta_up >= 6 or conta_down >= 6:
            testes.append("🟡 6 contagens consecutivas em tendência (subindo ou descendo).")
            break

    # Texto
    texto = f"""
**Carta C**
- Número médio de defeitos (c̄): {c_barra:.4f}
- Limites: LSC={LSC:.4f}, LIC={LIC:.4f}

**Resultados dos testes**
"""

    if testes:
        texto += "\n".join(testes)
    else:
        texto += "✅ Processo dentro dos padrões esperados (nenhum alarme nos testes aplicados)."

    texto += "\n"
    if testes:
        texto += "⚠ Recomenda-se investigar causas especiais e revisar estabilidade do processo."
    else:
        texto += "✅ O processo está estável no momento da análise."

    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dados.index, dados.values, marker='o', label='Defeitos')
    ax.axhline(c_barra, color='black', linestyle='-', label='Média (c̄)')
    ax.axhline(LSC, color='red', linestyle='--', label='LSC')
    ax.axhline(LIC, color='red', linestyle='--', label='LIC')
    ax.set_title("Carta C")
    ax.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return texto.strip(), grafico_base64

def analise_carta_u(df: pd.DataFrame, colunas_usadas: list, field=None):
    if len(colunas_usadas) != 2:
        return "❌ A Carta U requer 1 coluna Y (nº de defeitos) e 1 coluna Subgrupo.", None

    y_col = colunas_usadas[0]
    subgrupo_col = colunas_usadas[1]

    if subgrupo_col not in df.columns:
        return f"❌ A coluna de subgrupo '{subgrupo_col}' não foi encontrada.", None

    dados = df[[y_col, subgrupo_col]].dropna()
    if dados.shape[0] < 5:
        return "❌ É necessário pelo menos 5 dados para gerar a Carta U.", None

    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    # Agrupa dados
    grupos = dados.groupby(subgrupo_col)[y_col]
    defeitos = grupos.sum()
    tamanhos = grupos.count()

    u = defeitos / tamanhos
    u_barra = defeitos.sum() / tamanhos.sum()

    sigma_u = np.sqrt(u_barra / tamanhos)
    LSC = u_barra + 3 * sigma_u
    LIC = np.clip(u_barra - 3 * sigma_u, 0, None)

    # Testes
    testes = []

    # 1 ponto fora do limite
    fora_limite = u[(u > LSC) | (u < LIC)]
    if not fora_limite.empty:
        testes.append(f"🔴 {fora_limite.shape[0]} subgrupo(s) com taxa fora dos limites de controle.")

    # 9 pontos no mesmo lado
    lado = np.where(u > u_barra, 1, -1)
    conta = 0
    for i in range(len(lado)):
        if i == 0 or lado[i] == lado[i-1]:
            conta += 1
            if conta >= 9:
                testes.append("🟠 9 taxas consecutivas no mesmo lado da linha central.")
                break
        else:
            conta = 1

    # 6 pontos em tendência
    conta_up = conta_down = 0
    for i in range(1, len(u)):
        if u.iloc[i] > u.iloc[i-1]:
            conta_up += 1
            conta_down = 0
        elif u.iloc[i] < u.iloc[i-1]:
            conta_down += 1
            conta_up = 0
        else:
            conta_up = conta_down = 0
        if conta_up >= 6 or conta_down >= 6:
            testes.append("🟡 6 taxas consecutivas em tendência (subindo ou descendo).")
            break

    # Texto
    texto = f"""
**Carta U**
- Taxa média de defeitos por unidade (ū): {u_barra:.4f}

**Resultados dos testes**
"""

    if testes:
        texto += "\n".join(testes)
    else:
        texto += "✅ Processo dentro dos padrões esperados (nenhum alarme nos testes aplicados)."

    texto += "\n"
    if testes:
        texto += "⚠ Recomenda-se investigar causas especiais e revisar estabilidade do processo."
    else:
        texto += "✅ O processo está estável no momento da análise."

    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(u.index, u.values, marker='o', label='Taxa de defeitos')
    ax.plot(u.index, LSC, 'r--', label='LSC')
    ax.plot(u.index, LIC, 'r--', label='LIC')
    ax.axhline(u_barra, color='black', linestyle='-', label='Média (ū)')
    ax.set_title("Carta U")
    ax.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return texto.strip(), grafico_base64

ANALISES = {
    "Carta I-MR": analise_carta_imr,
    "Carta X-Barra R": analise_carta_xbarra_r, 
    "Carta X-Barra S": analise_carta_xbarra_s, 
    "Carta P": analise_carta_p,
    "Carta NP": analise_carta_np,
    "Carta C": analise_carta_c,
    "Carta U": analise_carta_u

    
}


