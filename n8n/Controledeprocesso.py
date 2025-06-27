from suporte import *

def analise_carta_imr(df, coluna_y):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A Carta I-MR requer uma coluna Y válida.", None

    dados = df[coluna_y].dropna()
    if len(dados) < 5:
        return "❌ É necessário pelo menos 5 dados para gerar a Carta I-MR.", None

    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    media_I = np.mean(dados)
    mr = np.abs(np.diff(dados))
    media_MR = np.mean(mr)

    d2 = 1.128
    sigma = media_MR / d2 if d2 != 0 else 0
    LSC_I = media_I + 3 * sigma
    LIC_I = media_I - 3 * sigma
    LSC_MR = 3.267 * media_MR
    LIC_MR = 0

    testes = []

    fora_limite = np.where((dados > LSC_I) | (dados < LIC_I))[0]
    if len(fora_limite) > 0:
        testes.append(f"🔴 {len(fora_limite)} ponto(s) fora dos limites de controle.")

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
        texto += "\n⚠ Recomenda-se investigar causas especiais e revisar estabilidade do processo."
    else:
        texto += "✅ Processo dentro dos padrões esperados (nenhum alarme nos testes aplicados).\n✅ O processo está estável no momento da análise."

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(dados, marker='o')
    ax1.axhline(media_I, color='black', linestyle='-', label='Média')
    ax1.axhline(LSC_I, color='red', linestyle='--', label='LSC')
    ax1.axhline(LIC_I, color='red', linestyle='--', label='LIC')
    ax1.set_title("Carta I")
    ax1.legend()

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


def analise_carta_xbarra_r(df, coluna_y, subgrupo):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A Carta X-Barra R requer uma coluna Y válida.", None

    if not subgrupo or subgrupo not in df.columns:
        return f"❌ A coluna de subgrupo '{subgrupo}' não foi encontrada.", None

    dados = df[[coluna_y, subgrupo]].dropna()
    if dados.shape[0] < 5:
        return "❌ É necessário pelo menos 5 dados para gerar a Carta X-Barra R.", None

    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    grupos = dados.groupby(subgrupo)[coluna_y]
    medias = grupos.mean()
    ranges = grupos.max() - grupos.min()
    n_sub = grupos.size().mean()

    if n_sub < 2:
        return "❌ Cada subgrupo deve ter pelo menos 2 elementos.", None

    A2, D3, D4 = 0.577, 0, 2.114

    media_X = medias.mean()
    media_R = ranges.mean()

    LSC_X = media_X + A2 * media_R
    LIC_X = media_X - A2 * media_R
    LSC_R = D4 * media_R
    LIC_R = D3 * media_R

    testes = []

    fora_limite = medias[(medias > LSC_X) | (medias < LIC_X)]
    if not fora_limite.empty:
        testes.append(f"🔴 {fora_limite.shape[0]} média(s) de subgrupo fora dos limites de controle.")

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
        texto += "\n⚠ Recomenda-se investigar causas especiais e revisar estabilidade do processo."
    else:
        texto += "✅ Processo dentro dos padrões esperados (nenhum alarme nos testes aplicados).\n✅ O processo está estável no momento da análise."

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(medias.index, medias.values, marker='o')
    ax1.axhline(media_X, color='black', linestyle='-', label='Média')
    ax1.axhline(LSC_X, color='red', linestyle='--', label='LSC')
    ax1.axhline(LIC_X, color='red', linestyle='--', label='LIC')
    ax1.set_title("Carta X-Barra")
    ax1.legend()

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

def analise_carta_xbarra_s(df, coluna_y, subgrupo):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A Carta X-Barra S requer uma coluna Y válida.", None

    if not subgrupo or subgrupo not in df.columns:
        return f"❌ A coluna de subgrupo '{subgrupo}' não foi encontrada.", None

    dados = df[[coluna_y, subgrupo]].dropna()
    if dados.shape[0] < 5:
        return "❌ É necessário pelo menos 5 dados para gerar a Carta X-Barra S.", None

    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    grupos = dados.groupby(subgrupo)[coluna_y]
    medias = grupos.mean()
    desvios = grupos.std(ddof=1)
    n_sub = grupos.size().mean()

    if n_sub < 2:
        return "❌ Cada subgrupo deve ter pelo menos 2 elementos.", None

    A3, B3, B4 = 1.427, 0, 2.089

    media_X = medias.mean()
    media_S = desvios.mean()

    LSC_X = media_X + A3 * media_S
    LIC_X = media_X - A3 * media_S
    LSC_S = B4 * media_S
    LIC_S = B3 * media_S

    testes = []

    fora_limite = medias[(medias > LSC_X) | (medias < LIC_X)]
    if not fora_limite.empty:
        testes.append(f"🔴 {fora_limite.shape[0]} média(s) de subgrupo fora dos limites de controle.")

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
        texto += "\n⚠ Recomenda-se investigar causas especiais e revisar estabilidade do processo."
    else:
        texto += "✅ Processo dentro dos padrões esperados (nenhum alarme nos testes aplicados).\n✅ O processo está estável no momento da análise."

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(medias.index, medias.values, marker='o')
    ax1.axhline(media_X, color='black', linestyle='-', label='Média')
    ax1.axhline(LSC_X, color='red', linestyle='--', label='LSC')
    ax1.axhline(LIC_X, color='red', linestyle='--', label='LIC')
    ax1.set_title("Carta X-Barra")
    ax1.legend()

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

def analise_carta_p(df, coluna_y, subgrupo):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A Carta P requer uma coluna com os dados de não conformes.", None

    if not subgrupo or subgrupo not in df.columns:
        return f"❌ A coluna de tamanho de amostra '{subgrupo}' não foi encontrada.", None

    dados = df[[coluna_y, subgrupo]].dropna()
    if dados.shape[0] < 5:
        return "❌ É necessário pelo menos 5 subgrupos para gerar a Carta P.", None

    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    nc = dados[coluna_y].astype(float)
    n = dados[subgrupo].astype(float)
    p = nc / n
    p_barra = nc.sum() / n.sum()
    sigma_p = np.sqrt(p_barra * (1 - p_barra) / n)
    LSC = p_barra + 3 * sigma_p
    LIC = p_barra - 3 * sigma_p
    LIC = np.clip(LIC, 0, None)

    testes = []
    fora_limite = p[(p > LSC) | (p < LIC)]
    if not fora_limite.empty:
        testes.append(f"🔴 {fora_limite.shape[0]} subgrupo(s) com proporção fora dos limites de controle.")

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

    texto = f"""
**Carta P**
- Proporção média de não conformes (p̄): {p_barra:.4f}

**Resultados dos testes**
"""

    if testes:
        texto += "\n".join(testes)
        texto += "\n⚠ Recomenda-se investigar causas especiais e revisar estabilidade do processo."
    else:
        texto += "✅ Processo dentro dos padrões esperados (nenhum alarme nos testes aplicados).\n✅ O processo está estável no momento da análise."

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(p.index, p.values, marker='o', label='Proporção')
    ax.plot(p.index, LSC, color='red', linestyle='--', label='LSC')
    ax.plot(p.index, LIC, color='red', linestyle='--', label='LIC')
    ax.axhline(p_barra, color='black', linestyle='-', label='Média (p̄)')
    ax.set_title("Carta P")
    ax.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return texto.strip(), grafico_base64



def analise_carta_np(df, coluna_y, subgrupo):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A Carta np requer uma coluna Y com número de não conformes.", None

    if not subgrupo or subgrupo not in df.columns:
        return f"❌ A coluna de tamanho de amostra '{subgrupo}' não foi encontrada.", None

    dados = df[[coluna_y, subgrupo]].dropna()
    if dados.shape[0] < 5:
        return "❌ É necessário pelo menos 5 subgrupos para gerar a Carta np.", None

    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    nc = dados[coluna_y].astype(float)
    n = dados[subgrupo].astype(float)

    if n.nunique() > 1:
        return "❌ A Carta np requer que todos os subgrupos tenham o mesmo tamanho de amostra.", None

    n_subgrupo = n.iloc[0]
    p = nc / n_subgrupo
    p_barra = p.mean()
    np_barra = p_barra * n_subgrupo
    sigma_np = np.sqrt(n_subgrupo * p_barra * (1 - p_barra))
    LSC = np_barra + 3 * sigma_np
    LIC = np_barra - 3 * sigma_np
    LIC = np.clip(LIC, 0, None)

    testes = []
    fora_limite = nc[(nc > LSC) | (nc < LIC)]
    if not fora_limite.empty:
        testes.append(f"🔴 {fora_limite.shape[0]} subgrupo(s) com contagem fora dos limites de controle.")

    lado = np.where(nc > np_barra, 1, -1)
    conta = 0
    for i in range(len(lado)):
        if i == 0 or lado[i] == lado[i-1]:
            conta += 1
            if conta >= 9:
                testes.append("🟠 9 pontos consecutivos no mesmo lado da média.")
                break
        else:
            conta = 1

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
            testes.append("🟡 6 pontos consecutivos em tendência (subindo ou descendo).")
            break

    texto = f"""
**Carta np**
- Proporção média de não conformes (p̄): {p_barra:.4f}
- Média (np̄): {np_barra:.2f}

**Resultados dos testes**
"""

    if testes:
        texto += "\n".join(testes)
        texto += "\n⚠ Recomenda-se investigar causas especiais e revisar estabilidade do processo."
    else:
        texto += "✅ Processo dentro dos padrões esperados (nenhum alarme nos testes aplicados).\n✅ O processo está estável no momento da análise."

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(nc.index, nc.values, marker='o', label='Não conformes')
    ax.hlines(LSC, xmin=0, xmax=len(nc)-1, colors='red', linestyles='--', label='LSC')
    ax.hlines(LIC, xmin=0, xmax=len(nc)-1, colors='red', linestyles='--', label='LIC')
    ax.axhline(np_barra, color='black', linestyle='-', label='Média (np̄)')
    ax.set_title("Carta np")
    ax.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return texto.strip(), grafico_base64


def analise_carta_c(df, coluna_y):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A Carta C requer uma coluna Y válida.", None

    dados = df[coluna_y].dropna()
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

    testes = []

    fora_limite = dados[(dados > LSC) | (dados < LIC)]
    if len(fora_limite) > 0:
        testes.append(f"🔴 {len(fora_limite)} unidade(s) com defeitos fora dos limites de controle.")

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

    texto = f"""
**Carta C**
- Número médio de defeitos (c̄): {c_barra:.4f}
- Limites: LSC={LSC:.4f}, LIC={LIC:.4f}

**Resultados dos testes**
"""

    if testes:
        texto += "\n".join(testes)
        texto += "\n⚠ Recomenda-se investigar causas especiais e revisar estabilidade do processo."
    else:
        texto += "✅ Processo dentro dos padrões esperados (nenhum alarme nos testes aplicados).\n✅ O processo está estável no momento da análise."

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


def analise_carta_u(df, coluna_y, subgrupo):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A Carta U requer uma coluna Y válida.", None

    if not subgrupo or subgrupo not in df.columns:
        return f"❌ A coluna de subgrupo '{subgrupo}' não foi encontrada.", None

    dados = df[[coluna_y, subgrupo]].dropna()
    if dados.shape[0] < 5:
        return "❌ É necessário pelo menos 5 dados para gerar a Carta U.", None

    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    grupos = dados.groupby(subgrupo)[coluna_y]
    defeitos = grupos.sum()
    tamanhos = grupos.count()

    u = defeitos / tamanhos
    u_barra = defeitos.sum() / tamanhos.sum()

    sigma_u = np.sqrt(u_barra / tamanhos)
    LSC = u_barra + 3 * sigma_u
    LIC = np.clip(u_barra - 3 * sigma_u, 0, None)

    testes = []

    fora_limite = u[(u > LSC) | (u < LIC)]
    if not fora_limite.empty:
        testes.append(f"🔴 {fora_limite.shape[0]} subgrupo(s) com taxa fora dos limites de controle.")

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

    texto = f"""
**Carta U**
- Taxa média de defeitos por unidade (ū): {u_barra:.4f}

**Resultados dos testes**
"""

    if testes:
        texto += "\n".join(testes)
        texto += "\n⚠ Recomenda-se investigar causas especiais e revisar estabilidade do processo."
    else:
        texto += "✅ Processo dentro dos padrões esperados (nenhum alarme nos testes aplicados).\n✅ O processo está estável no momento da análise."

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(u.index, u.values, marker='o', label='Taxa de defeitos')
    ax.hlines(LSC, xmin=0, xmax=len(p)-1, colors='red', linestyles='--', label='LSC')
    ax.hlines(LIC, xmin=0, xmax=len(p)-1, colors='red', linestyles='--', label='LIC')
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
    "Carta X-BarraR": analise_carta_xbarra_r, 
    "Carta X-BarraS": analise_carta_xbarra_s, 
    "Carta P": analise_carta_p,
    "Carta NP": analise_carta_np,
    "Carta C": analise_carta_c,
    "Carta U": analise_carta_u

    
}


