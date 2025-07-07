from suporte import *

def analise_carta_imr(df, coluna_y):
    import matplotlib.pyplot as plt
    import numpy as np
    from io import BytesIO
    import base64
    import pandas as pd

    aplicar_estilo_minitab()

    nome_coluna_y = coluna_y if isinstance(coluna_y, str) else (coluna_y[0] if coluna_y else None)
    if not nome_coluna_y or nome_coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    if not pd.api.types.is_numeric_dtype(df[nome_coluna_y]):
        return f"❌ A coluna '{nome_coluna_y}' contém dados não numéricos e não pode ser usada na análise.", None

    dados = df[[nome_coluna_y]].dropna().copy()
    dados["Subgrupo"] = range(1, len(dados) + 1)

    if dados.empty:
        return "❌ Dados insuficientes para análise.", None

    # Estatísticas
    media = dados[nome_coluna_y].mean()
    sigma = dados[nome_coluna_y].std()
    UCL_I = media + 3 * sigma
    LCL_I = media - 3 * sigma

    mr = dados[nome_coluna_y].diff().abs()
    mr_mean = mr[1:].mean()
    UCL_MR = mr_mean * 3.267

    # Gráfico estilo Minitab
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Carta Individual (I)
    y = dados[nome_coluna_y].values
    x = dados["Subgrupo"].values
    axs[0].plot(x, y, color="black", linestyle="-")
    axs[0].scatter(x, y, color="black")
    axs[0].axhline(media, color="green", linestyle="-")
    axs[0].axhline(UCL_I, color="red", linestyle="-")
    axs[0].axhline(LCL_I, color="red", linestyle="-")
    axs[0].set_title(f"Carta I de {nome_coluna_y}", fontsize=18)
    axs[0].set_ylabel("Valor Individual", fontsize=16)
    axs[0].set_xlabel(nome_coluna_y, fontsize=16)  # alteração aqui

    axs[0].set_xticks(x)  # mostrar todos os números no eixo X

    xlim = axs[0].get_xlim()
    axs[0].text(xlim[1]+1, media, f"X̄ = {media:.3f}", va='center', fontsize=12, color="green")
    axs[0].text(xlim[1]+1, UCL_I, f"LSC = {UCL_I:.3f}", va='center', fontsize=12, color="red")
    axs[0].text(xlim[1]+1, LCL_I, f"LIC = {LCL_I:.3f}", va='center', fontsize=12, color="red")

    crit1_flag_I = []
    for idx, (xi, yi) in enumerate(zip(x, y)):
        if yi > UCL_I or yi < LCL_I:
            axs[0].scatter(xi, yi, color="red")
            crit1_flag_I.append((idx+1, yi))  # salva linha e valor

    # Carta MR
    x_mr = dados["Subgrupo"].values[1:]
    y_mr = mr[1:].values
    axs[1].plot(x_mr, y_mr, color="black", linestyle="-")
    axs[1].scatter(x_mr, y_mr, color="black")
    axs[1].axhline(mr_mean, color="green", linestyle="-")
    axs[1].axhline(UCL_MR, color="red", linestyle="-")
    axs[1].set_title("Carta MR", fontsize=18)
    axs[1].set_ylabel("Amplitude Móvel", fontsize=16)
    axs[1].set_xlabel(nome_coluna_y, fontsize=16)  # alteração aqui

    axs[1].set_xticks(x_mr)  # mostrar todos os números no eixo X

    xlim_mr = axs[1].get_xlim()
    axs[1].text(xlim_mr[1]+1, mr_mean, f"MR̄ = {mr_mean:.3f}", va='center', fontsize=12, color="green")
    axs[1].text(xlim_mr[1]+1, UCL_MR, f"LSC = {UCL_MR:.3f}", va='center', fontsize=12, color="red")
    axs[1].text(xlim_mr[1]+1, 0, f"LIC = 0.000", va='center', fontsize=12, color="red")

    crit1_flag_MR = []
    for idx, (xi, yi) in enumerate(zip(x_mr, y_mr)):
        if yi > UCL_MR:
            axs[1].scatter(xi, yi, color="red")
            crit1_flag_MR.append((xi, yi))  # salva linha e valor

    plt.tight_layout()

    # Critérios 2 e 3 (sequências)
    def check_crit2(y):
        count = 1
        lados = []
        for i in range(1, len(y)):
            if (y[i] > media and y[i-1] > media) or (y[i] < media and y[i-1] < media):
                count += 1
                lados.append(i+1)
                if count >= 9:
                    return True, lados[-9:]
            else:
                count = 1
                lados = [i+1] if (y[i] > media or y[i] < media) else []
        return False, []

    def check_crit3(y):
        count_up = 1
        seq_up = [1]
        count_down = 1
        seq_down = [1]
        for i in range(1, len(y)):
            if y[i] > y[i-1]:
                count_up += 1
                seq_up.append(i+1)
                count_down = 1
                seq_down = [i+1]
                if count_up >= 6:
                    return True, seq_up[-6:]
            elif y[i] < y[i-1]:
                count_down += 1
                seq_down.append(i+1)
                count_up = 1
                seq_up = [i+1]
                if count_down >= 6:
                    return True, seq_down[-6:]
            else:
                count_up = 1
                seq_up = [i+1]
                count_down = 1
                seq_down = [i+1]
        return False, []

    crit2_I, linhas_crit2_I = check_crit2(y)
    crit3_I, linhas_crit3_I = check_crit3(y)
    crit2_MR, linhas_crit2_MR = check_crit2(y_mr)
    crit3_MR, linhas_crit3_MR = check_crit3(y_mr)

    # 🔷 REPORT – Individual
    texto_I = f"📊 **Carta I ({nome_coluna_y})**\n"
    texto_I += "🔎 Critérios avaliados:\n"
    if crit1_flag_I:
        pontos = ", ".join([f"Linha {linha}: {valor:.2f}" for linha, valor in crit1_flag_I])
        texto_I += f"1. Critério 1 – Pontos fora dos limites: ❌ Detectado ({pontos})\n"
    else:
        texto_I += "1. Critério 1 – Pontos fora dos limites: ✅ OK\n"

    if crit2_I:
        linhas = ", ".join([str(l) for l in linhas_crit2_I])
        texto_I += f"2. Critério 2 – 9 pontos do mesmo lado da média: ❌ Detectado (Linhas {linhas})\n"
    else:
        texto_I += "2. Critério 2 – 9 pontos do mesmo lado da média: ✅ OK\n"

    if crit3_I:
        linhas = ", ".join([str(l) for l in linhas_crit3_I])
        texto_I += f"3. Critério 3 – 6 pontos subindo ou descendo: ❌ Detectado (Linhas {linhas})\n"
    else:
        texto_I += "3. Critério 3 – 6 pontos subindo ou descendo: ✅ OK\n"

    if crit1_flag_I or crit2_I or crit3_I:
        texto_I += "🔎 Conclusão: Causa especial detectada. O processo não está sob controle estatístico.\n"
        texto_I += "🔎 Recomendação: Investigue o processo para entender e se possível remover a causa especial identificada.\n"
    else:
        texto_I += "🔎 Conclusão: Processo está estável.\n"

    # 🔷 REPORT – MR
    texto_MR = f"📊 **Carta MR**\n"
    texto_MR += "🔎 Critérios avaliados:\n"
    if crit1_flag_MR:
        pontos = ", ".join([f"Linha {linha}: {valor:.2f}" for linha, valor in crit1_flag_MR])
        texto_MR += f"1. Critério 1 – Pontos fora dos limites: ❌ Detectado ({pontos})\n"
    else:
        texto_MR += "1. Critério 1 – Pontos fora dos limites: ✅ OK\n"

    if crit2_MR:
        linhas = ", ".join([str(l) for l in linhas_crit2_MR])
        texto_MR += f"2. Critério 2 – 9 pontos do mesmo lado da média: ❌ Detectado (Linhas {linhas})\n"
    else:
        texto_MR += "2. Critério 2 – 9 pontos do mesmo lado da média: ✅ OK\n"

    if crit3_MR:
        linhas = ", ".join([str(l) for l in linhas_crit3_MR])
        texto_MR += f"3. Critério 3 – 6 pontos subindo ou descendo: ❌ Detectado (Linhas {linhas})\n"
    else:
        texto_MR += "3. Critério 3 – 6 pontos subindo ou descendo: ✅ OK\n"

    if crit1_flag_MR or crit2_MR or crit3_MR:
        texto_MR += "🔎 Conclusão: Causa especial detectada. O processo não está sob controle estatístico.\n"
        texto_MR += "🔎 Recomendação: Investigue o processo para entender e se possível remover a causa especial identificada.\n"
    else:
        texto_MR += "🔎 Conclusão: Processo está estável.\n"

    # Salva gráfico
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    # Retorna os dois relatórios juntos + imagem
    return (texto_I + "\n" + texto_MR), img_base64




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
    from suporte import aplicar_estilo_minitab

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

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(u.index, u.values, marker='o', label='Taxa de defeitos')
    ax.plot(u.index, LSC, linestyle='--', color='red', label='LSC')
    ax.plot(u.index, LIC, linestyle='--', color='red', label='LIC')
    ax.axhline(u_barra, color='black', linestyle='-', label='Média (ū)')
    ax.set_title("Carta U")
    ax.set_ylabel("Taxa de Defeitos por Unidade")
    ax.set_xlabel("Subgrupos")
    ax.legend()
    aplicar_estilo_minitab()

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


