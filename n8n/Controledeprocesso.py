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
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    # Carta Individual (I)
    y = dados[nome_coluna_y].values
    x = dados["Subgrupo"].values
    axs[0].plot(x, y, color="black", linestyle="-")
    axs[0].axhline(media, color="green", linestyle="-")
    axs[0].axhline(UCL_I, color="red", linestyle="-")
    axs[0].axhline(LCL_I, color="red", linestyle="-")
    axs[0].set_title(f"Carta I de {nome_coluna_y}", fontsize=18, fontweight='bold')
    axs[0].set_ylabel("Valor Individual", fontsize=16, fontweight='bold')
    axs[0].set_xlabel(nome_coluna_y, fontsize=16, fontweight='bold')
    axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    xlim = axs[0].get_xlim()
    axs[0].text(xlim[1]+1, media, f"X̄ = {media:.3f}", va='center', fontsize=12, color="green")
    axs[0].text(xlim[1]+1, UCL_I, f"LSC = {UCL_I:.3f}", va='center', fontsize=12, color="red")
    axs[0].text(xlim[1]+1, LCL_I, f"LIC = {LCL_I:.3f}", va='center', fontsize=12, color="red")

    # Critérios 2 e 3 (funções revisadas)
    def check_crit2(y, ref_media):
        count = 0
        seq = []
        for i, val in enumerate(y):
            if val > ref_media:
                if count >= 0:
                    count += 1
                    seq.append(i+1)
                else:
                    count = 1
                    seq = [i+1]
            elif val < ref_media:
                if count <= 0:
                    count -= 1
                    seq.append(i+1)
                else:
                    count = -1
                    seq = [i+1]
            else:
                count = 0
                seq = []
            if abs(count) >= 9:
                return True, seq[-9:]
        return False, []

    def check_crit3(y):
        count_up = 0
        seq_up = []
        count_down = 0
        seq_down = []
        for i in range(1, len(y)):
            if y[i] > y[i-1]:
                count_up += 1
                seq_up.append(i+1)
                count_down = 0
                seq_down = []
                if count_up >= 6:
                    return True, seq_up[-6:]
            elif y[i] < y[i-1]:
                count_down += 1
                seq_down.append(i+1)
                count_up = 0
                seq_up = []
                if count_down >= 6:
                    return True, seq_down[-6:]
            else:
                count_up = 0
                seq_up = []
                count_down = 0
                seq_down = []
        return False, []

    # Critérios Carta I
    crit2_I, linhas_crit2_I = check_crit2(y, media)
    crit3_I, linhas_crit3_I = check_crit3(y)

    # Critério 1 – Carta I
    crit1_flag_I = []
    for idx, (xi, yi) in enumerate(zip(x, y)):
        cor = "black"
        if yi > UCL_I or yi < LCL_I:
            cor = "red"
            crit1_flag_I.append((idx+1, yi))
        elif crit2_I and (idx+1) in linhas_crit2_I:
            cor = "red"
        elif crit3_I and (idx+1) in linhas_crit3_I:
            cor = "red"
        axs[0].scatter(xi, yi, color=cor)

    # Carta MR
    x_mr = dados["Subgrupo"].values[1:]
    y_mr = mr[1:].values
    axs[1].plot(x_mr, y_mr, color="black", linestyle="-")
    axs[1].axhline(mr_mean, color="green", linestyle="-")
    axs[1].axhline(UCL_MR, color="red", linestyle="-")
    axs[1].axhline(0, color="red", linestyle="-")  # LIC=0 linha vermelha
    axs[1].set_title("Carta MR", fontsize=18, fontweight='bold')
    axs[1].set_ylabel("Amplitude Móvel", fontsize=16, fontweight='bold')
    axs[1].set_xlabel(nome_coluna_y, fontsize=16, fontweight='bold')
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    xlim_mr = axs[1].get_xlim()
    axs[1].text(xlim_mr[1]+1, mr_mean, f"MR̄ = {mr_mean:.3f}", va='center', fontsize=12, color="green")
    axs[1].text(xlim_mr[1]+1, UCL_MR, f"LSC = {UCL_MR:.3f}", va='center', fontsize=12, color="red")
    axs[1].text(xlim_mr[1]+1, 0, f"LIC = 0.000", va='center', fontsize=12, color="red")

    # Critérios Carta MR
    crit2_MR, linhas_crit2_MR = check_crit2(y_mr, mr_mean)
    crit3_MR, linhas_crit3_MR = check_crit3(y_mr)

    # Critério 1 – Carta MR
    crit1_flag_MR = []
    for idx, (xi, yi) in enumerate(zip(x_mr, y_mr)):
        cor = "black"
        if yi > UCL_MR:
            cor = "red"
            crit1_flag_MR.append((xi, yi))
        elif crit2_MR and (idx+2) in linhas_crit2_MR:
            cor = "red"
        elif crit3_MR and (idx+2) in linhas_crit3_MR:
            cor = "red"
        axs[1].scatter(xi, yi, color=cor)

    plt.tight_layout()

    # 🔷 REPORT – Individual
    texto_I = f"📊 **Carta I ({nome_coluna_y})**\n"
    texto_I += "🔎 **Critérios avaliados:**\n"
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
        texto_I += "🔎 **Conclusão:** Causa especial detectada. O processo não está sob controle estatístico.\n"
        texto_I += "🔎 **Recomendação:** Investigue o processo para entender e se possível remover a causa especial identificada.\n"
    else:
        texto_I += "🔎 **Conclusão:** Processo está estável.\n"

    # 🔷 REPORT – MR
    texto_MR = f"📊 **Carta MR**\n"
    texto_MR += "🔎 **Critérios avaliados:**\n"
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
        texto_MR += "🔎 **Conclusão:** Causa especial detectada. O processo não está sob controle estatístico.\n"
        texto_MR += "🔎 **Recomendação:** Investigue o processo para entender e se possível remover a causa especial identificada.\n"
    else:
        texto_MR += "🔎 **Conclusão:** Processo está estável.\n"

    # Salva gráfico
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return (texto_I + "\n" + texto_MR), img_base64



def analise_carta_xbarra_r(df, coluna_y, subgrupo):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from io import BytesIO
    import base64

    aplicar_estilo_minitab()

    # 🔷 Validação coluna Y e subgrupo
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A Carta X-Barra R requer uma coluna Y válida.", None

    if not subgrupo or subgrupo not in df.columns:
        return f"❌ A coluna de subgrupo '{subgrupo}' não foi encontrada.", None

    dados = df[[coluna_y, subgrupo]].dropna()
    if dados.shape[0] < 5:
        return "❌ É necessário pelo menos 5 dados para gerar a Carta X-Barra R.", None

    grupos = dados.groupby(subgrupo)[coluna_y]
    medias = grupos.mean()
    ranges = grupos.max() - grupos.min()
    n_sub = grupos.size().mean()

    if n_sub < 2:
        return "❌ Cada subgrupo deve ter pelo menos 2 elementos.", None

    # 🔷 Constantes
    A2, D3, D4 = 0.577, 0, 2.114

    media_X = medias.mean()
    media_R = ranges.mean()

    LSC_X = media_X + A2 * media_R
    LIC_X = media_X - A2 * media_R
    LSC_R = D4 * media_R
    LIC_R = D3 * media_R

    # 🔷 Gráficos
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Carta X-Barra
    ax1.plot(medias.index, medias.values, marker='o', color='black')
    ax1.axhline(media_X, color='green', linestyle='-')
    ax1.axhline(LSC_X, color='red', linestyle='-')
    ax1.axhline(LIC_X, color='red', linestyle='-')
    ax1.set_title(f"Carta X̄ de {coluna_y}", fontsize=18, fontweight='bold')
    ax1.set_ylabel("Média Subgrupo", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Subgrupo", fontsize=16, fontweight='bold')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Critérios
    crit1_X = medias[(medias > LSC_X) | (medias < LIC_X)]
    crit2_X, linhas_crit2_X = False, []
    lado = np.where(medias > media_X, 1, -1)
    count = 0
    seq = []
    for i in range(len(lado)):
        if i == 0 or lado[i] == lado[i-1]:
            count += 1
            seq.append(i+1)
            if count >= 9:
                crit2_X = True
                linhas_crit2_X = seq[-9:]
                break
        else:
            count = 1
            seq = [i+1]

    crit3_X, linhas_crit3_X = False, []
    count_up = count_down = 0
    seq_up = seq_down = []
    for i in range(1, len(medias)):
        if medias.iloc[i] > medias.iloc[i-1]:
            count_up += 1
            seq_up.append(i+1)
            count_down = 0
            seq_down = []
            if count_up >= 6:
                crit3_X = True
                linhas_crit3_X = seq_up[-6:]
                break
        elif medias.iloc[i] < medias.iloc[i-1]:
            count_down += 1
            seq_down.append(i+1)
            count_up = 0
            seq_up = []
            if count_down >= 6:
                crit3_X = True
                linhas_crit3_X = seq_down[-6:]
                break
        else:
            count_up = count_down = 0
            seq_up = seq_down = []

    # Pontos vermelhos Carta X-Barra
    for idx, val in enumerate(medias):
        cor = "black"
        if (medias.index[idx] in crit1_X.index):
            cor = "red"
        elif crit2_X and (idx+1) in linhas_crit2_X:
            cor = "red"
        elif crit3_X and (idx+1) in linhas_crit3_X:
            cor = "red"
        ax1.scatter(medias.index[idx], val, color=cor)

    # Carta R
    ax2.plot(ranges.index, ranges.values, marker='o', color='black')
    ax2.axhline(media_R, color='green', linestyle='-')
    ax2.axhline(LSC_R, color='red', linestyle='-')
    ax2.axhline(LIC_R, color='red', linestyle='-')
    ax2.set_title("Carta R", fontsize=18, fontweight='bold')
    ax2.set_ylabel("Amplitude", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Subgrupo", fontsize=16, fontweight='bold')
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Critérios Carta R
    crit1_R = ranges[(ranges > LSC_R) | (ranges < LIC_R)]
    crit2_R, linhas_crit2_R = False, []
    lado_r = np.where(ranges > media_R, 1, -1)
    count_r = 0
    seq_r = []
    for i in range(len(lado_r)):
        if i == 0 or lado_r[i] == lado_r[i-1]:
            count_r += 1
            seq_r.append(i+1)
            if count_r >= 9:
                crit2_R = True
                linhas_crit2_R = seq_r[-9:]
                break
        else:
            count_r = 1
            seq_r = [i+1]

    crit3_R, linhas_crit3_R = False, []
    count_up_r = count_down_r = 0
    seq_up_r = seq_down_r = []
    for i in range(1, len(ranges)):
        if ranges.iloc[i] > ranges.iloc[i-1]:
            count_up_r += 1
            seq_up_r.append(i+1)
            count_down_r = 0
            seq_down_r = []
            if count_up_r >= 6:
                crit3_R = True
                linhas_crit3_R = seq_up_r[-6:]
                break
        elif ranges.iloc[i] < ranges.iloc[i-1]:
            count_down_r += 1
            seq_down_r.append(i+1)
            count_up_r = 0
            seq_up_r = []
            if count_down_r >= 6:
                crit3_R = True
                linhas_crit3_R = seq_down_r[-6:]
                break
        else:
            count_up_r = count_down_r = 0
            seq_up_r = seq_down_r = []

    # Pontos vermelhos Carta R
    for idx, val in enumerate(ranges):
        cor = "black"
        if (ranges.index[idx] in crit1_R.index):
            cor = "red"
        elif crit2_R and (idx+1) in linhas_crit2_R:
            cor = "red"
        elif crit3_R and (idx+1) in linhas_crit3_R:
            cor = "red"
        ax2.scatter(ranges.index[idx], val, color=cor)

    plt.tight_layout()

    # 🔷 Reporte final
    texto_X = f"📊 **Carta X̄ ({coluna_y})**\n🔎 **Critérios avaliados:**\n"
    texto_X += f"1. Critério 1 – Pontos fora dos limites: {'❌ Detectado' if not crit1_X.empty else '✅ OK'}\n"
    texto_X += f"2. Critério 2 – 9 pontos do mesmo lado da média: {'❌ Detectado (Subgrupos ' + ', '.join(map(str, linhas_crit2_X)) + ')' if crit2_X else '✅ OK'}\n"
    texto_X += f"3. Critério 3 – 6 pontos subindo ou descendo: {'❌ Detectado (Subgrupos ' + ', '.join(map(str, linhas_crit3_X)) + ')' if crit3_X else '✅ OK'}\n"
    if not crit1_X.empty or crit2_X or crit3_X:
        texto_X += "🔎 **Conclusão:** Causa especial detectada. O processo não está sob controle estatístico.\n🔎 **Recomendação:** Investigue o processo para entender e se possível remover a causa especial identificada.\n"
    else:
        texto_X += "🔎 **Conclusão:** Processo está estável.\n"

    texto_R = f"📊 **Carta R**\n🔎 **Critérios avaliados:**\n"
    texto_R += f"1. Critério 1 – Pontos fora dos limites: {'❌ Detectado' if not crit1_R.empty else '✅ OK'}\n"
    texto_R += f"2. Critério 2 – 9 pontos do mesmo lado da média: {'❌ Detectado (Subgrupos ' + ', '.join(map(str, linhas_crit2_R)) + ')' if crit2_R else '✅ OK'}\n"
    texto_R += f"3. Critério 3 – 6 pontos subindo ou descendo: {'❌ Detectado (Subgrupos ' + ', '.join(map(str, linhas_crit3_R)) + ')' if crit3_R else '✅ OK'}\n"
    if not crit1_R.empty or crit2_R or crit3_R:
        texto_R += "🔎 **Conclusão:** Causa especial detectada. O processo não está sob controle estatístico.\n🔎 **Recomendação:** Investigue o processo para entender e se possível remover a causa especial identificada.\n"
    else:
        texto_R += "🔎 **Conclusão:** Processo está estável.\n"

    # 🔷 Salvar imagem
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return texto_X + "\n" + texto_R, grafico_base64




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


