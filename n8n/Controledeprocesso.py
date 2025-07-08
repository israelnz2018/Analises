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

    nome_coluna_y = coluna_y if isinstance(coluna_y, str) else (coluna_y[0] if coluna_y else None)
    if not nome_coluna_y or nome_coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    if not pd.api.types.is_numeric_dtype(df[nome_coluna_y]):
        return f"❌ A coluna '{nome_coluna_y}' contém dados não numéricos e não pode ser usada na análise.", None

    # 🔷 Verificação do subgrupo como coluna existente
    if not subgrupo or subgrupo not in df.columns:
        return f"❌ A coluna de subgrupo '{subgrupo}' não foi encontrada no arquivo.", None

    dados = df[[nome_coluna_y, subgrupo]].dropna().copy()

    if dados.empty:
        return "❌ Dados insuficientes para análise.", None

    # Agrupamento em subgrupos (usando a coluna de subgrupo existente)
    grupos = dados.groupby(subgrupo)[nome_coluna_y]

    xbar = grupos.mean()
    r = grupos.max() - grupos.min()
    n = grupos.size().mean()

    # Constantes para Xbarra-R
    A2_table = {2:1.88, 3:1.023, 4:0.729, 5:0.577, 6:0.483, 7:0.419, 8:0.373, 9:0.337, 10:0.308}
    D3_table = {2:0, 3:0, 4:0, 5:0, 6:0, 7:0.076, 8:0.136, 9:0.184, 10:0.223}
    D4_table = {2:3.267, 3:2.574, 4:2.282, 5:2.114, 6:2.004, 7:1.924, 8:1.864, 9:1.816, 10:1.777}

    A2 = A2_table.get(int(n), 0.577)
    D3 = D3_table.get(int(n), 0)
    D4 = D4_table.get(int(n), 2.114)

    xbar_bar = xbar.mean()
    r_bar = r.mean()

    UCL_X = xbar_bar + A2 * r_bar
    LCL_X = xbar_bar - A2 * r_bar

    UCL_R = D4 * r_bar
    LCL_R = D3 * r_bar

    # Gráficos
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    # Carta Xbarra
    x = xbar.index.values
    y = xbar.values
    axs[0].plot(x, y, color="black", linestyle="-")
    axs[0].axhline(xbar_bar, color="green", linestyle="-")
    axs[0].axhline(UCL_X, color="red", linestyle="-")
    axs[0].axhline(LCL_X, color="red", linestyle="-")
    axs[0].set_title(f"Carta X̄ de {nome_coluna_y}", fontsize=18, fontweight='bold')
    axs[0].set_ylabel("Média Subgrupo", fontsize=16, fontweight='bold')
    axs[0].set_xlabel("Subgrupo", fontsize=16, fontweight='bold')
    axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    xlim = axs[0].get_xlim()
    axs[0].text(xlim[1]+1, xbar_bar, f"X̄̄ = {xbar_bar:.3f}", va='center', fontsize=12, color="green")
    axs[0].text(xlim[1]+1, UCL_X, f"LSC = {UCL_X:.3f}", va='center', fontsize=12, color="red")
    axs[0].text(xlim[1]+1, LCL_X, f"LIC = {LCL_X:.3f}", va='center', fontsize=12, color="red")

    # Critérios 2 e 3
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

    crit2_X, linhas_crit2_X = check_crit2(y, xbar_bar)
    crit3_X, linhas_crit3_X = check_crit3(y)

    crit1_flag_X = []
    for idx, (xi, yi) in enumerate(zip(x, y)):
        cor = "black"
        if yi > UCL_X or yi < LCL_X:
            cor = "red"
            crit1_flag_X.append((idx+1, yi))
        elif crit2_X and (idx+1) in linhas_crit2_X:
            cor = "red"
        elif crit3_X and (idx+1) in linhas_crit3_X:
            cor = "red"
        axs[0].scatter(xi, yi, color=cor)

    # Carta R
    x_r = r.index.values
    y_r = r.values
    axs[1].plot(x_r, y_r, color="black", linestyle="-")
    axs[1].axhline(r_bar, color="green", linestyle="-")
    axs[1].axhline(UCL_R, color="red", linestyle="-")
    axs[1].axhline(LCL_R, color="red", linestyle="-")
    axs[1].set_title("Carta R", fontsize=18, fontweight='bold')
    axs[1].set_ylabel("Amplitude", fontsize=16, fontweight='bold')
    axs[1].set_xlabel("Subgrupo", fontsize=16, fontweight='bold')
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    xlim_r = axs[1].get_xlim()
    axs[1].text(xlim_r[1]+1, r_bar, f"R̄ = {r_bar:.3f}", va='center', fontsize=12, color="green")
    axs[1].text(xlim_r[1]+1, UCL_R, f"LSC = {UCL_R:.3f}", va='center', fontsize=12, color="red")
    axs[1].text(xlim_r[1]+1, LCL_R, f"LIC = {LCL_R:.3f}", va='center', fontsize=12, color="red")

    crit2_R, linhas_crit2_R = check_crit2(y_r, r_bar)
    crit3_R, linhas_crit3_R = check_crit3(y_r)

    crit1_flag_R = []
    for idx, (xi, yi) in enumerate(zip(x_r, y_r)):
        cor = "black"
        if yi > UCL_R or yi < LCL_R:
            cor = "red"
            crit1_flag_R.append((idx+1, yi))
        elif crit2_R and (idx+1) in linhas_crit2_R:
            cor = "red"
        elif crit3_R and (idx+1) in linhas_crit3_R:
            cor = "red"
        axs[1].scatter(xi, yi, color=cor)

    plt.tight_layout()

    # Report – Xbarra
    texto_X = f"📊 **Carta X̄ ({nome_coluna_y})**\n"
    texto_X += "🔎 **Critérios avaliados:**\n"
    if crit1_flag_X:
        pontos = ", ".join([f"Subgrupo {linha}: {valor:.2f}" for linha, valor in crit1_flag_X])
        texto_X += f"1. Critério 1 – Pontos fora dos limites: ❌ Detectado ({pontos})\n"
    else:
        texto_X += "1. Critério 1 – Pontos fora dos limites: ✅ OK\n"

    if crit2_X:
        linhas = ", ".join([str(l) for l in linhas_crit2_X])
        texto_X += f"2. Critério 2 – 9 pontos do mesmo lado da média: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto_X += "2. Critério 2 – 9 pontos do mesmo lado da média: ✅ OK\n"

    if crit3_X:
        linhas = ", ".join([str(l) for l in linhas_crit3_X])
        texto_X += f"3. Critério 3 – 6 pontos subindo ou descendo: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto_X += "3. Critério 3 – 6 pontos subindo ou descendo: ✅ OK\n"

    if crit1_flag_X or crit2_X or crit3_X:
        texto_X += "🔎 **Conclusão:** Causa especial detectada. O processo não está sob controle estatístico.\n"
        texto_X += "🔎 **Recomendação:** Investigue o processo para entender e se possível remover a causa especial identificada.\n"
    else:
        texto_X += "🔎 **Conclusão:** Processo está estável.\n"

    # Report – R
    texto_R = f"📊 **Carta R**\n"
    texto_R += "🔎 **Critérios avaliados:**\n"
    if crit1_flag_R:
        pontos = ", ".join([f"Subgrupo {linha}: {valor:.2f}" for linha, valor in crit1_flag_R])
        texto_R += f"1. Critério 1 – Pontos fora dos limites: ❌ Detectado ({pontos})\n"
    else:
        texto_R += "1. Critério 1 – Pontos fora dos limites: ✅ OK\n"

    if crit2_R:
        linhas = ", ".join([str(l) for l in linhas_crit2_R])
        texto_R += f"2. Critério 2 – 9 pontos do mesmo lado da média: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto_R += "2. Critério 2 – 9 pontos do mesmo lado da média: ✅ OK\n"

    if crit3_R:
        linhas = ", ".join([str(l) for l in linhas_crit3_R])
        texto_R += f"3. Critério 3 – 6 pontos subindo ou descendo: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto_R += "3. Critério 3 – 6 pontos subindo ou descendo: ✅ OK\n"

    if crit1_flag_R or crit2_R or crit3_R:
        texto_R += "🔎 **Conclusão:** Causa especial detectada. O processo não está sob controle estatístico.\n"
        texto_R += "🔎 **Recomendação:** Investigue o processo para entender e se possível remover a causa especial identificada.\n"
    else:
        texto_R += "🔎 **Conclusão:** Processo está estável.\n"

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return (texto_X + "\n" + texto_R), img_base64





def analise_carta_xbarra_s(df, coluna_y, subgrupo):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from io import BytesIO
    import base64

    aplicar_estilo_minitab()

    nome_coluna_y = coluna_y if isinstance(coluna_y, str) else (coluna_y[0] if coluna_y else None)
    if not nome_coluna_y or nome_coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    if not pd.api.types.is_numeric_dtype(df[nome_coluna_y]):
        return f"❌ A coluna '{nome_coluna_y}' contém dados não numéricos e não pode ser usada na análise.", None

    # 🔷 Verificação do subgrupo como coluna existente
    if not subgrupo or subgrupo not in df.columns:
        return f"❌ A coluna de subgrupo '{subgrupo}' não foi encontrada no arquivo.", None

    dados = df[[nome_coluna_y, subgrupo]].dropna().copy()

    if dados.empty:
        return "❌ Dados insuficientes para análise.", None

    # Agrupamento em subgrupos
    grupos = dados.groupby(subgrupo)[nome_coluna_y]

    xbar = grupos.mean()
    s = grupos.std()
    n = grupos.size().mean()

    # Constantes para Xbarra-S
    A3_table = {2:2.659, 3:1.954, 4:1.628, 5:1.427, 6:1.287, 7:1.182, 8:1.099, 9:1.032, 10:0.975}
    B3_table = {2:0, 3:0, 4:0, 5:0, 6:0, 7:0.075, 8:0.136, 9:0.184, 10:0.223}
    B4_table = {2:3.267, 3:2.568, 4:2.266, 5:2.089, 6:1.97, 7:1.882, 8:1.815, 9:1.761, 10:1.716}

    A3 = A3_table.get(int(n), 1.023)
    B3 = B3_table.get(int(n), 0)
    B4 = B4_table.get(int(n), 2.114)

    xbar_bar = xbar.mean()
    s_bar = s.mean()

    UCL_X = xbar_bar + A3 * s_bar
    LCL_X = xbar_bar - A3 * s_bar

    UCL_S = B4 * s_bar
    LCL_S = B3 * s_bar

    # Gráficos
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    # Carta Xbarra
    x = xbar.index.values
    y = xbar.values
    axs[0].plot(x, y, color="black", linestyle="-")
    axs[0].axhline(xbar_bar, color="green", linestyle="-")
    axs[0].axhline(UCL_X, color="red", linestyle="-")
    axs[0].axhline(LCL_X, color="red", linestyle="-")
    axs[0].set_title(f"Carta X̄ de {nome_coluna_y}", fontsize=18, fontweight='bold')
    axs[0].set_ylabel("Média Subgrupo", fontsize=16, fontweight='bold')
    axs[0].set_xlabel("Subgrupo", fontsize=16, fontweight='bold')
    axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    xlim = axs[0].get_xlim()
    axs[0].text(xlim[1]+1, xbar_bar, f"X̄̄ = {xbar_bar:.3f}", va='center', fontsize=12, color="green")
    axs[0].text(xlim[1]+1, UCL_X, f"LSC = {UCL_X:.3f}", va='center', fontsize=12, color="red")
    axs[0].text(xlim[1]+1, LCL_X, f"LIC = {LCL_X:.3f}", va='center', fontsize=12, color="red")

    # Critérios 2 e 3
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

    crit2_X, linhas_crit2_X = check_crit2(y, xbar_bar)
    crit3_X, linhas_crit3_X = check_crit3(y)

    crit1_flag_X = []
    for idx, (xi, yi) in enumerate(zip(x, y)):
        cor = "black"
        if yi > UCL_X or yi < LCL_X:
            cor = "red"
            crit1_flag_X.append((idx+1, yi))
        elif crit2_X and (idx+1) in linhas_crit2_X:
            cor = "red"
        elif crit3_X and (idx+1) in linhas_crit3_X:
            cor = "red"
        axs[0].scatter(xi, yi, color=cor)

    # Carta S
    x_s = s.index.values
    y_s = s.values
    axs[1].plot(x_s, y_s, color="black", linestyle="-")
    axs[1].axhline(s_bar, color="green", linestyle="-")
    axs[1].axhline(UCL_S, color="red", linestyle="-")
    axs[1].axhline(LCL_S, color="red", linestyle="-")
    axs[1].set_title("Carta S", fontsize=18, fontweight='bold')
    axs[1].set_ylabel("Desvio Padrão", fontsize=16, fontweight='bold')
    axs[1].set_xlabel("Subgrupo", fontsize=16, fontweight='bold')
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    xlim_s = axs[1].get_xlim()
    axs[1].text(xlim_s[1]+1, s_bar, f"S̄ = {s_bar:.3f}", va='center', fontsize=12, color="green")
    axs[1].text(xlim_s[1]+1, UCL_S, f"LSC = {UCL_S:.3f}", va='center', fontsize=12, color="red")
    axs[1].text(xlim_s[1]+1, LCL_S, f"LIC = {LCL_S:.3f}", va='center', fontsize=12, color="red")

    crit2_S, linhas_crit2_S = check_crit2(y_s, s_bar)
    crit3_S, linhas_crit3_S = check_crit3(y_s)

    crit1_flag_S = []
    for idx, (xi, yi) in enumerate(zip(x_s, y_s)):
        cor = "black"
        if yi > UCL_S or yi < LCL_S:
            cor = "red"
            crit1_flag_S.append((idx+1, yi))
        elif crit2_S and (idx+1) in linhas_crit2_S:
            cor = "red"
        elif crit3_S and (idx+1) in linhas_crit3_S:
            cor = "red"
        axs[1].scatter(xi, yi, color=cor)

    plt.tight_layout()

    # Report – Xbarra
    texto_X = f"📊 **Carta X̄ ({nome_coluna_y})**\n"
    texto_X += "🔎 **Critérios avaliados:**\n"
    if crit1_flag_X:
        pontos = ", ".join([f"Subgrupo {linha}: {valor:.2f}" for linha, valor in crit1_flag_X])
        texto_X += f"1. Critério 1 – Pontos fora dos limites: ❌ Detectado ({pontos})\n"
    else:
        texto_X += "1. Critério 1 – Pontos fora dos limites: ✅ OK\n"

    if crit2_X:
        linhas = ", ".join([str(l) for l in linhas_crit2_X])
        texto_X += f"2. Critério 2 – 9 pontos do mesmo lado da média: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto_X += "2. Critério 2 – 9 pontos do mesmo lado da média: ✅ OK\n"

    if crit3_X:
        linhas = ", ".join([str(l) for l in linhas_crit3_X])
        texto_X += f"3. Critério 3 – 6 pontos subindo ou descendo: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto_X += "3. Critério 3 – 6 pontos subindo ou descendo: ✅ OK\n"

    if crit1_flag_X or crit2_X or crit3_X:
        texto_X += "🔎 **Conclusão:** Causa especial detectada. O processo não está sob controle estatístico.\n"
        texto_X += "🔎 **Recomendação:** Investigue o processo para entender e se possível remover a causa especial identificada.\n"
    else:
        texto_X += "🔎 **Conclusão:** Processo está estável.\n"

    # Report – S
    texto_S = f"📊 **Carta S**\n"
    texto_S += "🔎 **Critérios avaliados:**\n"
    if crit1_flag_S:
        pontos = ", ".join([f"Subgrupo {linha}: {valor:.2f}" for linha, valor in crit1_flag_S])
        texto_S += f"1. Critério 1 – Pontos fora dos limites: ❌ Detectado ({pontos})\n"
    else:
        texto_S += "1. Critério 1 – Pontos fora dos limites: ✅ OK\n"

    if crit2_S:
        linhas = ", ".join([str(l) for l in linhas_crit2_S])
        texto_S += f"2. Critério 2 – 9 pontos do mesmo lado da média: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto_S += "2. Critério 2 – 9 pontos do mesmo lado da média: ✅ OK\n"

    if crit3_S:
        linhas = ", ".join([str(l) for l in linhas_crit3_S])
        texto_S += f"3. Critério 3 – 6 pontos subindo ou descendo: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto_S += "3. Critério 3 – 6 pontos subindo ou descendo: ✅ OK\n"

    if crit1_flag_S or crit2_S or crit3_S:
        texto_S += "🔎 **Conclusão:** Causa especial detectada. O processo não está sob controle estatístico.\n"
        texto_S += "🔎 **Recomendação:** Investigue o processo para entender e se possível remover a causa especial identificada.\n"
    else:
        texto_S += "🔎 **Conclusão:** Processo está estável.\n"

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return (texto_X + "\n" + texto_S), img_base64


def analise_carta_p(df, coluna_y, subgrupo):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from io import BytesIO
    import base64

    aplicar_estilo_minitab()

    nome_coluna_y = coluna_y if isinstance(coluna_y, str) else (coluna_y[0] if coluna_y else None)
    if not nome_coluna_y or nome_coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    if not subgrupo or subgrupo not in df.columns:
        return f"❌ A coluna de tamanho de amostra '{subgrupo}' não foi encontrada no arquivo.", None

    dados = df[[nome_coluna_y, subgrupo]].dropna().copy()
    if dados.empty:
        return "❌ Dados insuficientes para análise.", None

    # Conversão para float
    nc = dados[nome_coluna_y].astype(float)
    n = dados[subgrupo].astype(float)

    # Proporção
    p = nc / n
    p_barra = nc.sum() / n.sum()
    sigma_p = np.sqrt(p_barra * (1 - p_barra) / n)
    LSC = p_barra + 3 * sigma_p
    LIC = p_barra - 3 * sigma_p
    LIC = LIC.clip(lower=0)

    # Gráfico
    fig, ax = plt.subplots(figsize=(12, 6))

    x = p.index.values
    y = p.values
    ax.plot(x, y, color="black", linestyle="-")
    ax.plot(x, LSC, color="red", linestyle="-")
    ax.plot(x, LIC, color="red", linestyle="-")
    ax.axhline(p_barra, color="green", linestyle="-")
    ax.set_title(f"Carta P – Proporção de Defeitos ({nome_coluna_y})", fontsize=18, fontweight='bold')
    ax.set_ylabel("Proporção", fontsize=16, fontweight='bold')
    ax.set_xlabel("Subgrupo", fontsize=16, fontweight='bold')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    xlim = ax.get_xlim()
    ax.text(xlim[1]+1, p_barra, f"p̄ = {p_barra:.4f}", va='center', fontsize=12, color="green")
    ax.text(xlim[1]+1, LSC.iloc[0], f"LSC = {LSC.iloc[0]:.4f}", va='center', fontsize=12, color="red")
    ax.text(xlim[1]+1, LIC.iloc[0], f"LIC = {LIC.iloc[0]:.4f}", va='center', fontsize=12, color="red")

    # Critérios
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

    crit1_flag = []
    for idx, (xi, yi, lsc_i, lic_i) in enumerate(zip(x, y, LSC, LIC)):
        cor = "black"
        if yi > lsc_i or yi < lic_i:
            cor = "red"
            crit1_flag.append((idx+1, yi))
        ax.scatter(xi, yi, color=cor)

    crit2, linhas_crit2 = check_crit2(y, p_barra)
    crit3, linhas_crit3 = check_crit3(y)

    if crit2:
        for i in linhas_crit2:
            ax.scatter(x[i-1], y[i-1], color="red")
    if crit3:
        for i in linhas_crit3:
            ax.scatter(x[i-1], y[i-1], color="red")

    plt.tight_layout()

    # Report
    texto = f"📊 **Carta P (Proporção de Defeitos – {nome_coluna_y})**\n"
    texto += "🔎 **Estatísticas gerais:**\n"
    texto += f"- Proporção média de não conformes (p̄): {p_barra:.4f}\n\n"
    texto += "🔎 **Critérios avaliados:**\n"

    if crit1_flag:
        pontos = ", ".join([f"Subgrupo {linha}: {valor:.4f}" for linha, valor in crit1_flag])
        texto += f"1. Critério 1 – Pontos fora dos limites: ❌ Detectado ({pontos})\n"
    else:
        texto += "1. Critério 1 – Pontos fora dos limites: ✅ OK\n"

    if crit2:
        linhas = ", ".join([str(l) for l in linhas_crit2])
        texto += f"2. Critério 2 – 9 pontos do mesmo lado da média: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto += "2. Critério 2 – 9 pontos do mesmo lado da média: ✅ OK\n"

    if crit3:
        linhas = ", ".join([str(l) for l in linhas_crit3])
        texto += f"3. Critério 3 – 6 pontos subindo ou descendo: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto += "3. Critério 3 – 6 pontos subindo ou descendo: ✅ OK\n"

    if crit1_flag or crit2 or crit3:
        texto += "🔎 **Conclusão:** Causa especial detectada. O processo não está sob controle estatístico.\n"
        texto += "🔎 **Recomendação:** Investigue o processo para entender e se possível remover a causa especial identificada.\n"
    else:
        texto += "🔎 **Conclusão:** Processo está estável.\n"

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return texto, img_base64





def analise_carta_np(df, coluna_y, subgrupo):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from io import BytesIO
    import base64

    aplicar_estilo_minitab()

    nome_coluna_y = coluna_y if isinstance(coluna_y, str) else (coluna_y[0] if coluna_y else None)
    if not nome_coluna_y or nome_coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    if not subgrupo or subgrupo not in df.columns:
        return f"❌ A coluna de tamanho de amostra '{subgrupo}' não foi encontrada no arquivo.", None

    dados = df[[nome_coluna_y, subgrupo]].dropna().copy()
    if dados.empty:
        return "❌ Dados insuficientes para análise.", None

    # Conversão para float
    nc = dados[nome_coluna_y].astype(float)
    n = dados[subgrupo].astype(float)

    # Validação de subgrupo fixo
    if n.nunique() > 1:
        return "❌ A Carta NP requer subgrupos com tamanhos iguais. Verifique os dados.", None

    n_fixo = n.iloc[0]

    # Proporção média
    p_barra = nc.sum() / (n_fixo * len(nc))
    np_barra = n_fixo * p_barra

    # Limites de controle
    sigma_np = np.sqrt(n_fixo * p_barra * (1 - p_barra))
    LSC = np_barra + 3 * sigma_np
    LIC = np_barra - 3 * sigma_np
    LIC = max(0, LIC)  # LIC nunca negativo

    # Gráfico
    fig, ax = plt.subplots(figsize=(12, 6))

    x = nc.index.values
    y = nc.values
    ax.plot(x, y, color="black", linestyle="-")
    ax.axhline(np_barra, color="green", linestyle="-")
    ax.axhline(LSC, color="red", linestyle="-")
    ax.axhline(LIC, color="red", linestyle="-")
    ax.set_title(f"Carta NP – Número de Não Conformes ({nome_coluna_y})", fontsize=18, fontweight='bold')
    ax.set_ylabel("Número de Não Conformes", fontsize=16, fontweight='bold')
    ax.set_xlabel("Subgrupo", fontsize=16, fontweight='bold')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    xlim = ax.get_xlim()
    ax.text(xlim[1]+1, np_barra, f"np̄ = {np_barra:.2f}", va='center', fontsize=12, color="green")
    ax.text(xlim[1]+1, LSC, f"LSC = {LSC:.2f}", va='center', fontsize=12, color="red")
    ax.text(xlim[1]+1, LIC, f"LIC = {LIC:.2f}", va='center', fontsize=12, color="red")

    # Critérios
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

    crit1_flag = []
    for idx, (xi, yi) in enumerate(zip(x, y)):
        cor = "black"
        if yi > LSC or yi < LIC:
            cor = "red"
            crit1_flag.append((idx+1, yi))
        ax.scatter(xi, yi, color=cor)

    crit2, linhas_crit2 = check_crit2(y, np_barra)
    crit3, linhas_crit3 = check_crit3(y)

    if crit2:
        for i in linhas_crit2:
            ax.scatter(x[i-1], y[i-1], color="red")
    if crit3:
        for i in linhas_crit3:
            ax.scatter(x[i-1], y[i-1], color="red")

    plt.tight_layout()

    # Report
    texto = f"📊 **Carta NP (Número de Não Conformes – {nome_coluna_y})**\n"
    texto += "🔎 **Estatísticas gerais:**\n"
    texto += f"- Número médio de não conformes (np̄): {np_barra:.2f}\n\n"
    texto += "🔎 **Critérios avaliados:**\n"

    if crit1_flag:
        pontos = ", ".join([f"Subgrupo {linha}: {valor:.0f}" for linha, valor in crit1_flag])
        texto += f"1. Critério 1 – Pontos fora dos limites: ❌ Detectado ({pontos})\n"
    else:
        texto += "1. Critério 1 – Pontos fora dos limites: ✅ OK\n"

    if crit2:
        linhas = ", ".join([str(l) for l in linhas_crit2])
        texto += f"2. Critério 2 – 9 pontos do mesmo lado da média: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto += "2. Critério 2 – 9 pontos do mesmo lado da média: ✅ OK\n"

    if crit3:
        linhas = ", ".join([str(l) for l in linhas_crit3])
        texto += f"3. Critério 3 – 6 pontos subindo ou descendo: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto += "3. Critério 3 – 6 pontos subindo ou descendo: ✅ OK\n"

    if crit1_flag or crit2 or crit3:
        texto += "🔎 **Conclusão:** Causa especial detectada. O processo não está sob controle estatístico.\n"
        texto += "🔎 **Recomendação:** Investigue o processo para entender e se possível remover a causa especial identificada.\n"
    else:
        texto += "🔎 **Conclusão:** Processo está estável.\n"

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return texto, img_base64



def analise_carta_c(df, coluna_y):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from io import BytesIO
    import base64

    aplicar_estilo_minitab()

    nome_coluna_y = coluna_y if isinstance(coluna_y, str) else (coluna_y[0] if coluna_y else None)
    if not nome_coluna_y or nome_coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    dados = df[[nome_coluna_y]].dropna().copy()
    if dados.empty:
        return "❌ Dados insuficientes para análise.", None

    # Conversão para float
    c = dados[nome_coluna_y].astype(float)
    c_barra = c.mean()

    # Limites de controle
    sigma_c = np.sqrt(c_barra)
    LSC = c_barra + 3 * sigma_c
    LIC = c_barra - 3 * sigma_c
    LIC = max(0, LIC)  # LIC nunca negativo

    # Gráfico
    fig, ax = plt.subplots(figsize=(12, 6))

    x = c.index.values
    y = c.values
    ax.plot(x, y, color="black", linestyle="-")
    ax.axhline(c_barra, color="green", linestyle="-")
    ax.axhline(LSC, color="red", linestyle="-")
    ax.axhline(LIC, color="red", linestyle="-")
    ax.set_title(f"Carta C – Contagem de Não Conformes ({nome_coluna_y})", fontsize=18, fontweight='bold')
    ax.set_ylabel("Contagem de Não Conformes", fontsize=16, fontweight='bold')
    ax.set_xlabel("Subgrupo", fontsize=16, fontweight='bold')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    xlim = ax.get_xlim()
    ax.text(xlim[1]+1, c_barra, f"c̄ = {c_barra:.2f}", va='center', fontsize=12, color="green")
    ax.text(xlim[1]+1, LSC, f"LSC = {LSC:.2f}", va='center', fontsize=12, color="red")
    ax.text(xlim[1]+1, LIC, f"LIC = {LIC:.2f}", va='center', fontsize=12, color="red")

    # Critérios
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

    crit1_flag = []
    for idx, (xi, yi) in enumerate(zip(x, y)):
        cor = "black"
        if yi > LSC or yi < LIC:
            cor = "red"
            crit1_flag.append((idx+1, yi))
        ax.scatter(xi, yi, color=cor)

    crit2, linhas_crit2 = check_crit2(y, c_barra)
    crit3, linhas_crit3 = check_crit3(y)

    if crit2:
        for i in linhas_crit2:
            ax.scatter(x[i-1], y[i-1], color="red")
    if crit3:
        for i in linhas_crit3:
            ax.scatter(x[i-1], y[i-1], color="red")

    plt.tight_layout()

    # Report
    texto = f"📊 **Carta C (Contagem de Não Conformes – {nome_coluna_y})**\n"
    texto += "🔎 **Estatísticas gerais:**\n"
    texto += f"- Contagem média de não conformes (c̄): {c_barra:.2f}\n\n"
    texto += "🔎 **Critérios avaliados:**\n"

    if crit1_flag:
        pontos = ", ".join([f"Subgrupo {linha}: {valor:.0f}" for linha, valor in crit1_flag])
        texto += f"1. Critério 1 – Pontos fora dos limites: ❌ Detectado ({pontos})\n"
    else:
        texto += "1. Critério 1 – Pontos fora dos limites: ✅ OK\n"

    if crit2:
        linhas = ", ".join([str(l) for l in linhas_crit2])
        texto += f"2. Critério 2 – 9 pontos do mesmo lado da média: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto += "2. Critério 2 – 9 pontos do mesmo lado da média: ✅ OK\n"

    if crit3:
        linhas = ", ".join([str(l) for l in linhas_crit3])
        texto += f"3. Critério 3 – 6 pontos subindo ou descendo: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto += "3. Critério 3 – 6 pontos subindo ou descendo: ✅ OK\n"

    if crit1_flag or crit2 or crit3:
        texto += "🔎 **Conclusão:** Causa especial detectada. O processo não está sob controle estatístico.\n"
        texto += "🔎 **Recomendação:** Investigue o processo para entender e se possível remover a causa especial identificada.\n"
    else:
        texto += "🔎 **Conclusão:** Processo está estável.\n"

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return texto, img_base64


def analise_carta_u(df, coluna_y, subgrupo):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from io import BytesIO
    import base64

    aplicar_estilo_minitab()

    nome_coluna_y = coluna_y if isinstance(coluna_y, str) else (coluna_y[0] if coluna_y else None)
    if not nome_coluna_y or nome_coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    if not subgrupo or subgrupo not in df.columns:
        return f"❌ A coluna de tamanho de amostra '{subgrupo}' não foi encontrada no arquivo.", None

    dados = df[[nome_coluna_y, subgrupo]].dropna().copy()
    if dados.empty:
        return "❌ Dados insuficientes para análise.", None

    # Conversão para float
    c = dados[nome_coluna_y].astype(float)
    n = dados[subgrupo].astype(float)

    # Taxa de não conformes por unidade
    u = c / n
    u_barra = c.sum() / n.sum()

    # Limites de controle
    sigma_u = np.sqrt(u_barra / n)
    LSC = u_barra + 3 * sigma_u
    LIC = u_barra - 3 * sigma_u
    LIC = LIC.clip(lower=0)

    # Gráfico
    fig, ax = plt.subplots(figsize=(12, 6))

    x = u.index.values
    y = u.values
    ax.plot(x, y, color="black", linestyle="-")
    ax.axhline(u_barra, color="green", linestyle="-")
    ax.plot(x, LSC, color="red", linestyle="-")
    ax.plot(x, LIC, color="red", linestyle="-")
    ax.set_title(f"Carta U – Taxa de Não Conformes por Unidade ({nome_coluna_y})", fontsize=18, fontweight='bold')
    ax.set_ylabel("Taxa de Não Conformes", fontsize=16, fontweight='bold')
    ax.set_xlabel("Subgrupo", fontsize=16, fontweight='bold')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    xlim = ax.get_xlim()
    ax.text(xlim[1]+1, u_barra, f"ū = {u_barra:.4f}", va='center', fontsize=12, color="green")
    ax.text(xlim[1]+1, LSC.iloc[0], f"LSC = {LSC.iloc[0]:.4f}", va='center', fontsize=12, color="red")
    ax.text(xlim[1]+1, LIC.iloc[0], f"LIC = {LIC.iloc[0]:.4f}", va='center', fontsize=12, color="red")

    # Critérios
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

    crit1_flag = []
    for idx, (xi, yi, lsc_i, lic_i) in enumerate(zip(x, y, LSC, LIC)):
        cor = "black"
        if yi > lsc_i or yi < lic_i:
            cor = "red"
            crit1_flag.append((idx+1, yi))
        ax.scatter(xi, yi, color=cor)

    crit2, linhas_crit2 = check_crit2(y, u_barra)
    crit3, linhas_crit3 = check_crit3(y)

    if crit2:
        for i in linhas_crit2:
            ax.scatter(x[i-1], y[i-1], color="red")
    if crit3:
        for i in linhas_crit3:
            ax.scatter(x[i-1], y[i-1], color="red")

    plt.tight_layout()

    # Report
    texto = f"📊 **Carta U (Taxa de Não Conformes por Unidade – {nome_coluna_y})**\n"
    texto += "🔎 **Estatísticas gerais:**\n"
    texto += f"- Taxa média de não conformes por unidade (ū): {u_barra:.4f}\n\n"
    texto += "🔎 **Critérios avaliados:**\n"

    if crit1_flag:
        pontos = ", ".join([f"Subgrupo {linha}: {valor:.4f}" for linha, valor in crit1_flag])
        texto += f"1. Critério 1 – Pontos fora dos limites: ❌ Detectado ({pontos})\n"
    else:
        texto += "1. Critério 1 – Pontos fora dos limites: ✅ OK\n"

    if crit2:
        linhas = ", ".join([str(l) for l in linhas_crit2])
        texto += f"2. Critério 2 – 9 pontos do mesmo lado da média: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto += "2. Critério 2 – 9 pontos do mesmo lado da média: ✅ OK\n"

    if crit3:
        linhas = ", ".join([str(l) for l in linhas_crit3])
        texto += f"3. Critério 3 – 6 pontos subindo ou descendo: ❌ Detectado (Subgrupos {linhas})\n"
    else:
        texto += "3. Critério 3 – 6 pontos subindo ou descendo: ✅ OK\n"

    if crit1_flag or crit2 or crit3:
        texto += "🔎 **Conclusão:** Causa especial detectada. O processo não está sob controle estatístico.\n"
        texto += "🔎 **Recomendação:** Investigue o processo para entender e se possível remover a causa especial identificada.\n"
    else:
        texto += "🔎 **Conclusão:** Processo está estável.\n"

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return texto, img_base64


def analise_carta_ewma(df, coluna_y, lambda_val=0.2):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from io import BytesIO
    import base64

    aplicar_estilo_minitab()

    # Validação coluna
    nome_coluna_y = coluna_y if isinstance(coluna_y, str) else (coluna_y[0] if coluna_y else None)
    if not nome_coluna_y or nome_coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    dados = df[[nome_coluna_y]].dropna().copy()
    if dados.empty:
        return "❌ Dados insuficientes para análise.", None

    # Parâmetros
    x = dados.index.values
    y = dados[nome_coluna_y].astype(float).values
    mu = np.mean(y)
    sigma = np.std(y, ddof=1)
    L = 3

    # Calcular EWMA
    ewma = []
    ewma.append(mu)  # inicia com média do processo
    for i in range(1, len(y)):
        ewma.append(lambda_val * y[i] + (1 - lambda_val) * ewma[i-1])
    ewma = np.array(ewma)

    # Calcular limites de controle
    t = np.arange(1, len(y)+1)
    sigma_ewma = sigma * np.sqrt((lambda_val / (2 - lambda_val)) * (1 - (1 - lambda_val) ** (2 * t)))
    UCL = mu + L * sigma_ewma
    LCL = mu - L * sigma_ewma

    # Gráfico
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(x, y, color="lightgrey", linestyle="--", label="Dados Originais")
    ax.plot(x, ewma, color="black", linestyle="-", label="EWMA")
    ax.plot(x, UCL, color="red", linestyle="-", label="LSC")
    ax.plot(x, LCL, color="red", linestyle="-", label="LIC")
    ax.axhline(mu, color="green", linestyle="-", label="Média")

    ax.set_title(f"Carta EWMA ({nome_coluna_y})", fontsize=18, fontweight='bold')
    ax.set_ylabel("Valor EWMA", fontsize=16, fontweight='bold')
    ax.set_xlabel("Observação", fontsize=16, fontweight='bold')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend()

    xlim = ax.get_xlim()
    ax.text(xlim[1]+1, mu, f"μ = {mu:.3f}", va='center', fontsize=12, color="green")
    ax.text(xlim[1]+1, UCL[-1], f"LSC = {UCL[-1]:.3f}", va='center', fontsize=12, color="red")
    ax.text(xlim[1]+1, LCL[-1], f"LIC = {LCL[-1]:.3f}", va='center', fontsize=12, color="red")

    # Critério 1 – Pontos EWMA fora dos limites
    crit1_flag = []
    for idx, val in enumerate(ewma):
        cor = "black"
        if val > UCL[idx] or val < LCL[idx]:
            cor = "red"
            crit1_flag.append((idx+1, val))
        ax.scatter(x[idx], val, color=cor)

    plt.tight_layout()

    # Report
    texto = f"📊 **Carta EWMA ({nome_coluna_y})**\n"
    texto += f"🔎 **Parâmetros:** λ = {lambda_val}, Média = {mu:.3f}, Sigma = {sigma:.3f}\n\n"
    texto += "🔎 **Critérios avaliados:**\n"

    if crit1_flag:
        pontos = ", ".join([f"{linha}: {valor:.3f}" for linha, valor in crit1_flag])
        texto += f"1. Critério 1 – Pontos fora dos limites: ❌ Detectado (Observações {pontos})\n"
    else:
        texto += "1. Critério 1 – Pontos fora dos limites: ✅ OK\n"

    # Conclusão
    if crit1_flag:
        texto += "🔎 **Conclusão:** Causa especial detectada. O processo não está sob controle estatístico.\n"
        texto += "🔎 **Recomendação:** Investigue o processo para entender e se possível remover a causa especial identificada.\n"
    else:
        texto += "🔎 **Conclusão:** Processo está estável.\n"

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return texto, img_base64



ANALISES = {
    "Carta I-MR": analise_carta_imr,
    "Carta X-BarraR": analise_carta_xbarra_r, 
    "Carta X-BarraS": analise_carta_xbarra_s, 
    "Carta P": analise_carta_p,
    "Carta NP": analise_carta_np,
    "Carta C": analise_carta_c,
    "Carta U": analise_carta_u,
    "Carta EWMA": analise_carta_ewma,
    

    
}


