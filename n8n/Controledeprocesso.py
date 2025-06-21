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

