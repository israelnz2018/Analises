import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
import os
import io
from io import BytesIO
from suporte import aplicar_estilo_minitab


def salvar_grafico():
    caminho = "grafico.png"
    plt.tight_layout()
    plt.savefig(caminho)
    plt.close()
    with open(caminho, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(caminho)
    return img_base64 


def gerar_histograma(df, coluna_y, subgrupo=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    aplicar_estilo_minitab()

    info_grafico = {
        "titulo_grafico": "Histograma com Curva de Densidade",
        "tamanho_fonte": 12,
        "titulo_x": coluna_y,
        "titulo_y": "Frequência",
        "inclinacao_x": 0,
        "cor": "steelblue",
        "lista_y": [coluna_y] if coluna_y else []
    }

    if isinstance(coluna_y, list):
        coluna_y = coluna_y[0] if coluna_y else None

    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None, None

    try:
        if subgrupo:
            if subgrupo not in df.columns:
                return "❌ A coluna Subgrupo informada não foi encontrada no arquivo.", None, None

            dados = df[[coluna_y, subgrupo]].dropna()
            if dados.empty:
                return "❌ Dados insuficientes para gerar o gráfico.", None, None

            subgrupos = dados[subgrupo].unique()
            if len(subgrupos) != 2:
                return f"❌ O gráfico espera exatamente 2 subgrupos e encontrou {len(subgrupos)}.", None, None

            fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
            for ax, sub in zip(axs, subgrupos):
                sns.histplot(
                    dados[dados[subgrupo] == sub][coluna_y],
                    bins=10,
                    kde=True,
                    stat="count",
                    alpha=0.4,
                    color="steelblue",
                    edgecolor="black",
                    ax=ax
                )
                ax.set_title(f"Histograma - {sub}", fontsize=12)
                ax.set_xlabel(coluna_y, fontsize=12)
                ax.set_ylabel("Frequência", fontsize=12)

            plt.tight_layout()

        else:
            plt.figure(figsize=(10, 6))
            sns.histplot(
                df[coluna_y].dropna(),
                bins=10,
                kde=True,
                stat="count",
                alpha=0.4,
                color="steelblue",
                edgecolor="black"
            )
            plt.xlabel(coluna_y, fontsize=12)
            plt.ylabel("Frequência", fontsize=12)
            plt.title("Histograma com Curva de Densidade", fontsize=12)
            plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return "", imagem_base64, info_grafico

    except Exception as e:
        return f"❌ Erro ao gerar o histograma: {str(e)}", None, None


def personalizar_histograma(df, coluna_y, subgrupo=None, cor="#000000", titulo_x="", titulo_y="", titulo_grafico="", tamanho_fonte=12, inclinacao_x=0):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    aplicar_estilo_minitab()

    if isinstance(coluna_y, list):
        coluna_y = coluna_y[0] if coluna_y else None

    if not coluna_y or coluna_y not in df.columns:
        return None, None

    dados = df[[coluna_y, subgrupo]].dropna() if subgrupo and subgrupo in df.columns else df[[coluna_y]].dropna()
    if dados.empty:
        return None, None

    if subgrupo and subgrupo in df.columns:
        subgrupos = dados[subgrupo].unique()
        fig, axs = plt.subplots(1, len(subgrupos), figsize=(8 * len(subgrupos), 5), sharey=True)
        if len(subgrupos) == 1:
            axs = [axs]

        for ax, sub in zip(axs, subgrupos):
            sns.histplot(
                dados[dados[subgrupo] == sub][coluna_y],
                bins=10,
                kde=True,
                stat="count",
                alpha=0.4,
                color=cor,
                edgecolor="black",
                ax=ax
            )

            # 🔧 Ajustado: título exibe o nome real do subgrupo
            ax.set_title(f"{sub}", fontsize=int(tamanho_fonte))
            ax.set_xlabel(titulo_x if titulo_x else coluna_y, fontsize=int(tamanho_fonte))
            ax.set_ylabel(titulo_y if titulo_y else "Frequência", fontsize=int(tamanho_fonte))
            ax.tick_params(axis='x', rotation=int(inclinacao_x))

        titulo_padrao = f"Histograma por {' e '.join(subgrupos)}"

    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            dados[coluna_y],
            bins=10,
            kde=True,
            stat="count",
            alpha=0.4,
            color=cor,
            edgecolor="black",
            ax=ax
        )

        ax.set_title(titulo_grafico if titulo_grafico else f"Histograma de {coluna_y}", fontsize=int(tamanho_fonte))
        ax.set_xlabel(titulo_x if titulo_x else coluna_y, fontsize=int(tamanho_fonte))
        ax.set_ylabel(titulo_y if titulo_y else "Frequência", fontsize=int(tamanho_fonte))
        ax.tick_params(axis='x', rotation=int(inclinacao_x))

        titulo_padrao = f"Histograma de {coluna_y}"

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    info_grafico = {
        "cor": cor or "",
        "titulo_grafico": titulo_grafico.strip() if titulo_grafico.strip() else titulo_padrao,
        "titulo_x": titulo_x,
        "titulo_y": titulo_y,
        "tamanho_fonte": tamanho_fonte or "",
        "inclinacao_x": inclinacao_x or "",
        "inclinacao_y": "",
        "espessura": "",
        "lista_y": [coluna_y]
    }

    info_grafico["subgrupo"] = subgrupo if subgrupo else ""

    return imagem_base64, info_grafico



def gerar_pareto(df, coluna_x, coluna_y=None, subgrupo=None):
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab
    from matplotlib.colors import to_hex

    aplicar_estilo_minitab()

    if not coluna_x or coluna_x not in df.columns:
        return "❌ A coluna X informada não foi encontrada no arquivo.", None, None
    if coluna_y and coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None, None
    if subgrupo and subgrupo not in df.columns:
        return "❌ A coluna Subgrupo informada não foi encontrada no arquivo.", None, None

    info_grafico = {
        "titulo_grafico": f"Pareto - {coluna_x}",
        "tamanho_fonte": 12,
        "titulo_x": coluna_x,
        "titulo_y": "Frequência",
        "inclinacao_x": 0,
        "cor": "",  # ✅ atualizado abaixo após plot
        "lista_y": [coluna_y] if coluna_y else []
    }

    try:
        def plotar(dados, titulo, ax):
            if coluna_y:
                contagem = dados.groupby(coluna_x)[coluna_y].sum().sort_values(ascending=False)
            else:
                contagem = dados[coluna_x].value_counts().sort_values(ascending=False)

            if contagem.empty:
                ax.axis('off')
                ax.set_title(f"{titulo} - Sem dados")
                return None

            acumulado = contagem.cumsum() / contagem.sum() * 100
            bars = contagem.plot(kind="bar", ax=ax)
            real_color = bars.patches[0].get_facecolor() if bars.patches else None

            ax.set_ylabel("Frequência")
            ax.set_title(titulo)

            ax2 = ax.twinx()
            ax2.plot(acumulado.index, acumulado.values, color="red", marker="o")
            ax2.set_ylabel("Acumulado (%)")
            ax2.set_ylim(0, 110)

            for i, (x, y) in enumerate(zip(contagem.index, acumulado)):
                ax2.text(i, y + 2, f"{y:.1f}%", color="red", ha="center", fontsize=8)

            return to_hex(real_color) if real_color else ""

        if subgrupo:
            colunas = [coluna_x, subgrupo] + ([coluna_y] if coluna_y else [])
            dados = df[colunas].dropna()
            subgrupos = dados[subgrupo].dropna().unique()

            if dados.empty:
                return "❌ Dados insuficientes para gerar o gráfico.", None, None
            if len(subgrupos) != 2:
                return f"❌ O gráfico espera exatamente 2 categorias no subgrupo e encontrou {len(subgrupos)}.", None, None

            fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

            for ax, sub in zip(axs, subgrupos):
                dados_sub = dados[dados[subgrupo] == sub]
                cor_usada = plotar(dados_sub, f"Pareto - {coluna_x} ({sub})", ax)

        else:
            dados = df[[coluna_x, coluna_y]].dropna() if coluna_y else df[[coluna_x]].dropna()
            if dados.empty:
                return "❌ Dados insuficientes para gerar o gráfico.", None, None

            fig, ax = plt.subplots(figsize=(10, 5))
            cor_usada = plotar(dados, f"Pareto - {coluna_x}", ax)

        plt.tight_layout()

        # ✅ Atualiza info_grafico com cor convertida para string hex
        info_grafico["cor"] = cor_usada

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return "", imagem_base64, info_grafico

    except Exception as e:
        return f"❌ Erro ao gerar o gráfico de Pareto: {str(e)}", None, None




def personalizar_pareto(df, coluna_x, coluna_y=None, subgrupo=None, cor=None, titulo_x="", titulo_y="", titulo_grafico="", tamanho_fonte=12, inclinacao_x=0):
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    aplicar_estilo_minitab()

    if not coluna_x or coluna_x not in df.columns:
        return None, None
    if coluna_y and coluna_y not in df.columns:
        coluna_y = None
    if subgrupo and subgrupo not in df.columns:
        subgrupo = None

    colunas_necessarias = [coluna_x] + ([coluna_y] if coluna_y else []) + ([subgrupo] if subgrupo else [])
    dados = df.dropna(subset=colunas_necessarias)
    if dados.empty:
        return None, None

    def plotar(ax, dados, titulo):
        if coluna_y:
            contagem = dados.groupby(coluna_x)[coluna_y].sum().sort_values(ascending=False)
        else:
            contagem = dados[coluna_x].value_counts().sort_values(ascending=False)

        if contagem.empty:
            ax.axis('off')
            ax.set_title(f"{titulo} - Sem dados")
            return

        acumulado = contagem.cumsum() / contagem.sum() * 100

        contagem.plot(kind="bar", ax=ax, color=cor if cor else None, edgecolor="black")
        
        ax.set_xlabel(titulo_x.strip() if titulo_x.strip() != "" else coluna_x, fontsize=int(tamanho_fonte))
        ax.set_ylabel(titulo_y.strip() if titulo_y.strip() != "" else "Frequência / Soma", fontsize=int(tamanho_fonte))
        ax.set_title(titulo, fontsize=int(tamanho_fonte))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=int(inclinacao_x))

        ax2 = ax.twinx()
        ax2.plot(acumulado.index, acumulado.values, color="red", marker="o")
        ax2.set_ylabel("Acumulado (%)", fontsize=int(tamanho_fonte))
        ax2.set_ylim(0, 110)

        for i, (x, y) in enumerate(zip(contagem.index, acumulado)):
            ax2.text(i, y + 2, f"{y:.1f}%", color="red", ha="center", fontsize=8)

    if subgrupo:
        subgrupos = dados[subgrupo].unique()
        fig, axs = plt.subplots(1, len(subgrupos), figsize=(8 * len(subgrupos), 6), sharey=True)

        if len(subgrupos) == 1:
            axs = [axs]

        for ax, sub in zip(axs, subgrupos):
            dados_sub = dados[dados[subgrupo] == sub]
            plotar(ax, dados_sub, f"{sub}")

        titulo_padrao = f"Pareto por {' e '.join(subgrupos)}"

    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        titulo = titulo_grafico.strip() if titulo_grafico.strip() != "" else f"Pareto - {coluna_x}"
        plotar(ax, dados, titulo)
        titulo_padrao = titulo

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    info_grafico = {
        "cor": cor if cor else "",
        "titulo_grafico": titulo_padrao,
        "titulo_x": titulo_x.strip() if titulo_x.strip() != "" else coluna_x,
        "titulo_y": titulo_y.strip() if titulo_y.strip() != "" else "Frequência",
        "tamanho_fonte": tamanho_fonte or "",
        "inclinacao_x": inclinacao_x or "",
        "inclinacao_y": "",
        "espessura": "",
        "lista_y": [coluna_y] if coluna_y else []
    }

    info_grafico["subgrupo"] = subgrupo if subgrupo else ""

    return imagem_base64, info_grafico








def gerar_pizza(df, coluna_x, coluna_y=None, subgrupo=None):
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    if not coluna_x or coluna_x not in df.columns:
        return "❌ A coluna X informada não foi encontrada no arquivo.", None, None

    if coluna_y and coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None, None

    if subgrupo and subgrupo not in df.columns:
        return "❌ A coluna Subgrupo informada não foi encontrada no arquivo.", None, None

    aplicar_estilo_minitab()

    info_grafico = {
        "titulo_grafico": f"Pizza de {coluna_x}",
        "tamanho_fonte": 12,
        "titulo_x": "",
        "titulo_y": "",
        "inclinacao_x": 0,
        "cor": "",
        "lista_y": [coluna_y] if coluna_y else []
    }

    try:
        if subgrupo:
            dados = df[[coluna_x, subgrupo] + ([coluna_y] if coluna_y else [])].dropna()
            subgrupos = dados[subgrupo].dropna().unique()

            if len(subgrupos) != 2:
                return f"❌ O gráfico espera exatamente 2 subgrupos e encontrou {len(subgrupos)}.", None, None

            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            for ax, sub in zip(axs, subgrupos):
                dados_sub = dados[dados[subgrupo] == sub]
                if coluna_y:
                    soma = dados_sub.groupby(coluna_x)[coluna_y].sum()
                else:
                    soma = dados_sub[coluna_x].value_counts()

                if soma.empty or soma.sum() == 0:
                    return f"❌ Dados insuficientes para gerar o gráfico para o subgrupo {sub}.", None, None

                soma.plot.pie(autopct='%1.1f%%', startangle=90, legend=False, ax=ax)
                ax.set_ylabel("")
                ax.set_title(f"Pizza de {coluna_x} ({sub})")

            plt.tight_layout()

        else:
            dados = df[[coluna_x] + ([coluna_y] if coluna_y else [])].dropna()
            if dados.empty:
                return "❌ Dados insuficientes para gerar o gráfico.", None, None

            if coluna_y:
                soma = dados.groupby(coluna_x)[coluna_y].sum()
            else:
                soma = dados[coluna_x].value_counts()

            if soma.empty or soma.sum() == 0:
                return "❌ Dados insuficientes para gerar o gráfico.", None, None

            plt.figure(figsize=(8, 6))
            soma.plot.pie(autopct='%1.1f%%', startangle=90, legend=False)
            plt.ylabel("")
            plt.title(f"Pizza de {coluna_x}")
            plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return "", imagem_base64, info_grafico

    except Exception as e:
        return f"❌ Erro ao gerar o gráfico: {str(e)}", None, None


def personalizar_pizza(df, coluna_x, coluna_y=None, subgrupo=None, titulo_grafico="", tamanho_fonte=12):
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    aplicar_estilo_minitab()

    if not coluna_x or coluna_x not in df.columns:
        return None, None

    if coluna_y and coluna_y not in df.columns:
        coluna_y = None

    if subgrupo and subgrupo not in df.columns:
        subgrupo = None

    dados = df.dropna(subset=[coluna_x] + ([coluna_y] if coluna_y else []) + ([subgrupo] if subgrupo else []))
    if dados.empty:
        return None, None

    if subgrupo:
        subgrupos = dados[subgrupo].dropna().unique()
        fig, axs = plt.subplots(1, len(subgrupos), figsize=(8 * len(subgrupos), 6))

        if len(subgrupos) == 1:
            axs = [axs]

        for ax, sub in zip(axs, subgrupos):
            dados_sub = dados[dados[subgrupo] == sub]

            if coluna_y:
                soma = dados_sub.groupby(coluna_x)[coluna_y].sum()
            else:
                soma = dados_sub[coluna_x].value_counts()

            if soma.empty or soma.sum() == 0:
                ax.axis('off')
                ax.set_title(f"{sub} (Sem dados)", fontsize=int(tamanho_fonte))
                continue

            soma.plot.pie(autopct='%1.1f%%', startangle=90, legend=False, ax=ax)
            ax.set_ylabel("")
            ax.set_title(str(sub), fontsize=int(tamanho_fonte))

            for text in ax.texts:
                text.set_fontsize(int(tamanho_fonte))

        titulo_padrao = f"Pizza por {' e '.join([str(s) for s in subgrupos])}"

    else:
        if coluna_y:
            soma = dados.groupby(coluna_x)[coluna_y].sum()
        else:
            soma = dados[coluna_x].value_counts()

        if soma.empty or soma.sum() == 0:
            return None, None

        fig, ax = plt.subplots(figsize=(8, 6))
        soma.plot.pie(autopct='%1.1f%%', startangle=90, legend=False, ax=ax)
        ax.set_ylabel("")

        titulo_padrao = titulo_grafico.strip() if titulo_grafico.strip() != "" else f"Pizza de {coluna_x}"
        ax.set_title(titulo_padrao, fontsize=int(tamanho_fonte))

        for text in ax.texts:
            text.set_fontsize(int(tamanho_fonte))

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    info_grafico = {
        "cor": "",  # Pizza não usa cor única
        "titulo_grafico": titulo_padrao,
        "titulo_x": "",
        "titulo_y": "",
        "tamanho_fonte": tamanho_fonte or "",
        "inclinacao_x": "",
        "inclinacao_y": "",
        "espessura": "",
        "lista_y": [coluna_y] if coluna_y else []
    }
    info_grafico["subgrupo"] = subgrupo if subgrupo else ""

    return imagem_base64, info_grafico





def gerar_barras(df, coluna_x, coluna_y=None, subgrupo=None):
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    aplicar_estilo_minitab()

    info_grafico = {
        "titulo_grafico": f"Barras de {coluna_x}",
        "tamanho_fonte": 12,
        "titulo_x": coluna_x,
        "titulo_y": "Frequência",
        "inclinacao_x": 90,
        "cor": "",  # ✅ inicializa como string vazia
        "lista_y": [coluna_y] if coluna_y else []
    }

    if not coluna_x or coluna_x not in df.columns:
        return "❌ A coluna X informada não foi encontrada no arquivo.", None, None

    if coluna_y and coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None, None

    if subgrupo and subgrupo not in df.columns:
        return "❌ A coluna Subgrupo informada não foi encontrada no arquivo.", None, None

    try:
        def converter_cor_rgb_para_hex(cor_rgb):
            if isinstance(cor_rgb, tuple):
                return '#%02x%02x%02x' % tuple(int(255*x) for x in cor_rgb[:3])
            return cor_rgb

        cor_usada_final = ""

        if subgrupo:
            dados = df[[coluna_x, coluna_y, subgrupo]].dropna() if coluna_y else df[[coluna_x, subgrupo]].dropna()
            subgrupos = dados[subgrupo].unique()

            if len(subgrupos) != 2:
                return f"❌ O gráfico espera exatamente 2 subgrupos e encontrou {len(subgrupos)}.", None, None

            categorias = sorted(dados[coluna_x].dropna().unique())

            fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

            for ax, sub in zip(axs, subgrupos):
                dados_sub = dados[dados[subgrupo] == sub]

                if coluna_y:
                    contagem = dados_sub.groupby(coluna_x)[coluna_y].sum().reindex(categorias, fill_value=0)
                else:
                    contagem = dados_sub[coluna_x].value_counts().reindex(categorias, fill_value=0)

                bars = contagem.plot(kind="bar", ax=ax)
                if bars.patches:
                    cor_usada_final = converter_cor_rgb_para_hex(bars.patches[0].get_facecolor())

                ax.set_ylabel("Frequência")
                ax.set_title(f"Barras de {coluna_x} ({sub})")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

            plt.tight_layout()

        else:
            dados = df[[coluna_x, coluna_y]].dropna() if coluna_y else df[[coluna_x]].dropna()

            if coluna_y:
                contagem = dados.groupby(coluna_x)[coluna_y].sum()
            else:
                contagem = dados[coluna_x].value_counts()

            plt.figure(figsize=(10,6))
            bars = contagem.plot(kind="bar")
            if bars.patches:
                cor_usada_final = converter_cor_rgb_para_hex(bars.patches[0].get_facecolor())

            plt.ylabel("Frequência")
            plt.title(f"Barras de {coluna_x}")
            plt.xticks(rotation=90)
            plt.tight_layout()

        info_grafico["cor"] = cor_usada_final

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return "", imagem_base64, info_grafico

    except Exception as e:
        return f"❌ Erro ao gerar o gráfico de barras: {str(e)}", None, None



def personalizar_barras(df, coluna_x, coluna_y=None, subgrupo=None, cor=None, titulo_x="", titulo_y="", titulo_grafico="", tamanho_fonte=12, inclinacao_x=0):
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    aplicar_estilo_minitab()

    if not coluna_x or coluna_x not in df.columns:
        return None, None
    if coluna_y and coluna_y not in df.columns:
        coluna_y = None
    if subgrupo and subgrupo not in df.columns:
        subgrupo = None

    cor_usada_final = ""

    colunas_necessarias = [coluna_x] + ([coluna_y] if coluna_y else []) + ([subgrupo] if subgrupo else [])
    dados = df.dropna(subset=colunas_necessarias)
    if dados.empty:
        return None, None

    def plotar(ax, contagem, titulo):
        nonlocal cor_usada_final
        bars = contagem.plot(kind="bar", color=cor if cor else None, edgecolor="black", ax=ax)
        if bars.patches:
            cor_usada_final = bars.patches[0].get_facecolor()

        ax.set_xlabel(titulo_x.strip() if titulo_x.strip() != "" else coluna_x, fontsize=int(tamanho_fonte))
        ax.set_ylabel(titulo_y.strip() if titulo_y.strip() != "" else ("Soma de Y" if coluna_y else "Frequência"), fontsize=int(tamanho_fonte))
        ax.set_title(titulo, fontsize=int(tamanho_fonte))
        ax.tick_params(axis='x', rotation=int(inclinacao_x), labelsize=int(tamanho_fonte))
        ax.tick_params(axis='y', labelsize=int(tamanho_fonte))

    if subgrupo:
        subgrupos = dados[subgrupo].dropna().unique()
        fig, axs = plt.subplots(1, len(subgrupos), figsize=(8 * len(subgrupos), 6), sharey=True)

        if len(subgrupos) == 1:
            axs = [axs]

        for ax, sub in zip(axs, subgrupos):
            dados_sub = dados[dados[subgrupo] == sub]
            if coluna_y:
                contagem = dados_sub.groupby(coluna_x)[coluna_y].sum()
            else:
                contagem = dados_sub[coluna_x].value_counts()

            if contagem.empty:
                ax.axis('off')
                ax.set_title(f"{sub} (Sem dados)", fontsize=int(tamanho_fonte))
                continue

            plotar(ax, contagem, str(sub))

        titulo_padrao = f"Barras por {' e '.join([str(s) for s in subgrupos])}"

    else:
        if coluna_y:
            contagem = dados.groupby(coluna_x)[coluna_y].sum()
        else:
            contagem = dados[coluna_x].value_counts()

        if contagem.empty:
            return None, None

        fig, ax = plt.subplots(figsize=(10, 6))
        titulo = titulo_grafico.strip() if titulo_grafico.strip() != "" else f"Barras de {coluna_x}"
        plotar(ax, contagem, titulo)
        titulo_padrao = titulo

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    info_grafico = {
        "cor": cor if cor else cor_usada_final,
        "titulo_grafico": titulo_padrao,
        "titulo_x": titulo_x.strip() if titulo_x.strip() != "" else coluna_x,
        "titulo_y": titulo_y.strip() if titulo_y.strip() != "" else ("Soma de Y" if coluna_y else "Frequência"),
        "tamanho_fonte": tamanho_fonte or "",
        "inclinacao_x": inclinacao_x or "",
        "inclinacao_y": "",
        "espessura": "",
        "lista_y": [coluna_y] if coluna_y else [],
        "subgrupo": subgrupo if subgrupo else ""
    }

    return imagem_base64, info_grafico








def gerar_boxplot(df, lista_y, subgrupo=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    aplicar_estilo_minitab()

    info_grafico = {
        "titulo_grafico": "",
        "tamanho_fonte": 12,
        "titulo_x": "",
        "titulo_y": "",
        "inclinacao_x": 0,
        "cor": "steelblue"
    }

    if not lista_y or any(y not in df.columns for y in lista_y):
        return "❌ Uma ou mais colunas Y informadas não foram encontradas no arquivo.", None, None

    if subgrupo and subgrupo not in df.columns:
        return "❌ A coluna Subgrupo informada não foi encontrada no arquivo.", None, None

    if subgrupo:
        dados = df[lista_y + [subgrupo]].dropna()
        subgrupos = dados[subgrupo].unique()

        if len(subgrupos) != 2:
            return f"❌ O gráfico espera exatamente 2 subgrupos e encontrou {len(subgrupos)}.", None, None

        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        for ax, sub in zip(axs, subgrupos):
            dados_sub = dados[dados[subgrupo] == sub]
            sns.boxplot(data=dados_sub[lista_y], orient="v", ax=ax)
            ax.set_title(f"Boxplot por {sub}")

        plt.tight_layout()

        info_grafico["titulo_grafico"] = f"Boxplot por {subgrupos[0]} e {subgrupos[1]}"
        info_grafico["titulo_x"] = ", ".join(lista_y)
        info_grafico["titulo_y"] = ""
    else:
        dados = df[lista_y].dropna()
        if dados.empty:
            return "❌ Dados insuficientes para gerar o gráfico.", None, None

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=dados, orient="v")
        plt.title(f"Boxplot de {' e '.join(lista_y)}")
        plt.tight_layout()

        info_grafico["titulo_grafico"] = f"Boxplot de {' e '.join(lista_y)}"
        info_grafico["titulo_x"] = ", ".join(lista_y) if len(lista_y) > 1 else lista_y[0]
        info_grafico["titulo_y"] = "Valor"

    # ✅ Adiciona lista_y no info_grafico para o frontend
    info_grafico["lista_y"] = lista_y

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return "", imagem_base64, info_grafico


def personalizar_boxplot(df, lista_y, 
                         subgrupo=None,
                         titulo_grafico="", 
                         tamanho_fonte=12,
                         titulo_x="",
                         titulo_y="",
                         inclinacao_x="",
                         cor=""):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    aplicar_estilo_minitab()

    if not lista_y:
        return None, None

    if isinstance(lista_y, str):
        lista_y = [lista_y]

    colunas_validas = [col for col in lista_y if col in df.columns]
    if not colunas_validas:
        return None, None

    dados = df[colunas_validas + ([subgrupo] if subgrupo and subgrupo in df.columns else [])].dropna()
    if dados.empty:
        return None, None

    if subgrupo and subgrupo in df.columns:
        subgrupos = dados[subgrupo].unique()
        fig, axs = plt.subplots(1, len(subgrupos), figsize=(8*len(subgrupos), 6), sharey=True)

        if len(subgrupos) == 1:
            axs = [axs]

        for ax, sg in zip(axs, subgrupos):
            dados_sub = dados[dados[subgrupo] == sg]
            if len(colunas_validas) == 1:
                sns.boxplot(y=colunas_validas[0], data=dados_sub, ax=ax, color=cor if cor else None)
                ax.set_ylabel(titulo_y if titulo_y else colunas_validas[0], fontsize=int(tamanho_fonte))
            else:
                sns.boxplot(data=dados_sub[colunas_validas], orient="v", ax=ax)
                ax.set_ylabel(titulo_y if titulo_y else "Valor", fontsize=int(tamanho_fonte))

            # ✅ NUNCA define xlabel duplicado quando mais de 1 Y
            if len(colunas_validas) == 1:
                ax.set_xlabel(titulo_x, fontsize=int(tamanho_fonte))
            else:
                ax.set_xlabel("", fontsize=int(tamanho_fonte))

            ax.set_title(f"Boxplot por {sg}", fontsize=int(tamanho_fonte))

            if inclinacao_x:
                plt.setp(ax.get_xticklabels(), rotation=float(inclinacao_x), fontsize=int(tamanho_fonte))
            else:
                plt.setp(ax.get_xticklabels(), fontsize=int(tamanho_fonte))

        titulo_padrao = f"Boxplot por {' e '.join(subgrupos)}"

    else:
        fig, ax = plt.subplots(figsize=(10, 6))

        if len(colunas_validas) == 1:
            sns.boxplot(y=colunas_validas[0], data=dados, ax=ax, color=cor if cor else None)
            ax.set_ylabel(titulo_y if titulo_y else colunas_validas[0], fontsize=int(tamanho_fonte))
            ax.set_xlabel(titulo_x, fontsize=int(tamanho_fonte))
        else:
            sns.boxplot(data=dados[colunas_validas], orient="v", ax=ax)
            ax.set_ylabel(titulo_y if titulo_y else "Valor", fontsize=int(tamanho_fonte))
            # ✅ Remove xlabel duplicado para 2Y
            ax.set_xlabel("", fontsize=int(tamanho_fonte))

        titulo_padrao = f"Boxplot de {' e '.join(colunas_validas)}"
        ax.set_title(titulo_grafico.strip() if titulo_grafico.strip() else titulo_padrao, fontsize=int(tamanho_fonte))

        if inclinacao_x:
            plt.setp(ax.get_xticklabels(), rotation=float(inclinacao_x), fontsize=int(tamanho_fonte))
        else:
            plt.setp(ax.get_xticklabels(), fontsize=int(tamanho_fonte))

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    info_grafico = {
        "cor": cor or "",
        "titulo_grafico": titulo_grafico.strip() if titulo_grafico.strip() else titulo_padrao,
        "titulo_x": titulo_x,
        "titulo_y": titulo_y,
        "tamanho_fonte": tamanho_fonte or "",
        "inclinacao_x": inclinacao_x or "",
        "inclinacao_y": "",
        "espessura": "",
        "lista_y": lista_y
    }

    info_grafico["subgrupo"] = subgrupo if subgrupo else ""

    return imagem_base64, info_grafico


def gerar_dispersao(df, coluna_y, coluna_x, subgrupo=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None, None

    if not coluna_x or coluna_x not in df.columns:
        return "❌ A coluna X informada não foi encontrada no arquivo.", None, None

    aplicar_estilo_minitab()

    info_grafico = {
        "titulo_grafico": f"Dispersão de {coluna_y} por {coluna_x}",
        "tamanho_fonte": 12,
        "titulo_x": coluna_x,
        "titulo_y": coluna_y,
        "inclinacao_x": 0,
        "cor": None,  # ✅ inicializa sem cor fixa
        "lista_y": [coluna_y]
    }

    try:
        plt.figure(figsize=(10, 6))

        # 🔧 Captura cor usada
        if subgrupo and subgrupo in df.columns:
            plot = sns.scatterplot(x=coluna_x, y=coluna_y, hue=subgrupo, data=df)
            cor_usada = None  # quando há hue, cores múltiplas
            plt.title(f"Dispersão de {coluna_y} por {coluna_x} (Subgrupo: {subgrupo})")
        else:
            plot = sns.scatterplot(x=coluna_x, y=coluna_y, data=df)
            # Captura a cor real do primeiro ponto, se existir
            colecoes = plot.collections
            if colecoes:
                cor_usada = colecoes[0].get_facecolor()[0] if colecoes[0].get_facecolor().size > 0 else None
            else:
                cor_usada = None
            plt.title(f"Dispersão de {coluna_y} por {coluna_x}")

        info_grafico["cor"] = cor_usada

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return "", imagem_base64, info_grafico

    except Exception as e:
        return f"❌ Erro ao gerar o gráfico de dispersão: {str(e)}", None, None

def personalizar_dispersao(df, coluna_y, coluna_x, cor="#000000", titulo_x="", titulo_y="", titulo_grafico="", tamanho_fonte=12, inclinacao_x=0):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    aplicar_estilo_minitab()

    if not coluna_y or coluna_y not in df.columns:
        return None, None
    if not coluna_x or coluna_x not in df.columns:
        return None, None

    plt.figure(figsize=(10, 6))
    plot = sns.scatterplot(x=coluna_x, y=coluna_y, data=df, color=cor)

    # 🔧 Captura a cor real usada
    colecoes = plot.collections
    if colecoes:
        cor_usada_final = colecoes[0].get_facecolor()[0] if colecoes[0].get_facecolor().size > 0 else None
    else:
        cor_usada_final = cor

    plt.xlabel(titulo_x.strip() if titulo_x.strip() != "" else coluna_x, fontsize=int(tamanho_fonte))
    plt.ylabel(titulo_y.strip() if titulo_y.strip() != "" else coluna_y, fontsize=int(tamanho_fonte))
    titulo_padrao = titulo_grafico.strip() if titulo_grafico.strip() != "" else f"Dispersão de {coluna_y} por {coluna_x}"
    plt.title(titulo_padrao, fontsize=int(tamanho_fonte))
    plt.xticks(rotation=int(inclinacao_x), fontsize=int(tamanho_fonte))
    plt.yticks(fontsize=int(tamanho_fonte))
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    info_grafico = {
        "cor": cor if cor else cor_usada_final,
        "titulo_grafico": titulo_padrao,
        "titulo_x": titulo_x,
        "titulo_y": titulo_y,
        "tamanho_fonte": tamanho_fonte or "",
        "inclinacao_x": inclinacao_x or "",
        "inclinacao_y": "",
        "espessura": "",
        "lista_y": [coluna_y] if coluna_y else []
    }

    return imagem_base64, info_grafico





  
def gerar_tendencia(df, coluna_y, Data=None, subgrupo=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None, None

    if Data and Data not in df.columns:
        return "❌ A coluna Data informada não foi encontrada no arquivo.", None, None

    if subgrupo and subgrupo not in df.columns:
        return "❌ A coluna Subgrupo informada não foi encontrada no arquivo.", None, None

    aplicar_estilo_minitab()

    info_grafico = {
        "titulo_grafico": "",
        "tamanho_fonte": 12,
        "titulo_x": "",
        "titulo_y": coluna_y,
        "inclinacao_x": 0,
        "cor": None,  # ✅ inicializa sem cor fixa
        "lista_y": [coluna_y]
    }

    try:
        plt.figure(figsize=(10, 6))

        df = df.dropna(subset=[coluna_y]).reset_index(drop=True)

        if Data:
            df = df.dropna(subset=[Data])
            eixo_x = df[Data]
            titulo_base = f"Tendência temporal de {coluna_y} por {Data}"
            x_label = Data
        else:
            df["sequencia"] = df.index + 1
            eixo_x = df["sequencia"]
            titulo_base = f"Tendência temporal de {coluna_y} por sequência"
            x_label = "Tempo / Sequência"

        if subgrupo:
            plot = sns.lineplot(x=eixo_x, y=coluna_y, hue=subgrupo, data=df, marker="o")
            cor_usada = None  # múltiplas cores quando há subgrupo
            titulo = f"{titulo_base} (Subgrupo: {subgrupo})"
        else:
            plot = sns.lineplot(x=eixo_x, y=coluna_y, data=df, marker="o")
            colecoes = plot.collections
            if colecoes:
                cor_usada = colecoes[0].get_facecolor()[0] if colecoes[0].get_facecolor().size > 0 else None
            else:
                cor_usada = None
            titulo = titulo_base

        plt.title(titulo)
        plt.xlabel(x_label)
        plt.ylabel(coluna_y)
        plt.tight_layout()

        info_grafico["cor"] = cor_usada
        info_grafico["titulo_grafico"] = titulo
        info_grafico["titulo_x"] = x_label

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return "", imagem_base64, info_grafico

    except Exception as e:
        return f"❌ Erro ao gerar o gráfico de tendência: {str(e)}", None, None



def personalizar_tendencia(df, coluna_y, Data=None, cor="#000000", titulo_x="", titulo_y="", titulo_grafico="", tamanho_fonte=12, inclinacao_x=0):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    aplicar_estilo_minitab()

    if not coluna_y or coluna_y not in df.columns:
        return None, None
    if Data and Data not in df.columns:
        return None, None

    df = df.dropna(subset=[coluna_y]).reset_index(drop=True)

    plt.figure(figsize=(10, 6))

    if Data:
        df = df.dropna(subset=[Data])
        eixo_x = df[Data]
        x_label = titulo_x.strip() if titulo_x.strip() != "" else Data
    else:
        df["sequencia"] = df.index + 1
        eixo_x = df["sequencia"]
        x_label = titulo_x.strip() if titulo_x.strip() != "" else "Tempo / Sequência"

    plot = sns.lineplot(x=eixo_x, y=coluna_y, data=df, color=cor, marker="o")

    # 🔧 Captura a cor real usada
    colecoes = plot.collections
    if colecoes:
        cor_usada_final = colecoes[0].get_facecolor()[0] if colecoes[0].get_facecolor().size > 0 else None
    else:
        cor_usada_final = cor

    plt.xlabel(x_label, fontsize=int(tamanho_fonte))
    plt.ylabel(titulo_y.strip() if titulo_y.strip() != "" else coluna_y, fontsize=int(tamanho_fonte))
    titulo_padrao = titulo_grafico.strip() if titulo_grafico.strip() != "" else f"Tendência temporal de {coluna_y}"
    plt.title(titulo_padrao, fontsize=int(tamanho_fonte))
    plt.xticks(rotation=int(inclinacao_x), fontsize=int(tamanho_fonte))
    plt.yticks(fontsize=int(tamanho_fonte))
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    info_grafico = {
        "cor": cor if cor else cor_usada_final,
        "titulo_grafico": titulo_padrao,
        "titulo_x": titulo_x,
        "titulo_y": titulo_y,
        "tamanho_fonte": tamanho_fonte or "",
        "inclinacao_x": inclinacao_x or "",
        "inclinacao_y": "",
        "espessura": "",
        "lista_y": [coluna_y] if coluna_y else []
    }

    return imagem_base64, info_grafico




def gerar_bolhas_3d(df, coluna_y, coluna_x, coluna_z):
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    if not coluna_x or coluna_x not in df.columns:
        return "❌ A coluna X informada não foi encontrada no arquivo.", None, None
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None, None
    if not coluna_z or coluna_z not in df.columns:
        return "❌ A coluna Z informada não foi encontrada no arquivo.", None, None

    dados = df[[coluna_x, coluna_y, coluna_z]].dropna()
    if dados.empty:
        return "❌ Dados insuficientes para gerar o gráfico.", None, None

    aplicar_estilo_minitab()

    info_grafico = {
        "titulo_grafico": f"Gráfico de Bolhas: {coluna_x} vs {coluna_y} (Z = tamanho das bolhas)",
        "tamanho_fonte": 12,
        "titulo_x": coluna_x,
        "titulo_y": coluna_y,
        "inclinacao_x": 0,
        "cor": None,  # ✅ inicializa sem cor fixa
        "lista_y": [coluna_y, coluna_z]
    }

    try:
        plt.figure(figsize=(10, 6))

        scatter = plt.scatter(
            x=dados[coluna_x],
            y=dados[coluna_y],
            s=dados[coluna_z] * 30,
            alpha=0.5,
            edgecolors="w"
        )

        # 🔧 Captura cor real usada
        cor_usada = scatter.get_facecolor()[0] if scatter.get_facecolor().size > 0 else None
        info_grafico["cor"] = cor_usada

        plt.xlabel(coluna_x)
        plt.ylabel(coluna_y)
        plt.title(info_grafico["titulo_grafico"])
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return "", imagem_base64, info_grafico

    except Exception as e:
        return f"❌ Erro ao gerar o gráfico de bolhas 3D: {str(e)}", None, None


def personalizar_bolhas_3d(df, coluna_y, coluna_x, coluna_z, cor="#000000", titulo_x="", titulo_y="", titulo_grafico="", tamanho_fonte=12, inclinacao_x=0):
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    aplicar_estilo_minitab()

    if not coluna_x or coluna_x not in df.columns:
        return None, None
    if not coluna_y or coluna_y not in df.columns:
        return None, None
    if not coluna_z or coluna_z not in df.columns:
        return None, None

    dados = df[[coluna_x, coluna_y, coluna_z]].dropna()
    if dados.empty:
        return None, None

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        x=dados[coluna_x],
        y=dados[coluna_y],
        s=dados[coluna_z] * 30,
        alpha=0.5,
        color=cor,
        edgecolors="w"
    )

    # 🔧 Captura cor real usada
    cor_usada_final = scatter.get_facecolor()[0] if scatter.get_facecolor().size > 0 else cor

    plt.xlabel(titulo_x.strip() if titulo_x.strip() != "" else coluna_x, fontsize=int(tamanho_fonte))
    plt.ylabel(titulo_y.strip() if titulo_y.strip() != "" else coluna_y, fontsize=int(tamanho_fonte))
    titulo_padrao = titulo_grafico.strip() if titulo_grafico.strip() != "" else f"Gráfico de Bolhas: {coluna_x} vs {coluna_y} (Z = tamanho das bolhas)"
    plt.title(titulo_padrao, fontsize=int(tamanho_fonte))
    plt.xticks(rotation=int(inclinacao_x), fontsize=int(tamanho_fonte))
    plt.yticks(fontsize=int(tamanho_fonte))
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    info_grafico = {
        "cor": cor if cor else cor_usada_final,
        "titulo_grafico": titulo_padrao,
        "titulo_x": titulo_x,
        "titulo_y": titulo_y,
        "tamanho_fonte": tamanho_fonte or "",
        "inclinacao_x": inclinacao_x or "",
        "inclinacao_y": "",
        "espessura": "",
        "lista_y": [coluna_y, coluna_z]
    }

    return imagem_base64, info_grafico






import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
from scipy.interpolate import griddata
import base64
import pandas as pd

def gerar_superficie_3d(df, coluna_y, coluna_x, coluna_z):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from scipy.interpolate import griddata
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    if not coluna_x or coluna_x not in df.columns:
        return "❌ A coluna X informada não foi encontrada no arquivo.", None, None
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None, None
    if not coluna_z or coluna_z not in df.columns:
        return "❌ A coluna Z informada não foi encontrada no arquivo.", None, None

    dados = df[[coluna_x, coluna_y, coluna_z]].dropna()
    if dados.empty:
        return "❌ Dados insuficientes para gerar o gráfico.", None, None

    try:
        aplicar_estilo_minitab()

        X = dados[coluna_x].astype(float).values
        Y = dados[coluna_y].astype(float).values
        Z = dados[coluna_z].astype(float).values

        xi = np.linspace(X.min(), X.max(), 60)
        yi = np.linspace(Y.min(), Y.max(), 60)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((X, Y), Z, (xi, yi), method='cubic')

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.9, antialiased=True)

        ax.set_xlabel(coluna_x, labelpad=12)
        ax.set_ylabel(coluna_y, labelpad=12)
        ax.set_zlabel(coluna_z, labelpad=12)
        ax.set_title("Gráfico de Superfície 3D", pad=20)

        fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close()
        buf.seek(0)
        imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

        info_grafico = {
            "titulo_grafico": "Gráfico de Superfície 3D",
            "tamanho_fonte": 12,
            "titulo_x": coluna_x,
            "titulo_y": coluna_y,
            "inclinacao_x": 0,
            "cor": "",
            "lista_y": [coluna_y, coluna_z]
        }

        return "", imagem_base64, info_grafico

    except Exception as e:
        return f"❌ Erro ao gerar superfície 3D: {str(e)}", None, None


def personalizar_superficie_3d(df, coluna_y, coluna_x, coluna_z, cor=None, titulo_x="", titulo_y="", titulo_grafico="", tamanho_fonte=12, inclinacao_x=0):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from scipy.interpolate import griddata
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    aplicar_estilo_minitab()

    if not coluna_x or coluna_x not in df.columns:
        return None, None
    if not coluna_y or coluna_y not in df.columns:
        return None, None
    if not coluna_z or coluna_z not in df.columns:
        return None, None

    dados = df[[coluna_x, coluna_y, coluna_z]].dropna()
    if dados.empty:
        return None, None

    X = dados[coluna_x].astype(float).values
    Y = dados[coluna_y].astype(float).values
    Z = dados[coluna_z].astype(float).values

    xi = np.linspace(X.min(), X.max(), 60)
    yi = np.linspace(Y.min(), Y.max(), 60)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((X, Y), Z, (xi, yi), method='cubic')

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # ➡️ Usa cmap do gráfico original se vier preenchido, senão usa 'viridis'
    cmap_utilizado = cor if cor in plt.colormaps() else "viridis"

    surf = ax.plot_surface(xi, yi, zi, cmap=cmap_utilizado, edgecolor='k', linewidth=0.2, alpha=0.9, antialiased=True)

    ax.set_xlabel(titulo_x.strip() if titulo_x.strip() != "" else coluna_x, labelpad=12, fontsize=int(tamanho_fonte))
    ax.set_ylabel(titulo_y.strip() if titulo_y.strip() != "" else coluna_y, labelpad=12, fontsize=int(tamanho_fonte))
    ax.set_zlabel(coluna_z, labelpad=12, fontsize=int(tamanho_fonte))
    titulo_padrao = titulo_grafico.strip() if titulo_grafico.strip() != "" else "Gráfico de Superfície 3D"
    ax.set_title(titulo_padrao, pad=20, fontsize=int(tamanho_fonte))

    ax.view_init(elev=30, azim=int(inclinacao_x))

    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    info_grafico = {
        "cor": cmap_utilizado,
        "titulo_grafico": titulo_padrao,
        "titulo_x": titulo_x,
        "titulo_y": titulo_y,
        "tamanho_fonte": tamanho_fonte or "",
        "inclinacao_x": inclinacao_x or "",
        "inclinacao_y": "",
        "espessura": "",
        "lista_y": [coluna_y, coluna_z] if coluna_y and coluna_z else []
    }

    return imagem_base64, info_grafico





from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import base64
from io import BytesIO

def gerar_dispersao_3d_com_regressao(df, coluna_y, coluna_x, coluna_z):
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from io import BytesIO
    import base64
    from suporte import aplicar_estilo_minitab

    if not coluna_x or coluna_x not in df.columns:
        return "❌ A coluna X informada não foi encontrada no arquivo.", None, None
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None, None
    if not coluna_z or coluna_z not in df.columns:
        return "❌ A coluna Z informada não foi encontrada no arquivo.", None, None

    dados = df[[coluna_x, coluna_y, coluna_z]].dropna()
    if dados.empty:
        return "❌ Dados insuficientes para gerar o gráfico.", None, None

    try:
        aplicar_estilo_minitab()

        X = dados[[coluna_x, coluna_y]].values
        y = dados[coluna_z].values

        model = LinearRegression()
        model.fit(X, y)

        x_range = np.linspace(dados[coluna_x].min(), dados[coluna_x].max(), 20)
        y_range = np.linspace(dados[coluna_y].min(), dados[coluna_y].max(), 20)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        z_pred = model.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(dados[coluna_x], dados[coluna_y], dados[coluna_z], color='b', label='Pontos reais')

        ax.plot_surface(x_grid, y_grid, z_pred, alpha=0.5, color='red')

        ax.set_xlabel(coluna_x)
        ax.set_ylabel(coluna_y)
        ax.set_zlabel(coluna_z)
        titulo = f'Dispersão 3D com Regressão - {coluna_z} ~ {coluna_x} + {coluna_y}'
        ax.set_title(titulo)
        plt.tight_layout()

        # Captura cor real usada
        cor_usada = scatter.get_facecolor()[0] if scatter.get_facecolor().size > 0 else None

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

        info_grafico = {
            "titulo_grafico": titulo,
            "tamanho_fonte": 12,
            "titulo_x": coluna_x,
            "titulo_y": coluna_y,
            "inclinacao_x": 0,
            "cor": cor_usada,
            "lista_y": [coluna_y, coluna_z]
        }

        return "", imagem_base64, info_grafico

    except Exception as e:
        return f"❌ Erro ao gerar dispersão 3D com regressão: {str(e)}", None, None

def personalizar_dispersao_3d_com_regressao(df, coluna_y, coluna_x, coluna_z, cor="red", titulo_x="", titulo_y="", titulo_grafico="", tamanho_fonte=12, inclinacao_x=30):
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from io import BytesIO
    import base64
    from suporte import aplicar_estilo_minitab

    aplicar_estilo_minitab()

    if not coluna_x or coluna_x not in df.columns:
        return None, None
    if not coluna_y or coluna_y not in df.columns:
        return None, None
    if not coluna_z or coluna_z not in df.columns:
        return None, None

    dados = df[[coluna_x, coluna_y, coluna_z]].dropna()
    if dados.empty:
        return None, None

    X = dados[[coluna_x, coluna_y]].values
    y = dados[coluna_z].values

    model = LinearRegression()
    model.fit(X, y)

    x_range = np.linspace(dados[coluna_x].min(), dados[coluna_x].max(), 20)
    y_range = np.linspace(dados[coluna_y].min(), dados[coluna_y].max(), 20)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_pred = model.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(dados[coluna_x], dados[coluna_y], dados[coluna_z], color="blue", label='Pontos reais')

    ax.plot_surface(x_grid, y_grid, z_pred, alpha=0.5, color=cor)

    ax.set_xlabel(titulo_x.strip() if titulo_x.strip() != "" else coluna_x, fontsize=int(tamanho_fonte))
    ax.set_ylabel(titulo_y.strip() if titulo_y.strip() != "" else coluna_y, fontsize=int(tamanho_fonte))
    ax.set_zlabel(coluna_z, fontsize=int(tamanho_fonte))
    titulo_padrao = titulo_grafico.strip() if titulo_grafico.strip() != "" else f"Dispersão 3D com Regressão - {coluna_z} ~ {coluna_x} + {coluna_y}"
    ax.set_title(titulo_padrao, fontsize=int(tamanho_fonte))
    ax.view_init(elev=30, azim=int(inclinacao_x))

    plt.tight_layout()

    cor_usada_final = scatter.get_facecolor()[0] if scatter.get_facecolor().size > 0 else cor

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    info_grafico = {
        "cor": cor if cor else cor_usada_final,
        "titulo_grafico": titulo_padrao,
        "titulo_x": titulo_x,
        "titulo_y": titulo_y,
        "tamanho_fonte": tamanho_fonte or "",
        "inclinacao_x": inclinacao_x or "",
        "inclinacao_y": "",
        "espessura": "",
        "lista_y": [coluna_y, coluna_z]
    }

    return imagem_base64, info_grafico






GRAFICOS = {
    "Histograma": gerar_histograma,
    "Histograma Personalizado": personalizar_histograma,
    "Pareto": gerar_pareto,
    "Pareto Personalizado": personalizar_pareto,
    "Setores (Pizza)": gerar_pizza,
    "Setores (Pizza) Personalizado": personalizar_pizza,
    "Barras": gerar_barras,
    "Barras Personalizado": personalizar_barras,
    "BoxPlot": gerar_boxplot,
    "BoxPlot Personalizado": personalizar_boxplot,
    "Dispersão": gerar_dispersao,
    "Dispersão Personalizado": personalizar_dispersao,
    "Tendência": gerar_tendencia,
    "Tendência Personalizado": personalizar_tendencia,
    "Bolhas - 3D": gerar_bolhas_3d,
    "Bolhas - 3D Personalizado": personalizar_bolhas_3d,
    "Superfície - 3D": gerar_superficie_3d,
    "Superfície - 3D Personalizado": personalizar_superficie_3d,
    "Dispersão 3D com Regressão": gerar_dispersao_3d_com_regressao,
    "Dispersão 3D com Regressão Personalizado": personalizar_dispersao_3d_com_regressao
}

# ✅ Configuração de personalização permitida para cada tipo de boxplot
CONFIG_PERSONALIZACAO_BOX = {
    "BoxPlot_1Y": ["titulo_grafico", "titulo_x", "titulo_y", "tamanho_fonte", "inclinacao_x", "cor"],
    "BoxPlot_2Y": ["titulo_grafico", "titulo_y", "tamanho_fonte", "inclinacao_x"],
    "BoxPlot_2Y_subgrupo": ["titulo_y", "tamanho_fonte", "inclinacao_x"]
}
