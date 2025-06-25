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

    # Garante que coluna_y seja string
    if isinstance(coluna_y, list):
        coluna_y = coluna_y[0] if coluna_y else None

    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    aplicar_estilo_minitab()

    if subgrupo:
        if subgrupo not in df.columns:
            return "❌ A coluna Subgrupo informada não foi encontrada no arquivo.", None

        dados = df[[coluna_y, subgrupo]].dropna()
        if dados.empty:
            return "❌ Dados insuficientes para gerar o gráfico.", None

        subgrupos = dados[subgrupo].unique()
        if len(subgrupos) != 2:
            return f"❌ O gráfico espera exatamente 2 subgrupos e encontrou {len(subgrupos)}.", None

        fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        for ax, sub in zip(axs, subgrupos):
            sns.histplot(
                dados[dados[subgrupo] == sub][coluna_y],
                bins=10,
                kde=True,
                stat="density",
                alpha=0.4,
                color="steelblue",
                edgecolor="black",
                ax=ax
            )
            ax.set_title(f"Histograma - {sub}")
            ax.set_xlabel(coluna_y)
            ax.set_ylabel("Densidade")
    else:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            df[coluna_y].dropna(),
            bins=10,
            kde=True,
            stat="density",
            alpha=0.4,
            color="steelblue",
            edgecolor="black"
        )
        plt.xlabel(coluna_y)
        plt.ylabel("Densidade")
        plt.title("Histograma com Curva de Densidade")
        plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return "", imagem_base64


def gerar_pareto(df, coluna_x, coluna_y=None, subgrupo=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import base64
    from io import BytesIO
    from estilo import aplicar_estilo_minitab

    aplicar_estilo_minitab()

    if not coluna_x or coluna_x not in df.columns:
        return "❌ A coluna X informada não foi encontrada no arquivo.", None

    if subgrupo and subgrupo not in df.columns:
        return "❌ A coluna Subgrupo informada não foi encontrada no arquivo.", None

    if coluna_y and coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    try:
        if subgrupo:
            colunas_usadas = [coluna_x, subgrupo]
            if coluna_y:
                colunas_usadas.append(coluna_y)
            dados = df[colunas_usadas].dropna()
            if dados.empty:
                return "❌ Dados insuficientes para gerar o gráfico.", None

            subgrupos = dados[subgrupo].unique()
            if len(subgrupos) != 2:
                return f"❌ O gráfico espera exatamente 2 subgrupos e encontrou {len(subgrupos)}.", None

            fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

            for ax, sub in zip(axs, subgrupos):
                dados_sub = dados[dados[subgrupo] == sub]
                if coluna_y:
                    soma = dados_sub.groupby(coluna_x)[coluna_y].sum().sort_values(ascending=False)
                else:
                    soma = dados_sub[coluna_x].value_counts().sort_values(ascending=False)

                soma.plot(kind="bar", ax=ax)
                cum = soma.cumsum() / soma.sum() * 100
                ax2 = ax.twinx()
                ax2.plot(cum.index, cum.values, color='r', marker='o')
                ax.set_ylabel("Frequência / Soma")
                ax2.set_ylabel("Acumulado (%)")
                ax.set_title(f"Pareto - {coluna_x} ({sub})")

            plt.tight_layout()

        else:
            dados = df[[coluna_x, coluna_y]].dropna() if coluna_y else df[[coluna_x]].dropna()
            if dados.empty:
                return "❌ Dados insuficientes para gerar o gráfico.", None

            if coluna_y:
                soma = dados.groupby(coluna_x)[coluna_y].sum().sort_values(ascending=False)
            else:
                soma = dados[coluna_x].value_counts().sort_values(ascending=False)

            plt.figure(figsize=(10, 5))
            ax = soma.plot(kind="bar")
            cum = soma.cumsum() / soma.sum() * 100
            ax2 = ax.twinx()
            ax2.plot(cum.index, cum.values, color='r', marker='o')
            ax.set_ylabel("Frequência / Soma")
            ax2.set_ylabel("Acumulado (%)")
            ax.set_title(f"Pareto - {coluna_x}")
            plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return "", imagem_base64

    except Exception as e:
        return f"❌ Erro ao gerar o gráfico de Pareto: {str(e)}", None







def gerar_pizza(df, coluna_x, coluna_y=None, subgrupo=None):
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    if not coluna_x or coluna_x not in df.columns:
        return "❌ A coluna X informada não foi encontrada no arquivo.", None

    if coluna_y and coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    if subgrupo and subgrupo not in df.columns:
        return "❌ A coluna Subgrupo informada não foi encontrada no arquivo.", None

    aplicar_estilo_minitab()

    try:
        if subgrupo:
            dados = df[[coluna_x, subgrupo] + ([coluna_y] if coluna_y else [])].dropna()
            subgrupos = dados[subgrupo].dropna().unique()

            if len(subgrupos) != 2:
                return f"❌ O gráfico espera exatamente 2 subgrupos e encontrou {len(subgrupos)}.", None

            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            for ax, sub in zip(axs, subgrupos):
                dados_sub = dados[dados[subgrupo] == sub]
                if coluna_y:
                    soma = dados_sub.groupby(coluna_x)[coluna_y].sum()
                else:
                    soma = dados_sub[coluna_x].value_counts()

                if soma.empty or soma.sum() == 0:
                    return f"❌ Dados insuficientes para gerar o gráfico para o subgrupo {sub}.", None

                soma.plot.pie(autopct='%1.1f%%', startangle=90, legend=False, ax=ax)
                ax.set_ylabel("")
                ax.set_title(f"Pizza de {coluna_x} ({sub})")

            plt.tight_layout()

        else:
            dados = df[[coluna_x] + ([coluna_y] if coluna_y else [])].dropna()
            if dados.empty:
                return "❌ Dados insuficientes para gerar o gráfico.", None

            if coluna_y:
                soma = dados.groupby(coluna_x)[coluna_y].sum()
            else:
                soma = dados[coluna_x].value_counts()

            if soma.empty or soma.sum() == 0:
                return "❌ Dados insuficientes para gerar o gráfico.", None

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

        return "", imagem_base64

    except Exception as e:
        return f"❌ Erro ao gerar o gráfico: {str(e)}", None



def gerar_barras(df, coluna_x, coluna_y=None, subgrupo=None):
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    if not coluna_x or coluna_x not in df.columns:
        return "❌ A coluna X informada não foi encontrada no arquivo.", None

    if coluna_y and coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    if subgrupo and subgrupo not in df.columns:
        return "❌ A coluna Subgrupo informada não foi encontrada no arquivo.", None

    aplicar_estilo_minitab()

    if subgrupo:
        dados = df[[coluna_x, coluna_y, subgrupo]].dropna() if coluna_y else df[[coluna_x, subgrupo]].dropna()
        subgrupos = dados[subgrupo].unique()

        if len(subgrupos) != 2:
            return f"❌ O gráfico espera exatamente 2 subgrupos e encontrou {len(subgrupos)}.", None

        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        for ax, sub in zip(axs, subgrupos):
            dados_sub = dados[dados[subgrupo] == sub]
            if coluna_y:
                contagem = dados_sub.groupby(coluna_x)[coluna_y].sum()
                contagem.plot(kind="bar", ax=ax)
                ax.set_ylabel("Soma de Y")
            else:
                contagem = dados_sub[coluna_x].value_counts()
                contagem.plot(kind="bar", ax=ax)
                ax.set_ylabel("Frequência")
            ax.set_title(f"Barras de {coluna_x} ({sub})")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        plt.tight_layout()

    else:
        dados = df[[coluna_x, coluna_y]].dropna() if coluna_y else df[[coluna_x]].dropna()
        if coluna_y:
            contagem = dados.groupby(coluna_x)[coluna_y].sum()
            contagem.plot(kind="bar", figsize=(10,6))
            plt.ylabel("Soma de Y")
        else:
            contagem = dados[coluna_x].value_counts()
            contagem.plot(kind="bar", figsize=(10,6))
            plt.ylabel("Frequência")
        plt.title(f"Barras de {coluna_x}")
        plt.xticks(rotation=90)  # <<< Aqui está o ajuste
        plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return "", imagem_base64



def gerar_boxplot(df, lista_y, subgrupo=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO

    if not lista_y or any(y not in df.columns for y in lista_y):
        return "❌ Uma ou mais colunas Y informadas não foram encontradas no arquivo.", None

    if subgrupo and subgrupo not in df.columns:
        return "❌ A coluna Subgrupo informada não foi encontrada no arquivo.", None

    aplicar_estilo_minitab()

    if subgrupo:
        dados = df[lista_y + [subgrupo]].dropna()
        subgrupos = dados[subgrupo].unique()

        if len(subgrupos) != 2:
            return f"❌ O gráfico espera exatamente 2 subgrupos e encontrou {len(subgrupos)}.", None

        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        for ax, sub in zip(axs, subgrupos):
            dados_sub = dados[dados[subgrupo] == sub]
            sns.boxplot(data=dados_sub[lista_y], orient="v", ax=ax)
            ax.set_title(f"Boxplot ({sub})")

        plt.tight_layout()

    else:
        dados = df[lista_y].dropna()
        if dados.empty:
            return "❌ Dados insuficientes para gerar o gráfico.", None
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=dados, orient="v")
        plt.title("Boxplot de variáveis contínuas")
        plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return "", imagem_base64


    
def gerar_dispersao(df, coluna_y, coluna_x, subgrupo=None):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    if not coluna_x or coluna_x not in df.columns:
        return "❌ A coluna X informada não foi encontrada no arquivo.", None

    aplicar_estilo_minitab()
    plt.figure(figsize=(10, 6))

    if subgrupo and subgrupo in df.columns:
        sns.scatterplot(x=coluna_x, y=coluna_y, hue=subgrupo, data=df)
        plt.title(f"Dispersão de {coluna_y} por {coluna_x} (Subgrupo: {subgrupo})")
    else:
        sns.scatterplot(x=coluna_x, y=coluna_y, data=df)
        plt.title(f"Dispersão de {coluna_y} por {coluna_x}")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return "", imagem_base64

    
def gerar_tendencia(df, coluna_y, Data=None, subgrupo=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO

    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    if Data and Data not in df.columns:
        return "❌ A coluna Data informada não foi encontrada no arquivo.", None

    if subgrupo and subgrupo not in df.columns:
        return "❌ A coluna Subgrupo informada não foi encontrada no arquivo.", None

    aplicar_estilo_minitab()
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
        sns.lineplot(x=eixo_x, y=coluna_y, hue=subgrupo, data=df, marker="o")
        titulo = f"{titulo_base} (Subgrupo: {subgrupo})"
    else:
        sns.lineplot(x=eixo_x, y=coluna_y, data=df, marker="o")
        titulo = titulo_base

    plt.title(titulo)
    plt.xlabel(x_label)
    plt.ylabel(coluna_y)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return "", imagem_base64



def gerar_bolhas_3d(df, coluna_y, coluna_x, coluna_z):
    if not coluna_x or coluna_x not in df.columns:
        return "❌ A coluna X informada não foi encontrada no arquivo.", None
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None
    if not coluna_z or coluna_z not in df.columns:
        return "❌ A coluna Z informada não foi encontrada no arquivo.", None

    dados = df[[coluna_x, coluna_y, coluna_z]].dropna()
    if dados.empty:
        return "❌ Dados insuficientes para gerar o gráfico.", None

    aplicar_estilo_minitab()
    plt.figure(figsize=(10, 6))

    plt.scatter(
        x=dados[coluna_x],
        y=dados[coluna_y],
        s=dados[coluna_z] * 20,
        alpha=0.5,
        edgecolors="w"
    )

    plt.xlabel(coluna_x)
    plt.ylabel(coluna_y)
    plt.title(f"Gráfico de Bolhas: {coluna_x} vs {coluna_y} (Z = tamanho das bolhas)")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return "", imagem_base64

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

    if not coluna_x or coluna_x not in df.columns:
        return "❌ A coluna X informada não foi encontrada no arquivo.", None
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None
    if not coluna_z or coluna_z not in df.columns:
        return "❌ A coluna Z informada não foi encontrada no arquivo.", None

    dados = df[[coluna_x, coluna_y, coluna_z]].dropna()
    if dados.empty:
        return "❌ Dados insuficientes para gerar o gráfico.", None

    dados = dados.astype(float)

    try:
        aplicar_estilo_minitab()

        X = dados[coluna_x].values
        Y = dados[coluna_y].values
        Z = dados[coluna_z].values

        xi = np.linspace(X.min(), X.max(), 50)
        yi = np.linspace(Y.min(), Y.max(), 50)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((X, Y), Z, (xi, yi), method='linear')

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none', alpha=0.8)

        ax.set_xlabel(coluna_x)
        ax.set_ylabel(coluna_y)
        ax.set_zlabel(coluna_z)
        ax.set_title(f"Superfície 3D: {coluna_x} x {coluna_y} x {coluna_z}")

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return "", imagem_base64

    except Exception as e:
        return f"❌ Erro ao gerar superfície 3D: {str(e)}", None



GRAFICOS = {
    "Histograma": gerar_histograma,
    "Pareto": gerar_pareto,
    "Setores (Pizza)": gerar_pizza,
    "Barras": gerar_barras,
    "BoxPlot": gerar_boxplot,
    "Dispersão": gerar_dispersao,
    "Tendência": gerar_tendencia,
    "Bolhas - 3D": gerar_bolhas_3d,
    "Superfície - 3D": gerar_superficie_3d,
    

}
