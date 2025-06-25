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
    import base64
    from io import BytesIO
    from suporte import aplicar_estilo_minitab

    aplicar_estilo_minitab()

    if not coluna_x or coluna_x not in df.columns:
        return "❌ A coluna X informada não foi encontrada no arquivo.", None
    if coluna_y and coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None
    if subgrupo and subgrupo not in df.columns:
        return "❌ A coluna Subgrupo informada não foi encontrada no arquivo.", None

    try:
        def plotar(dados, titulo, ax):
            if coluna_y:
                contagem = dados.groupby(coluna_x)[coluna_y].sum().sort_values(ascending=False)
            else:
                contagem = dados[coluna_x].value_counts().sort_values(ascending=False)

            acumulado = contagem.cumsum() / contagem.sum() * 100
            contagem.plot(kind="bar", ax=ax, color="C0")
            ax.set_ylabel("Frequência / Soma")
            ax.set_title(titulo)

            ax2 = ax.twinx()
            ax2.plot(acumulado.index, acumulado.values, color="red", marker="o")
            ax2.set_ylabel("Acumulado (%)")
            ax2.set_ylim(0, 110)

            for i, (x, y) in enumerate(zip(contagem.index, acumulado)):
                ax2.text(i, y + 2, f"{y:.1f}%", color="red", ha="center", fontsize=8)

        if subgrupo:
            colunas = [coluna_x, subgrupo] + ([coluna_y] if coluna_y else [])
            dados = df[colunas].dropna()
            if dados.empty:
                return "❌ Dados insuficientes para gerar o gráfico.", None

            subgrupos = dados[subgrupo].unique()
            if len(subgrupos) != 2:
                return f"❌ O gráfico espera exatamente 2 subgrupos e encontrou {len(subgrupos)}.", None

            fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
            for ax, sub in zip(axs, subgrupos):
                dados_sub = dados[dados[subgrupo] == sub]
                plotar(dados_sub, f"Pareto - {coluna_x} ({sub})", ax)
        else:
            dados = df[[coluna_x, coluna_y]].dropna() if coluna_y else df[[coluna_x]].dropna()
            if dados.empty:
                return "❌ Dados insuficientes para gerar o gráfico.", None

            fig, ax = plt.subplots(figsize=(10, 5))
            plotar(dados, f"Pareto - {coluna_x}", ax)

        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
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
        s=dados[coluna_z] * 30,
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
    from suporte import aplicar_estilo_minitab

    if not coluna_x or coluna_x not in df.columns:
        return "❌ A coluna X informada não foi encontrada no arquivo.", None
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None
    if not coluna_z or coluna_z not in df.columns:
        return "❌ A coluna Z informada não foi encontrada no arquivo.", None

    dados = df[[coluna_x, coluna_y, coluna_z]].dropna()
    if dados.empty:
        return "❌ Dados insuficientes para gerar o gráfico.", None

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

        return "", imagem_base64

    except Exception as e:
        return f"❌ Erro ao gerar superfície 3D: {str(e)}", None


from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import base64
from io import BytesIO

def gerar_dispersao_3d_com_regressao(df, col_x, col_y, col_z):
    # Dados
    X = df[[col_x, col_y]].values
    y = df[col_z].values

    # Regressão
    model = LinearRegression()
    model.fit(X, y)

    # Grade para superfície
    x_range = np.linspace(df[col_x].min(), df[col_x].max(), 20)
    y_range = np.linspace(df[col_y].min(), df[col_y].max(), 20)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_pred = model.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)

    # Gráfico
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Dispersão real
    ax.scatter(df[col_x], df[col_y], df[col_z], color='b', label='Pontos reais')

    # Plano de regressão
    ax.plot_surface(x_grid, y_grid, z_pred, alpha=0.5, color='red', label='Plano de regressão')

    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.set_zlabel(col_z)
    ax.set_title(f'Dispersão 3D com Regressão - {col_z} ~ {col_x} + {col_y}')
    plt.tight_layout()

    # Exporta imagem
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return "", imagem_base64


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
    "Dispersão 3D com Regressão": gerar_dispersao_3d_regressao
    
}




}
