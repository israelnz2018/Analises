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
    # Garante que coluna_y seja string e não lista
    if isinstance(coluna_y, list):
        coluna_y = coluna_y[0] if coluna_y else None

    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    aplicar_estilo_minitab()
    plt.figure(figsize=(10, 6))

    if subgrupo and subgrupo in df.columns:
        grupos = df.groupby(subgrupo)
        cores = sns.color_palette("tab10", n_colors=len(grupos))

        for i, (nome, grupo) in enumerate(grupos):
            sns.histplot(
                grupo[coluna_y],
                bins=10,
                kde=True,
                stat="density",
                alpha=0.4,
                label=str(nome),
                color=cores[i],
                edgecolor="black"
            )
    else:
        sns.histplot(
            df[coluna_y],
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
    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return "", imagem_base64






def gerar_pareto(df, coluna_y, subgrupo=None, subgrupo2=None):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    aplicar_estilo_minitab()

    if not subgrupo:
        contagem = df[coluna_y].value_counts().sort_values(ascending=False)
        if contagem.sum() == 0:
            return "❌ Dados insuficientes para gerar o gráfico.", None

        plt.figure(figsize=(8, 5))
        ax = contagem.plot(kind="bar")
        cum = contagem.cumsum() / contagem.sum() * 100
        ax2 = cum.plot(secondary_y=True, color='r', marker='o', ax=ax)
        ax.set_ylabel("Frequência")
        ax2.set_ylabel("Acumulado (%)")
        ax.set_title(f"Pareto - {coluna_y}")
        plt.tight_layout()

    elif subgrupo and subgrupo in df.columns and not subgrupo2:
        dados = df[[coluna_y, subgrupo]].dropna()
        if dados.empty:
            return "❌ Dados insuficientes para gerar o gráfico.", None

        soma = dados.groupby(subgrupo)[coluna_y].value_counts().unstack().fillna(0)
        soma.plot(kind="bar", stacked=True, figsize=(8, 5))
        plt.ylabel("Frequência")
        plt.title(f"Pareto - {coluna_y} por {subgrupo}")
        plt.tight_layout()

    elif subgrupo and subgrupo in df.columns and subgrupo2 and subgrupo2 in df.columns:
        dados = df[[coluna_y, subgrupo, subgrupo2]].dropna()
        if dados.empty:
            return "❌ Dados insuficientes para gerar o gráfico.", None

        fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

        for ax, sub, titulo in zip(axs, [subgrupo, subgrupo2], [subgrupo, subgrupo2]):
            soma = dados.groupby(sub)[coluna_y].value_counts().unstack().fillna(0)
            soma.plot(kind="bar", stacked=True, ax=ax)
            ax.set_ylabel("Frequência")
            ax.set_title(f"Pareto - {coluna_y} por {titulo}")

        plt.tight_layout()

    else:
        return "❌ Os subgrupos informados não foram encontrados no arquivo.", None

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return "", imagem_base64




def gerar_pizza(df, coluna_y, subgrupo):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    aplicar_estilo_minitab()
    plt.figure(figsize=(8, 8))

    if subgrupo and subgrupo in df.columns:
        dados = df[[coluna_y, subgrupo]].dropna()
        soma = dados.groupby(subgrupo)[coluna_y].sum().sort_values(ascending=False)
        if soma.sum() == 0:
            return "❌ Dados insuficientes para gerar o gráfico.", None
        soma.plot.pie(autopct='%1.1f%%', startangle=90, legend=False)
        plt.title(f"Pizza de {coluna_y} por {subgrupo}")
    else:
        contagem = df[coluna_y].value_counts().sort_values(ascending=False)
        if contagem.sum() == 0:
            return "❌ Dados insuficientes para gerar o gráfico.", None
        contagem.plot.pie(autopct='%1.1f%%', startangle=90, legend=False)
        plt.title(f"Pizza de frequência por {coluna_y}")

    plt.ylabel("")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return "", imagem_base64


def gerar_barras(df, coluna_y, subgrupo, subgrupo2=None):
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    aplicar_estilo_minitab()
    plt.figure(figsize=(10, 6))

    if subgrupo and subgrupo in df.columns and not subgrupo2:
        dados = df[[coluna_y, subgrupo]].dropna()
        contagem = dados.groupby(subgrupo)[coluna_y].value_counts().unstack().fillna(0)
        contagem.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.ylabel("Frequência")
        plt.title(f"Barras de {coluna_y} por {subgrupo}")

    elif subgrupo and subgrupo in df.columns and subgrupo2 and subgrupo2 in df.columns:
        dados = df[[coluna_y, subgrupo, subgrupo2]].dropna()
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        for ax, sub, titulo in zip(axs, [subgrupo, subgrupo2], [subgrupo, subgrupo2]):
            contagem = dados.groupby(sub)[coluna_y].value_counts().unstack().fillna(0)
            contagem.plot(kind="bar", stacked=True, ax=ax)
            ax.set_ylabel("Frequência")
            ax.set_title(f"Barras de {coluna_y} por {titulo}")

        plt.tight_layout()

    else:
        contagem = df[coluna_y].value_counts().sort_values(ascending=False)
        contagem.plot(kind="bar", ax=plt.gca())
        plt.ylabel("Frequência")
        plt.title(f"Barras de frequência por {coluna_y}")

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return "", imagem_base64

def gerar_boxplot(df, lista_y, subgrupo, subgrupo2=None):
    if not lista_y or any(y not in df.columns for y in lista_y):
        return "❌ Uma ou mais colunas Y informadas não foram encontradas no arquivo.", None

    aplicar_estilo_minitab()
    plt.figure(figsize=(10, 6))

    if subgrupo and subgrupo in df.columns and not subgrupo2:
        if len(lista_y) != 1:
            return "⚠ Para BoxPlot com subgrupo, selecione apenas uma variável Y.", None
        sns.boxplot(x=subgrupo, y=lista_y[0], data=df, orient="v")
        plt.title(f"Boxplot de {lista_y[0]} por {subgrupo}")

    elif subgrupo and subgrupo in df.columns and subgrupo2 and subgrupo2 in df.columns:
        dados = df[[*lista_y, subgrupo, subgrupo2]].dropna()
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        for ax, sub, titulo in zip(axs, [subgrupo, subgrupo2], [subgrupo, subgrupo2]):
            sns.boxplot(x=sub, y=lista_y[0], data=dados, orient="v", ax=ax)
            ax.set_title(f"Boxplot de {lista_y[0]} por {titulo}")

        plt.tight_layout()

    else:
        dados = df[lista_y].dropna()
        sns.boxplot(data=dados, orient="v")
        plt.title("Boxplot de variáveis contínuas")

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
    if not coluna_y or coluna_y not in df.columns:
        return "❌ A coluna Y informada não foi encontrada no arquivo.", None

    aplicar_estilo_minitab()
    plt.figure(figsize=(10, 6))

    df = df.dropna(subset=[coluna_y]).reset_index(drop=True)

    if Data and Data in df.columns:
        df = df.dropna(subset=[Data])
        eixo_x = df[Data]
        titulo_base = f"Tendência temporal de {coluna_y} por {Data}"
    else:
        df["sequencia"] = df.index + 1
        eixo_x = df["sequencia"]
        titulo_base = f"Tendência temporal de {coluna_y} por sequência"

    if subgrupo and subgrupo in df.columns:
        sns.lineplot(x=eixo_x, y=coluna_y, hue=subgrupo, data=df, marker="o")
        titulo = f"{titulo_base} (Subgrupo: {subgrupo})"
    else:
        sns.lineplot(x=eixo_x, y=coluna_y, data=df, marker="o")
        titulo = titulo_base

    plt.title(titulo)
    plt.xlabel(Data if Data and Data in df.columns else "Tempo / Sequência")
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
        s=dados[coluna_z] * 10,
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
