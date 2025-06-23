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

        x


def grafico_linha_temporal(df, colunas_usadas, coluna_y=None):
    if len(colunas_usadas) != 2:
        raise ValueError("O gráfico de série temporal requer duas colunas: X (datas/períodos) e Y (valores numéricos).")

    nome_x = colunas_usadas[1]  # X (tempo)
    nome_y = colunas_usadas[0]  # Y (valor)

    # Força conversão da coluna Y para numérico
    df[nome_y] = pd.to_numeric(df[nome_y], errors="coerce")
    if df[nome_y].isnull().all():
        raise ValueError(f"A coluna '{nome_y}' não contém valores numéricos válidos.")

    # Tenta converter X para datetime (se possível)
    try:
        df[nome_x] = pd.to_datetime(df[nome_x], errors='coerce')
    except Exception:
        pass

    # Verifica se ao menos parte das datas foram entendidas
    if df[nome_x].isnull().all():
        raise ValueError(f"A coluna '{nome_x}' não contém datas ou períodos válidos para série temporal.")

    # Ordena pela coluna X (tempo)
    df = df.sort_values(by=nome_x)

    aplicar_estilo_minitab()

    plt.figure(figsize=(10, 5))
    plt.plot(df[nome_x], df[nome_y], color='blue', marker='o', markerfacecolor='red')
    plt.xlabel(nome_x)
    plt.ylabel(nome_y)
    plt.title("Gráfico de Série Temporal")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return imagem_base64



def grafico_ic_media(df, colunas_usadas, coluna_y=None):
    if len(colunas_usadas) != 2:
        raise ValueError("O gráfico de intervalos requer uma coluna Y (numérica) e uma coluna X (categórica).")

    nome_y = colunas_usadas[0]  # ✅ Agora Y vem primeiro
    nome_x = colunas_usadas[1]

    # Força conversão da coluna Y para numérico
    df[nome_y] = pd.to_numeric(df[nome_y], errors="coerce")
    if df[nome_y].isnull().all():
        raise ValueError(f"A coluna '{nome_y}' não contém valores numéricos válidos para calcular a média.")

    aplicar_estilo_minitab()

    grupos = df.groupby(nome_x)[nome_y]
    medias = grupos.mean()
    desvios = grupos.std()
    n = grupos.count()
    erro_padrao = desvios / n**0.5
    intervalo = 1.96 * erro_padrao

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        x=medias.index,
        y=medias.values,
        yerr=intervalo.values,
        fmt='o',
        capsize=6,
        elinewidth=2,
        marker='o',
        color='midnightblue'
    )

    plt.title("Intervalos de Confiança (IC 95%) para a Média")
    plt.xlabel(nome_x)
    plt.ylabel(nome_y)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return imagem_base64


def grafico_dispersao(df, colunas):
    if len(colunas) < 2:
        raise ValueError("Gráfico de dispersão requer exatamente duas colunas.")
    plt.figure(figsize=(8, 6))
    aplicar_estilo_minitab()
    sns.scatterplot(x=df[colunas[0]], y=df[colunas[1]])
    plt.title("Gráfico de Dispersão")
    return salvar_grafico()

def grafico_boxplot_simples(df, colunas, coluna_y=None):
    if not coluna_y:
        raise ValueError("Para o boxplot simples, a coluna Y (numérica) é obrigatória.")

    y = df[coluna_y].astype(str).str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(y) < 2:
        raise ValueError("Coluna Y deve conter ao menos dois valores numéricos.")

    df_box = pd.DataFrame({coluna_y: y, "grupo": "A"})

    plt.figure(figsize=(6, 6))
    aplicar_estilo_minitab()
    sns.boxplot(data=df_box, x="grupo", y=coluna_y, color="#89CFF0", width=0.3)
    sns.pointplot(data=df_box, x="grupo", y=coluna_y, estimator=np.mean,
                  markers="D", color="red", scale=1.2, errwidth=0)
    plt.xlabel("")
    plt.ylabel(coluna_y)
    plt.title("Boxplot Simples com Média (losango)")
    return salvar_grafico()


def grafico_boxplot_multiplo(df, colunas, coluna_y=None):
    if not coluna_y or not coluna_y.strip():
        raise ValueError("Você deve selecionar uma coluna Y com valores numéricos para o boxplot múltiplo.")

    coluna_y = coluna_y.strip()
    if coluna_y.startswith("Unnamed") or coluna_y not in df.columns:
        raise ValueError(f"A coluna Y '{coluna_y}' não tem título válido ou não foi encontrada.")

    y = df[coluna_y].astype(str).str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    y = pd.to_numeric(y, errors="coerce")
    if y.dropna().shape[0] < 2:
        raise ValueError("A coluna Y deve conter ao menos dois valores numéricos válidos.")

    colunas = [c.strip() for c in colunas if c.strip() and c.strip() != coluna_y and not c.strip().startswith("Unnamed")]
    if not colunas:
        raise ValueError("Nenhuma coluna X válida foi selecionada para o agrupamento.")

    x_col = colunas[0]
    if x_col not in df.columns:
        raise ValueError(f"A coluna X '{x_col}' não foi encontrada no arquivo.")

    grupo = df[x_col].astype(str)
    df_plot = pd.DataFrame({coluna_y: y, x_col: grupo}).dropna()

    if df_plot.empty:
        raise ValueError("Os dados da coluna Y e do grupo X selecionado não têm valores válidos simultaneamente.")

    plt.figure(figsize=(10, 6))
    aplicar_estilo_minitab()

    sns.boxplot(data=df_plot, x=x_col, y=coluna_y, color="#89CFF0")
    sns.pointplot(data=df_plot, x=x_col, y=coluna_y, estimator=np.mean,
                  markers="D", color="red", scale=1.1, errwidth=0)

    plt.title(f"Boxplot Múltiplo por '{x_col}'")
    plt.xlabel(x_col)
    plt.ylabel(coluna_y)
    plt.xticks(rotation=45)

    return salvar_grafico()




def grafico_barras_simples(df, colunas_usadas):
    if len(colunas_usadas) != 1:
        raise ValueError("Selecione exatamente uma coluna para o Gráfico de Barras Simples.")

    nome_coluna = colunas_usadas[0]
    serie = df[nome_coluna].dropna()
    contagem = serie.value_counts().sort_index()

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    contagem.plot(kind='bar', color='skyblue', ax=ax)

    ax.set_title(f"Gráfico de Barras - {nome_coluna}")
    ax.set_xlabel(nome_coluna)
    ax.set_ylabel("Frequência")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return imagem_base64


def grafico_barras_agrupado(df, colunas_x, coluna_y=None):
    if not colunas_x or len(colunas_x) < 2:
        raise ValueError("Selecione duas colunas no campo X — a primeira será o eixo X, a segunda o agrupador (cores).")

    coluna_x = colunas_x[0]            # Eixo X = primeiro item do Drop X
    coluna_agrupador = colunas_x[1]    # Agrupador (cor) = segundo item do Drop X

    if coluna_x not in df.columns:
        raise ValueError(f"A coluna '{coluna_x}' não existe no DataFrame.")
    if coluna_agrupador not in df.columns:
        raise ValueError(f"A coluna '{coluna_agrupador}' não existe no DataFrame.")

    aplicar_estilo_minitab()

    # Tabela cruzada
    tabela = df.groupby([coluna_x, coluna_agrupador]).size().unstack(fill_value=0)

    # Gráfico com barras agrupadas
    fig, ax = plt.subplots(figsize=(10, 6))
    largura_barra = 0.8 / len(tabela.columns)
    posicoes = np.arange(len(tabela))

    for i, categoria in enumerate(tabela.columns):
        valores = tabela[categoria].values
        ax.bar(posicoes + i * largura_barra, valores, width=largura_barra, label=str(categoria))

    ax.set_xticks(posicoes + largura_barra * (len(tabela.columns) - 1) / 2)
    ax.set_xticklabels(tabela.index, rotation=45)

    ax.set_title(f'Gráfico de Barras Agrupado: {coluna_x} por {coluna_agrupador}')
    ax.set_xlabel(coluna_x)
    ax.set_ylabel("Frequência")
    ax.legend(title=coluna_agrupador)
    plt.tight_layout()

    # Exporta imagem como base64
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return imagem_base64


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
    
    "Gráfico de tendecias": grafico_linha_temporal,
    "grafico_ic_media": grafico_ic_media,
    "Gráfico de disperao": grafico_dispersao,
    "BoxPlot simples": grafico_boxplot_simples,
    "boxplot_multiplo": grafico_boxplot_multiplo,
    "Gráfifo de barras": grafico_barras_simples,
    "Gráfifo de barras com subgrupo": grafico_barras_agrupado,

    
  
    


}
